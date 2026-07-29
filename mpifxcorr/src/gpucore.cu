#include "gpucore.cuh"
#include "gpumode.cuh"
#include "alert.h"
#include "gpumode_kernels.cuh"
#include <thread>
//#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>

using namespace std::chrono;

// One place for byte->MB formatting in log messages.
static const double MB = 1024.0 * 1024.0;

// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
// todo: deduplicate this function
__device__ void atomicAddFloatComplex1(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    float *x = (float*)a;
    float *y = x+1;
    //use atomicAdd for double variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}

/**
 * @brief Fused Cross-Multiply-Accumulate (XMAC) and Frequency Averaging Kernel
 * * Unlike the CPU `threadcrosscorrs` loop which chunks data by `xmacstridelen`
 * for cache locality, this kernel relies on coalesced memory access. Threads
 * are mapped to specific, final *output* channels. Each thread traverses the
 * time-domain (FFT buffers) and the fine-frequency domain (channels to average),
 * collapsing them into a single point.
 *
 * The output averaged-channel dimension is spread across BOTH the block
 * (threadIdx.x) and the grid (blockIdx.z) so that the launch produces enough
 * blocks to fill the device, rather than the numbaselines*numpol (~handful)
 * blocks of the previous grid-stride design.
 *
 * When baselines*pols*channels alone cannot fill the device (the geodesy
 * regime: 1-6 baselines, often 1 pol, modest channel counts), the FFT/time
 * dimension is also split across blockIdx.z: the host divides the
 * numBufferedFFTs integration into chunks of fftsPerChunk FFTs, and each block
 * accumulates a partial sum over its chunk. With a single chunk every
 * (baseline, pol, avg_chan) maps to a unique slot in the pre-zeroed results
 * buffer, so the thread writes its total with a plain store; with multiple
 * chunks the partial sums are combined with an atomic add into the pre-zeroed
 * slot (the extra atomic traffic is one add per chunk, only taken when the
 * grid would otherwise be too small to occupy the device anyway).
 */
__global__ void gpu_fuse_xmac_and_average(
        const cuFloatComplex* const * const gpuM1Freqs,
        const cuFloatComplex* const * const gpuM2Freqs,
        const bool* const * const gpuM1Valid,
        const bool* const * const gpuM2Valid,
        const int* const stream1BandIndexes,
        const int* const stream2BandIndexes,
        const int* const coreResultBaselineOffsets,
        cuFloatComplex* const results_gpu,
        int numbaselines,
        int numPolarisationProducts,
        int numBufferedFFTs,
        int fftsPerChunk,
        int num_averaged_channels,
        int channelstoaverage,
        const int* const stream1BandStride,
        const int* const stream1WindowStride,
        const int* const stream2BandStride,
        const int* const stream2WindowStride
) {
    // gridDim.x  = numbaselines
    // gridDim.y  = numPolarisationProducts
    // gridDim.z  = numChanBlocks * numFftChunks, where
    //              numChanBlocks = ceil(num_averaged_channels / blockDim.x)
    //              numFftChunks  = ceil(numBufferedFFTs / fftsPerChunk)
    // blockDim.x = output averaged channels processed per block (<= 256)

    int baseline = blockIdx.x;
    int pol = blockIdx.y;

    // Unpack the combined (FFT chunk, channel block) z index.
    int numChanBlocks = (num_averaged_channels + blockDim.x - 1) / blockDim.x;
    int fftChunk = blockIdx.z / numChanBlocks;
    int chanBlock = blockIdx.z % numChanBlocks;

    // Each thread owns exactly one output averaged channel within its FFT chunk.
    int avg_chan = chanBlock * blockDim.x + threadIdx.x;
    if (avg_chan >= num_averaged_channels) return;

    // The range of buffered FFTs this block integrates over.
    int fft_begin = fftChunk * fftsPerChunk;
    int fft_end = fft_begin + fftsPerChunk;
    if (fft_end > numBufferedFFTs) fft_end = numBufferedFFTs;

    // 1. Resolve Band Indexes
    // We pre-flattened the 2D [baseline][pol] config arrays on the host.
    int b1 = stream1BandIndexes[baseline * numPolarisationProducts + pol];
    int b2 = stream2BandIndexes[baseline * numPolarisationProducts + pol];

    // If the baseline doesn't participate in this frequency, it was flagged as -1.
    if (b1 < 0 || b2 < 0) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    const bool* valid1 = gpuM1Valid[baseline];
    const bool* valid2 = gpuM2Valid[baseline];

    // 2. Time Integration
    // Summation over this block's chunk of the FFTs residing in the GPU buffer.
    for (int fft = fft_begin; fft < fft_end; fft++) {

        // Skip FFT windows that were invalid on either datastream: the mode
        // kernels leave their spectra stale/uninitialised in the FFT buffers.
        // The CPU path zeroes such spectra, so skipping matches it exactly
        // (zero contribution, and the data weight already excludes them).
        if (!valid1[fft] || !valid2[fft]) continue;

        // Calculate the absolute starting index for this specific FFT and Band,
        // using each stream's OWN buffer strides (they differ on mixed
        // real x complex baselines).
        int base_idx1 = fft * stream1WindowStride[baseline] + b1 * stream1BandStride[baseline];
        int base_idx2 = fft * stream2WindowStride[baseline] + b2 * stream2BandStride[baseline];

        // 3. Frequency Integration
        // Collapse 'channelstoaverage' fine channels into this single output channel.
        for (int c = 0; c < channelstoaverage; c++) {
            int fine_chan = avg_chan * channelstoaverage + c;

            cuFloatComplex v1 = gpuM1Freqs[baseline][base_idx1 + fine_chan];
            // Note: v2 is pulled from conj_fftd_gpu, so it is already conjugated.
            cuFloatComplex v2 = gpuM2Freqs[baseline][base_idx2 + fine_chan];

            // Cross-multiply: V1 * V2*
            sum = cuCaddf(sum, cuCmulf(v1, v2));
        }
    }

    // 4. Apply Scaling
    // Time integration in DiFX is a pure SUM, but frequency integration is a MEAN.
    // We multiply by the reciprocal to avoid expensive division operations.
    float scale = 1.0f / (float)channelstoaverage;
    sum.x *= scale;
    sum.y *= scale;

    // 5. Direct Output Mapping
    // We completely bypass the CPU's intermediate `threadcrosscorrs` array.
    // The layout of procslots[index].results is contiguous: [Baseline][Polarisation][Channel]
    int base_offset = coreResultBaselineOffsets[baseline];
    int pol_offset = pol * num_averaged_channels;
    int final_index = base_offset + pol_offset + avg_chan;

    if (fftsPerChunk >= numBufferedFFTs) {
        // Single FFT chunk: this (baseline, pol, avg_chan) triple is unique
        // within the launch and the results buffer was pre-zeroed, so a plain
        // store is correct and avoids the serialisation of an atomic add.
        results_gpu[final_index] = sum;
    } else {
        // Multiple FFT chunks contribute partial sums to the same pre-zeroed
        // output slot; combine them atomically.
        atomicAddFloatComplex1(&results_gpu[final_index], sum);
    }
}

// Increment 2: per-subint baseline-weight reduction on the device. One thread
// per accumulator (freq, baseline, polproduct) sums dw1[w]*dw2[w] over the
// subint's FFT windows, replacing the host per-window baseline-weight loop that
// used to run in host_accumulate. dw1/dw2 point at the two datastreams'
// device per-window dataweight arrays (GPUMode::getGpuDataWeights). Windows
// beyond the subint carry weight 0, so summing [0, numWindows) is exact. The
// sequential accumulation reproduces the CPU loop's window order; weights are
// 0/1 except at the subint's start/end frames, so the sums agree with the host
// path to floating-point precision.
__global__ void gpu_baseline_weights(const float* const* dw1,
                                     const float* const* dw2,
                                     float* out, int numAccum, int numWindows) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= numAccum)
        return;
    const float* w1 = dw1[a];
    const float* w2 = dw2[a];
    float s = 0.0f;
    for (int w = 0; w < numWindows; w++)
        s += w1[w] * w2[w];
    out[a] = s;
}

GPUCore::GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm)
        : Core(id, conf, dids, rcomm) {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    cudaMultiProcessorCount = prop.multiProcessorCount;
    if(numprocessthreads > 1) {
      cerr << "GPU DiFX must have 1 thread per Core process - had " << numprocessthreads << endl;
      exit(1);
    }

    // Initialize the compute stream and share it with every GPUMode this Core
    // will construct, so all station processing and the XMAC form a single
    // in-order queue. This must happen before any Mode is built (Modes are
    // constructed later, in loopprocess -> updateconfig -> getMode).
    checkCuda(cudaStreamCreate(&cuStream));
    GPUMode::setSharedComputeStream(cuStream);
    // Separate stream for the visibility transfer back to the host, so it can
    // run concurrently with the next subintegration's compute on cuStream.
    checkCuda(cudaStreamCreate(&d2hStream));

    // Pipelining is on by default; DIFX_GPU_PIPELINE=0 forces the synchronous
    // (complete-immediately) path for A/B comparison.
    pipeline = true;
    {
        const char *e = getenv("DIFX_GPU_PIPELINE");
        if (e != NULL && atoi(e) == 0)
            pipeline = false;
    }
    // The DIFX_GPU_WEIGHTS_HOST fallback computes this subint's weights on the
    // host in issue_tofft (set_weights) and copies its autocorrelations in
    // issue_afterfft, both consumed by the host tail. Tail-overlap pipelining
    // inserts the NEXT subint's issue_tofft between afterfft and complete, which
    // would clobber those shared host mirrors - so force the synchronous path
    // when the device-weights path is off. (It is a debug/comparison path; it
    // does not need the overlap.)
    if (!GPUMode::useGpuWeights())
        pipeline = false;
    if (mpiid == numdatastreams + fxcorr::FIRSTTELESCOPEID)
        cinfo << startl << "GPU Core visibility-transfer pipelining is "
              << (pipeline ? "ENABLED" : "disabled") << endl;

    // Per-window WDEBUG (DIFX_WEIGHT_DEBUG) prints Mode scalars (datasec/datans/
    // nearestSamples/validflags) that, with pipelining on, the tail reads AFTER
    // the next subint's tofft has overwritten them - so the values would be
    // wrong. WDEBUG is a debug-only tool (slated for removal); require the
    // synchronous, no-overlap path so it stays exact by construction.
    if (pipeline && getenv("DIFX_WEIGHT_DEBUG") != NULL) {
        cfatal << startl << "DIFX_WEIGHT_DEBUG requires DIFX_GPU_PIPELINE=0 on the GPU path "
               << "(per-window weight debug is not valid while tail-overlap pipelining is on)"
               << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Page-lock the Core receive buffers (procslots[].databuffer[], allocated
    // pageable by the Core constructor) so the GPUModes can cudaMemcpyAsync
    // their input straight from them, eliminating the per-subint host staging
    // memcpy into packeddata_gpu. On by default; DIFX_GPU_PIN_INPUT=0 disables
    // (falling back to the staging path) for A/B comparison. Registration
    // failure (e.g. locked-memory limits) is not fatal - warn, unregister
    // whatever succeeded, and fall back. Like the shared stream above, this
    // must be settled before any GPUMode is constructed (Modes are built later,
    // in loopprocess -> updateconfig).
    bool inputPinned = false;
    bool wantpinned = true;
    {
        const char *e = getenv("DIFX_GPU_PIN_INPUT");
        if (e != NULL && atoi(e) == 0)
            wantpinned = false;
    }
    if (wantpinned) {
        cudaError_t pinerr = cudaSuccess;
        int npinned = 0;
        for (int i = 0; i < RECEIVE_RING_LENGTH && pinerr == cudaSuccess; i++) {
            for (int j = 0; j < numdatastreams && pinerr == cudaSuccess; j++) {
                // Deliberately not checkCuda - a failure here must not exit.
                pinerr = cudaHostRegister(procslots[i].databuffer[j], databytes,
                                          cudaHostRegisterDefault);
                if (pinerr == cudaSuccess)
                    npinned++;
            }
        }
        if (pinerr == cudaSuccess) {
            inputPinned = true;
            if (mpiid == numdatastreams + fxcorr::FIRSTTELESCOPEID)
                cinfo << startl << "GPU Core " << mpiid << " pinned "
                      << (RECEIVE_RING_LENGTH * numdatastreams * (double)databytes) / MB
                      << " MB of receive buffers for direct H2D input transfers" << endl;
        } else {
            cwarn << startl << "GPU Core " << mpiid << " could not page-lock the receive"
                  << " buffers (" << cudaGetErrorString(pinerr) << ") - falling back to"
                  << " staged input transfers" << endl;
            // Undo the registrations that did succeed (registration order is
            // row-major over (slot, stream)), and clear the sticky error so
            // later checkCuda calls don't trip over it.
            for (int k = 0; k < npinned; k++)
                cudaHostUnregister(procslots[k / numdatastreams].databuffer[k % numdatastreams]);
            cudaGetLastError();
        }
    } else if (mpiid == numdatastreams + fxcorr::FIRSTTELESCOPEID) {
        cinfo << startl << "GPU Core input-buffer pinning disabled by DIFX_GPU_PIN_INPUT" << endl;
    }
    GPUMode::setInputBuffersPinned(inputPinned);

    // Fail fast if this job cannot fit in device memory, before the large
    // per-Mode buffers start being allocated inside loopprocess.
    checkDeviceMemory();

    // Allocate one results buffer and one completion event per procslot, so the
    // deferred device->host copy of one subint does not collide with the next
    // subint's XMAC writes.
    // evComputeDone is recorded on cuStream after this subint's afterfft + XMAC +
    // output D2Hs; d2hStream waits on it before the visibility D2H, and it lets
    // the tail-overlap pipeline replace the old end-of-subint
    // cudaStreamSynchronize. validsubint[ds] captures each datastream's subint
    // validity at issue time (isSubintValid()), because the pipelined next-subint
    // tofft overwrites the Mode's datalengthbytes/offsetseconds before the
    // deferred tail folds this subint's weights/autocorrs.
    gpuprocslots.resize(RECEIVE_RING_LENGTH);
    for (int i = 0; i < RECEIVE_RING_LENGTH; i++) {
        gpuprocslot &gs = gpuprocslots[i];
        gs.validsubint.assign(numdatastreams, 0);
        checkCuda(cudaMalloc(&gs.results_gpu, maxcoreresultlength * sizeof(cuFloatComplex)));
        checkCuda(cudaMallocHost(&gs.results_host, maxcoreresultlength * sizeof(cuFloatComplex)));
        checkCuda(cudaEventCreateWithFlags(&gs.d2hDone, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&gs.h2dInputDone, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&gs.evComputeDone, cudaEventDisableTiming));
    }

    // Allocate the shared, frequency-independent FFT buffer pointer arrays (one
    // entry per baseline). These are populated once per configuration by
    // buildXmacPlans(). The per-frequency band-index/offset arrays are allocated
    // lazily inside buildXmacPlans() and cached in xmacPlans.
    checkCuda(cudaMalloc(&d_m1_ptrs, numbaselines * sizeof(cuFloatComplex*)));
    checkCuda(cudaMalloc(&d_m2_ptrs, numbaselines * sizeof(cuFloatComplex*)));
    checkCuda(cudaMalloc(&d_v1_ptrs, numbaselines * sizeof(bool*)));
    checkCuda(cudaMalloc(&d_v2_ptrs, numbaselines * sizeof(bool*)));
}

void GPUCore::checkDeviceMemory() {
    // Peak device usage is dominated by the per-datastream GPUMode buffers.
    // These are allocated for whichever config a scan uses, and a Core holds
    // one Mode per datastream at a time, so the worst case is the config whose
    // datastreams sum to the most memory. Take the max over configs.
    size_t maxConfigBytes = 0;
    for (int c = 0; c < config->getNumConfigs(); c++) {
        size_t configBytes = 0;
        for (int d = 0; d < numdatastreams; d++)
            configBytes += GPUMode::estimateDeviceBytes(config, c, d);
        if (configBytes > maxConfigBytes)
            maxConfigBytes = configBytes;
    }

    // This Core's own persistent device allocations: the results buffers (one
    // per procslot) plus the per-baseline pointer/validity arrays. The cached
    // XMAC plan arrays are a few ints per baseline per frequency - negligible,
    // but fold in a small allowance so the estimate errs high rather than low.
    size_t coreBytes = (size_t)RECEIVE_RING_LENGTH * maxcoreresultlength * sizeof(cuFloatComplex);
    coreBytes += (size_t)numbaselines * (2 * sizeof(cuFloatComplex*) + 2 * sizeof(bool*));
    coreBytes += (size_t)numbaselines * config->getFreqTableLength() * 16 * sizeof(int);

    const size_t required = maxConfigBytes + coreBytes;

    size_t freeBytes = 0, totalBytes = 0;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));

    // Keep a headroom margin: cuFFT/cuBLAS scratch, allocator fragmentation and
    // driver overhead all consume device memory beyond our own buffers.
    // CAVEAT: if several Core ranks share one physical GPU, each sees only the
    // free memory at its own construction time, not siblings' future
    // allocations - so this is a fail-fast guard against a clearly-too-big job,
    // not a guarantee that a marginal one will fit.
    const double SAFETY = 0.90;
    const size_t usable = (size_t)(freeBytes * SAFETY);

    cinfo << startl << "GPU Core " << mpiid << " estimated device memory need "
          << required / MB << " MB (modes " << maxConfigBytes / MB << " MB + core "
          << coreBytes / MB << " MB); device reports " << freeBytes / MB << " MB free of "
          << totalBytes / MB << " MB, usable after " << (int)((1.0 - SAFETY) * 100)
          << "% headroom " << usable / MB << " MB" << endl;

    if (required > usable) {
        cfatal << startl << "GPU Core " << mpiid << " needs an estimated " << required / MB
               << " MB of device memory but only " << usable / MB
               << " MB is usable (of " << freeBytes / MB << " MB free) - this job will not fit"
               << " on the GPU. Reduce the number of channels/bands, the subint length"
               << " (NUM CHANNELS / BLOCKS PER SEND), or the datastreams per Core, or use a"
               << " GPU with more memory. Aborting before allocation." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static unsigned long long avg_preprocess;
static unsigned long long avg_postprocess;
int core_calls;
GPUCore::~GPUCore() {
    if (core_calls > 0) {
        cout << "Average core pre: " << avg_preprocess / core_calls << endl;
        cout << "Average core post: " << avg_postprocess / core_calls << endl;
    }

    // Deliberately make NO CUDA calls here. This destructor only runs at
    // process teardown (end of main, after MPI_Finalize), by which point the
    // CUDA driver may already have destroyed the context via its own atexit
    // handlers - and calls like cudaStreamDestroy on the now-stale handles
    // don't just return an error, they can segfault inside libcuda (observed
    // with CUDA 12.8). The process is exiting and the driver reclaims all
    // device memory and streams itself, so freeing results_gpu, the pointer
    // arrays, the cached XMAC plans and cuStream here would gain nothing.
    // For the same reason there is no cudaHostUnregister of the pinned
    // procslots receive buffers and no destruction of the h2dInputDone/d2hDone
    // events here - the driver unwinds those registrations at process exit.
}

void GPUCore::freeXmacPlans() {
    // Only called while the CUDA context is alive (from buildXmacPlans on a
    // configuration change) - NOT from the destructor; see ~GPUCore.
    for (auto &plan : xmacPlans) {
        checkCuda(cudaFree(plan.d_stream1BandIndexes));
        checkCuda(cudaFree(plan.d_stream2BandIndexes));
        checkCuda(cudaFree(plan.d_coreResultBaselineOffsets));
        checkCuda(cudaFree(plan.d_stream1BandStride));
        checkCuda(cudaFree(plan.d_stream1WindowStride));
        checkCuda(cudaFree(plan.d_stream2BandStride));
        checkCuda(cudaFree(plan.d_stream2WindowStride));
    }
    xmacPlans.clear();
    xmacPlanConfigIndex = -1;

    // Baseline-weight reduction plan (Increment 2) is rebuilt alongside the
    // XMAC plans, so release its device/pinned buffers here too.
    if (d_bwDw1) { checkCuda(cudaFree(d_bwDw1)); d_bwDw1 = nullptr; }
    if (d_bwDw2) { checkCuda(cudaFree(d_bwDw2)); d_bwDw2 = nullptr; }
    if (d_bweightResults) { checkCuda(cudaFree(d_bweightResults)); d_bweightResults = nullptr; }
    if (h_bweightResults) { checkCuda(cudaFreeHost(h_bweightResults)); h_bweightResults = nullptr; }
    bwDestOffset.clear();
    bweightNumAccum = 0;
}

// Precompute, once per configuration, all the DiFX config-tree metadata that the
// fused XMAC kernel needs. This is invariant for a given configindex, so caching
// it removes the per-subintegration host-side config walk and the redundant
// device uploads that used to happen on every call to processgpudata().
void GPUCore::buildXmacPlans(int configindex, Mode **modes) {
    // Drop any previously-cached plans (e.g. on a configuration change).
    freeXmacPlans();

    // The FFT buffer pointers depend only on the baseline's datastreams (not on
    // frequency) and the GPUMode buffers never move, so they are gathered and
    // uploaded once here and shared across all per-frequency launches.
    const cuFloatComplex* h_m1_ptrs[numbaselines];
    const cuFloatComplex* h_m2_ptrs[numbaselines];
    const bool* h_v1_ptrs[numbaselines];
    const bool* h_v2_ptrs[numbaselines];
    for (int j = 0; j < numbaselines; j++) {
        int ds1index = config->getBOrderedDataStream1Index(configindex, j);
        int ds2index = config->getBOrderedDataStream2Index(configindex, j);
        h_m1_ptrs[j] = ((GPUMode*)modes[ds1index])->fftd_gpu->gpuPtr();
        h_m2_ptrs[j] = ((GPUMode*)modes[ds2index])->conj_fftd_gpu->gpuPtr();
        h_v1_ptrs[j] = ((GPUMode*)modes[ds1index])->getGpuValidSamples();
        h_v2_ptrs[j] = ((GPUMode*)modes[ds2index])->getGpuValidSamples();
    }
    checkCuda(cudaMemcpyAsync(d_m1_ptrs, h_m1_ptrs, numbaselines * sizeof(cuFloatComplex*),
                             cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(d_m2_ptrs, h_m2_ptrs, numbaselines * sizeof(cuFloatComplex*),
                             cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(d_v1_ptrs, h_v1_ptrs, numbaselines * sizeof(bool*),
                             cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(d_v2_ptrs, h_v2_ptrs, numbaselines * sizeof(bool*),
                             cudaMemcpyHostToDevice, cuStream));

    for (int f = 0; f < config->getFreqTableLength(); f++) {
        if (!config->isFrequencyUsed(configindex, f)) continue;

        int freqchannels = config->getFNumChannels(f);
        int channelstoaverage = config->getFChannelsToAverage(f);

        int numPolarisationProducts = 0;
        // Temprorary variable to hold the local frequency index of the reference baseline.  
        // varible localfreqindex is used later within a loop (same value)
        int localfreqindex_ref = -1;

        // Find the first baseline that has a local frequency index to use as a reference.
        for (int ref_baseline = 0; ref_baseline < numbaselines; ++ref_baseline) {
            localfreqindex_ref = config->getBLocalFreqIndex(configindex, ref_baseline, f);
            if (localfreqindex_ref >= 0) {
                // This baseline is active, use it to get the number of pol products.
                numPolarisationProducts = config->getBNumPolProducts(configindex, ref_baseline, localfreqindex_ref);
                break; // Exit the loop since we found a valid reference to obtain numPolarisationProducts.
            }
        }

        XmacFreqPlan plan;

        // It is a fatal error if the frequency is marked 'used' but we found no active
        // baselines or pol products for it.
        if (numPolarisationProducts == 0) {
            cerror << startl << "Error in buildXmacPlans: Frequency " << f 
                   << " is marked as used, but no active baselines/polarisation products were found for it." 
                   << endl;
            // Depending on desired robustness, you could 'continue' to the next frequency
            // or abort as the original code intended. Aborting is safer.
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        plan.numPolarisationProducts = numPolarisationProducts;
        plan.num_averaged_channels = freqchannels / channelstoaverage;
        plan.channelstoaverage = channelstoaverage;

        // Gather the per-baseline band indexes and result offsets for this frequency.
        int h_stream1BandIndexes[numbaselines * numPolarisationProducts];
        int h_stream2BandIndexes[numbaselines * numPolarisationProducts];
        int h_coreResultBaselineOffsets[numbaselines];
        // Per-baseline, per-stream buffer strides: each side's GPUMode fftd
        // buffer geometry follows THAT datastream's sampling type and band
        // count, so a mixed (e.g. real x complex) baseline has different
        // strides on its two sides.
        int h_s1BandStride[numbaselines];
        int h_s1WindowStride[numbaselines];
        int h_s2BandStride[numbaselines];
        int h_s2WindowStride[numbaselines];
        for (int j = 0; j < numbaselines; j++) {
            int ds1index = config->getBOrderedDataStream1Index(configindex, j);
            int ds2index = config->getBOrderedDataStream2Index(configindex, j);
            int mult1 = (config->getDSampling(configindex, ds1index) == Configuration::COMPLEX) ? 1 : 2;
            int mult2 = (config->getDSampling(configindex, ds2index) == Configuration::COMPLEX) ? 1 : 2;
            h_s1BandStride[j] = freqchannels * mult1;
            h_s1WindowStride[j] = h_s1BandStride[j] * config->getDNumRecordedBands(configindex, ds1index);
            h_s2BandStride[j] = freqchannels * mult2;
            h_s2WindowStride[j] = h_s2BandStride[j] * config->getDNumRecordedBands(configindex, ds2index);
            int localfreqindex = config->getBLocalFreqIndex(configindex, j, f);
            if (localfreqindex >= 0) {
                h_coreResultBaselineOffsets[j] =
                    config->getCoreResultBaselineOffset(configindex, f, j);
                for (int p = 0; p < numPolarisationProducts; p++) {
                    h_stream1BandIndexes[j * numPolarisationProducts + p] =
                        config->getBDataStream1BandIndex(configindex, j, localfreqindex, p);
                    h_stream2BandIndexes[j * numPolarisationProducts + p] =
                        config->getBDataStream2BandIndex(configindex, j, localfreqindex, p);
                }
            } else {
                // Baseline doesn't participate in this frequency; flag with -1 so
                // the kernel early-returns for it.
                h_coreResultBaselineOffsets[j] = -1;
                for (int p = 0; p < numPolarisationProducts; p++) {
                    h_stream1BandIndexes[j * numPolarisationProducts + p] = -1;
                    h_stream2BandIndexes[j * numPolarisationProducts + p] = -1;
                }
            }
        }

        // Allocate persistent device arrays and upload the gathered metadata.
        checkCuda(cudaMalloc(&plan.d_stream1BandIndexes,
                             numbaselines * numPolarisationProducts * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_stream2BandIndexes,
                             numbaselines * numPolarisationProducts * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_coreResultBaselineOffsets, numbaselines * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_stream1BandStride, numbaselines * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_stream1WindowStride, numbaselines * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_stream2BandStride, numbaselines * sizeof(int)));
        checkCuda(cudaMalloc(&plan.d_stream2WindowStride, numbaselines * sizeof(int)));
        checkCuda(cudaMemcpyAsync(plan.d_stream1BandIndexes, h_stream1BandIndexes,
                                 numbaselines * numPolarisationProducts * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream2BandIndexes, h_stream2BandIndexes,
                                 numbaselines * numPolarisationProducts * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_coreResultBaselineOffsets, h_coreResultBaselineOffsets,
                                 numbaselines * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream1BandStride, h_s1BandStride,
                                 numbaselines * sizeof(int), cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream1WindowStride, h_s1WindowStride,
                                 numbaselines * sizeof(int), cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream2BandStride, h_s2BandStride,
                                 numbaselines * sizeof(int), cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream2WindowStride, h_s2WindowStride,
                                 numbaselines * sizeof(int), cudaMemcpyHostToDevice, cuStream));

        xmacPlans.push_back(plan);
    }

    // Baseline-weight reduction plan (Increment 2). Enumerate the accumulators
    // in the exact order the host finalize fold consumes them (used freq ->
    // baseline with localfreqindex>=0 -> polproduct), gathering each baseline's
    // two datastreams' device dataweight arrays. gDataWeights buffers are
    // allocated once per GPUMode and never move, so this is config-invariant
    // like the XMAC plans above. (freeXmacPlans, called at the top, already
    // released any previous config's buffers.)
    std::vector<const float*> h_bwDw1, h_bwDw2;
    for (int f = 0; f < config->getFreqTableLength(); f++) {
        if (!config->isFrequencyUsed(configindex, f)) continue;
        for (int l = 0; l < numbaselines; l++) {
            int localfreqindex = config->getBLocalFreqIndex(configindex, l, f);
            if (localfreqindex < 0) continue;
            int ds1index = config->getBOrderedDataStream1Index(configindex, l);
            int ds2index = config->getBOrderedDataStream2Index(configindex, l);
            const float* dw1 = ((GPUMode*)modes[ds1index])->getGpuDataWeights();
            const float* dw2 = ((GPUMode*)modes[ds2index])->getGpuDataWeights();
            int npol = config->getBNumPolProducts(configindex, l, localfreqindex);
            // Destination of this baseline's bin-0 weights in floatresults; pol p
            // lands at base + p (the host fold's resultindex++ walk for bin 0).
            int bwbase = config->getCoreResultBWeightOffset(configindex, f, l) * 2;
            for (int p = 0; p < npol; p++) {
                h_bwDw1.push_back(dw1);
                h_bwDw2.push_back(dw2);
                bwDestOffset.push_back(bwbase + p);
            }
        }
    }
    bweightNumAccum = (int)h_bwDw1.size();
    if (bweightNumAccum > 0) {
        checkCuda(cudaMalloc(&d_bwDw1, bweightNumAccum * sizeof(float*)));
        checkCuda(cudaMalloc(&d_bwDw2, bweightNumAccum * sizeof(float*)));
        checkCuda(cudaMalloc(&d_bweightResults, bweightNumAccum * sizeof(float)));
        checkCuda(cudaMallocHost(&h_bweightResults, bweightNumAccum * sizeof(float)));
        checkCuda(cudaMemcpyAsync(d_bwDw1, h_bwDw1.data(),
                                 bweightNumAccum * sizeof(float*),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(d_bwDw2, h_bwDw2.data(),
                                 bweightNumAccum * sizeof(float*),
                                 cudaMemcpyHostToDevice, cuStream));
    }

    // The uploads above source from host stack/vector buffers that go out of
    // scope when this function returns; synchronise so the (one-off) copies
    // complete first.
    checkCuda(cudaStreamSynchronize(cuStream));

    xmacPlanConfigIndex = configindex;
}

void GPUCore::loopprocess(int threadid) {
    int perr, numprocessed, startblock, numblocks, lastconfigindex, numpolycos, maxchan, maxpolycos, stadumpchannels, strideplussteplen, maxrotatestrideplussteplength, maxxmaclength, slen;
    double sec;
    bool pulsarbin, somepulsarbin, somescrunch, dumpingsta, nowdumpingsta;
    Polyco **polycos = 0;
    Polyco *currentpolyco = 0;
    Mode **modes;
    threadscratchspace *scratchspace = new threadscratchspace;
    scratchspace->shifterrorcount = 0;
    scratchspace->threadcrosscorrs = vectorAlloc_cf32(maxthreadresultlength);
    scratchspace->baselineweight = new f32 ***[config->getFreqTableLength()];
    scratchspace->baselineshiftdecorr = new f32 **[config->getFreqTableLength()];
    if (scratchspace->threadcrosscorrs == NULL) {
        cfatal << startl << "Could not allocate thread cross corr space (tried to allocate "
               << maxthreadresultlength / (1024 * 1024) << " MB)!!! Aborting." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    scratchspace->pulsarscratchspace = 0;
    scratchspace->pulsaraccumspace = 0;
    scratchspace->starecordbuffer = 0;

    pulsarbin = false;
    somepulsarbin = false;
    somescrunch = false;
    dumpingsta = false;
    maxpolycos = 0;
    maxchan = config->getMaxNumChannels();
    slen = config->getRotateStrideLength(0);
    maxrotatestrideplussteplength = slen + maxchan / slen;
    maxxmaclength = config->getXmacStrideLength(0);
    for (int i = 1; i < config->getNumConfigs(); i++) {
        slen = config->getRotateStrideLength(i);
        strideplussteplen = slen + maxchan / slen;
        if (strideplussteplen > maxrotatestrideplussteplength)
            maxrotatestrideplussteplength = strideplussteplen;
        if (config->getXmacStrideLength(i) > maxxmaclength)
            maxxmaclength = config->getXmacStrideLength(i);
    }
    scratchspace->chanfreqs = vectorAlloc_f64(maxrotatestrideplussteplength);
    scratchspace->rotator = vectorAlloc_cf32(maxrotatestrideplussteplength);
    scratchspace->rotated = vectorAlloc_cf32(maxchan);
    scratchspace->channelsums = vectorAlloc_cf32(maxchan);
    scratchspace->argument = vectorAlloc_f32(3 * maxrotatestrideplussteplength);
    // FIXME: explicitly calculate "28" below.
    threadbytes[threadid] += 16 * maxchan + 28 * maxrotatestrideplussteplength;

    //work out whether we'll need to do any pulsar binning, and work out the maximum # channels (and # polycos if applicable)
    for (int i = 0; i < config->getNumConfigs(); i++) {
        if (config->pulsarBinOn(i)) {
            somepulsarbin = true;
            somescrunch = somescrunch || config->scrunchOutputOn(i);
            numpolycos = config->getNumPolycos(i);
            if (numpolycos > maxpolycos)
                maxpolycos = numpolycos;
        }
    }

    //create the necessary pulsar scratch space if required
    if (somepulsarbin) {
        scratchspace->pulsarscratchspace = vectorAlloc_cf32(maxxmaclength);
        if (somescrunch) //need separate accumulation space
        {
            scratchspace->pulsaraccumspace = new cf32 ******[config->getFreqTableLength()];
        }
        createPulsarVaryingSpace(scratchspace->pulsaraccumspace, &(scratchspace->bins), procslots[0].configindex, -1,
                                 threadid); //don't need to delete old space
    }

    //create the baselineweight and xmacstrideoffset arrays
    allocateConfigSpecificThreadArrays(scratchspace->baselineweight, scratchspace->baselineshiftdecorr,
                                       procslots[0].configindex, -1, threadid); //don't need to delete old space

    //set to first configuration and set up, creating Modes, Polycos etc
    lastconfigindex = procslots[0].configindex;
    modes = new Mode *[numdatastreams];
    if (somepulsarbin)
        polycos = new Polyco *[maxpolycos];
    updateconfig(lastconfigindex, lastconfigindex, threadid, startblock, numblocks, numpolycos, pulsarbin, modes,
                 polycos, true);
    numprocessed = 0;
//  cinfo << startl << "Core thread id " << threadid << " will be processing from block " << startblock << ", length " << numblocks << endl;

    //lock the end section
    perr = pthread_mutex_lock(&(procslots[RECEIVE_RING_LENGTH - 1].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock mutex "
                << RECEIVE_RING_LENGTH - 1 << endl;

    //grab the lock we really want, unlock the end section and signal the main thread we're ready to go
    perr = pthread_mutex_lock(&(procslots[0].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock mutex 0" << endl;
    perr = pthread_mutex_unlock(&(procslots[RECEIVE_RING_LENGTH - 1].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock mutex "
                << RECEIVE_RING_LENGTH - 1 << endl;
    processthreadinitialised[threadid] = true;
    perr = pthread_cond_signal(&processconds[threadid]);
    if (perr != 0)
        csevere << startl << "Core processthread " << mpiid << "/" << threadid
                << " error trying to signal main thread to wake up!!!" << endl;
    if (threadid == 0)
        cinfo << startl << "Core " << mpiid << " PROCESSTHREAD " << threadid + 1 << "/" << numprocessthreads
              << " is about to start processing" << endl;

    //while valid, process data.
    //
    // Tail-overlap pipeline: process_gpu is split into a first half (tofft: input
    // H2D, unpack, weights, fringe rotation, FFT - issue_tofft) and a second half
    // (afterfft: weight reduction, pcal, fractional rotation, then the fused XMAC
    // + baseline-weight reduction + output D2Hs - issue_afterfft_xmac_drain).
    // Invariant at each loop top: this slot's first half has already been issued
    // (prologue or previous iteration) and its slot lock is held. Each iteration
    // issues this subint's second half, then - before completing it - pre-issues
    // the NEXT subint's first half onto the compute stream, so that first half
    // runs while this subint's outputs drain and its host tail (completegpudata)
    // runs. The pipeline is broken (next tofft deferred until after complete)
    // across a config change, at end of data, and when DIFX_GPU_PIPELINE=0.
    //
    // Per-subint pulsar-polyco + STA setup, then issue_tofft, for one slot.
    auto setupAndTofft = [&](int slot) {
        processslot *cs = &(procslots[slot]);
        if (pulsarbin) {
            sec = double(startseconds + model->getScanStartSec(cs->offsets[0], startmjd, startseconds) +
                         cs->offsets[1]) + ((double) cs->offsets[2]) / 1000000000.0;
            //get the correct Polyco for this time range and set it up correctly
            currentpolyco = Polyco::getCurrentPolyco(cs->configindex, startmjd, sec / 86400.0, polycos,
                                                     numpolycos, false);
            if (currentpolyco == NULL) {
                cfatal << startl << "Could not locate a polyco to cover time " << startmjd + sec / 86400.0
                       << " - aborting!!!" << endl;
                currentpolyco = Polyco::getCurrentPolyco(cs->configindex, startmjd, sec / 86400.0, polycos,
                                                         numpolycos, true);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            currentpolyco->setTime(startmjd, sec / 86400.0);
        }

        //if necessary, allocate/reallocate space for the STAs
        scratchspace->dumpsta = config->dumpSTA();
        scratchspace->dumpkurtosis = config->dumpKurtosis();
        nowdumpingsta = scratchspace->dumpsta || scratchspace->dumpkurtosis;
        if (nowdumpingsta != dumpingsta) {
            if (scratchspace->starecordbuffer != 0) {
                free(scratchspace->starecordbuffer);
                scratchspace->starecordbuffer = 0;
            }
            if (nowdumpingsta) {
                stadumpchannels = config->getSTADumpChannels();
                scratchspace->starecordbuffer = (DifxMessageSTARecord *) malloc(config->getMTU());
                if (sizeof(DifxMessageSTARecord) + sizeof(f32) * stadumpchannels > config->getMTU())
                    cerror << startl << "Can't even fit one DiFXSTAMessage into an MTU! No STA dumping will be possible"
                           << endl;
            }
            dumpingsta = nowdumpingsta;
        }

        issue_tofft(slot, threadid, startblock, numblocks, modes, currentpolyco, scratchspace);
    };

    // Prologue: issue the first subint's first half (its slot lock is held from
    // the initialisation above).
    setupAndTofft(numprocessed % RECEIVE_RING_LENGTH);

    while (procslots[(numprocessed) % RECEIVE_RING_LENGTH].keepprocessing) {
        int index = numprocessed % RECEIVE_RING_LENGTH;

        //second half + XMAC + output drain for this subint (its first half ran in
        //the prologue or the previous iteration)
        issue_afterfft_xmac_drain(index, threadid, startblock, numblocks, modes, currentpolyco, scratchspace);

        //grab the next slot's lock (blocks until the manager has filled it),
        //keeping the manager one slot ahead - exactly as the pre-pipelining code
        //did. If pipelining and the next slot is a real subint with the same
        //config, pre-issue its first half NOW so it runs on the compute stream
        //during this subint's drain + host tail below.
        int nextindex = (index + 1) % RECEIVE_RING_LENGTH;
        perr = pthread_mutex_lock(&(procslots[nextindex].slotlocks[threadid]));
        if (perr != 0)
            csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid
                    << " error trying lock mutex " << nextindex << endl;
        bool nextToffted = false;
        if (pipeline && procslots[nextindex].keepprocessing &&
            procslots[nextindex].configindex == lastconfigindex) {
            setupAndTofft(nextindex);
            nextToffted = true;
        }

        //complete this subint (await its outputs, run the host tail) while the
        //next subint's first half executes on the GPU, then release its lock
        completegpudata(index, threadid, startblock, numblocks, modes, currentpolyco, scratchspace);
        perr = pthread_mutex_unlock(&(procslots[index].slotlocks[threadid]));
        if (perr != 0)
            csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid
                    << " error trying unlock mutex " << index << endl;

        if (threadid == 0)
            numcomplete++;
        numprocessed++;

        //restore the loop-top invariant for the new current slot: handle a config
        //change, and issue its first half if it was not pre-issued above (i.e.
        //pipelining off, a config change, or the slot after a config boundary).
        //We already hold its lock (grabbed above as nextindex).
        int newindex = numprocessed % RECEIVE_RING_LENGTH;
        if (procslots[newindex].keepprocessing) {
            if (procslots[newindex].configindex != lastconfigindex) {
                cinfo << startl << "Core " << mpiid << " threadid " << threadid << ": changing config to "
                      << procslots[newindex].configindex << endl;
                updateconfig(lastconfigindex, procslots[newindex].configindex, threadid, startblock, numblocks,
                             numpolycos, pulsarbin, modes, polycos, false);
                cinfo << startl << "Core " << mpiid << " threadid " << threadid
                      << ": config changed successfully - pulsarbin is now " << pulsarbin << endl;
                createPulsarVaryingSpace(scratchspace->pulsaraccumspace, &(scratchspace->bins),
                                         procslots[newindex].configindex, lastconfigindex, threadid);
                allocateConfigSpecificThreadArrays(scratchspace->baselineweight, scratchspace->baselineshiftdecorr,
                                                   procslots[newindex].configindex, lastconfigindex, threadid);
                lastconfigindex = procslots[newindex].configindex;
            }
            if (!nextToffted)
                setupAndTofft(newindex);
        }
    }

    //fallen out of loop, so must be finished. completegpudata ran inside the loop
    //for every real subint, so nothing is left in flight - we only still hold the
    //lock on the current (terminator) slot, whose first half was never issued.
    perr = pthread_mutex_unlock(&(procslots[numprocessed % RECEIVE_RING_LENGTH].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock mutex "
                << (numprocessed) % RECEIVE_RING_LENGTH << endl;

    //free resources

    std::cout << "Deleting modes for each datastream." << std::endl;
    std::cout << "Just quit now!" << std::endl;
    //exit(0);
    for (int j = 0; j < numdatastreams; j++)
        delete modes[j];
    std::cout << "deleting again????" << std::endl;
    delete[] modes;
    std::cout << "done deleting agian" << std::endl;
    if (somepulsarbin) {
        if (threadid > 0 && pulsarbin) {
            for (int i = 0; i < numpolycos; i++)
                delete polycos[i];
        }
        delete[] polycos;
        vectorFree(scratchspace->pulsarscratchspace);
        createPulsarVaryingSpace(scratchspace->pulsaraccumspace, &(scratchspace->bins), -1,
                                 procslots[(numprocessed + 1) % RECEIVE_RING_LENGTH].configindex, threadid);
        if (somescrunch) {
            delete[] scratchspace->pulsaraccumspace;
        }
    }
    std::cout << "Begin vectorFree calls" << std::endl;
    vectorFree(scratchspace->threadcrosscorrs);
    vectorFree(scratchspace->chanfreqs);
    vectorFree(scratchspace->rotator);
    vectorFree(scratchspace->rotated);
    vectorFree(scratchspace->channelsums);
    vectorFree(scratchspace->argument);
    std::cout << "scratchspace" << std::endl;
    if (scratchspace->starecordbuffer != 0) {
        free(scratchspace->starecordbuffer);
    }
    delete scratchspace;

    cinfo << startl << "PROCESS " << mpiid << "/" << threadid << " process thread exiting!!!" << endl;

//    extern int calls;
//    cout << "process calls: " << calls << endl;
}

__global__ void _gpu_processBaselineBased(
        cuFloatComplex** const gpuM1Freqs,
        cuFloatComplex** const gpuM2Freqs,
        cuFloatComplex* const threadcrosscorrs_gpu,
        const char* const stream1BandIndexes_gpu,
        const char* const stream2BandIndexes_gpu,
        int xmacpasses,
        int numbaselines,
        int xmacstridelength,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels,
        size_t numrecordedbands
) {
    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numPolarisationProducts index [0 .. numPolarisationProducts]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numPolarisationProducts size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Get the subloopindex
    const size_t subloopindex = blockIdx.x;

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t polidx = threadIdx.x;
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
    const size_t numPolarisationProducts = blockDim.x;

    if (channelindex >= xmacstridelength) {
        return;
    }

    for (auto x = 0; x < xmacpasses; x++) {
        for (auto j = 0; j < numbaselines; j++) {
            size_t resultindex = (x * numbaselines + j) * numPolarisationProducts * xmacstridelength;
            const size_t crosscorrIndex = resultindex + polidx * xmacstridelength + channelindex;

            auto xmacstart = x * xmacstridelength;

            const size_t freqIndex = (subloopindex * fftchannels * numrecordedbands) + (stream1BandIndexes_gpu[(subloopindex * numPolarisationProducts * numbaselines) + (j * numPolarisationProducts) + polidx] * fftchannels) + channelindex + xmacstart;
            const size_t conjIndex = (subloopindex * fftchannels * numrecordedbands) + (stream2BandIndexes_gpu[(subloopindex * numPolarisationProducts * numbaselines) + (j * numPolarisationProducts) + polidx] * fftchannels) + channelindex + xmacstart;

            atomicAddFloatComplex1(&threadcrosscorrs_gpu[crosscorrIndex], cuCmulf(gpuM1Freqs[j][freqIndex], gpuM2Freqs[j][conjIndex]));
        }
    }
}

void
GPUCore::issue_tofft(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                     threadscratchspace *scratchspace) {
#ifndef NEUTERED_DIFX
    int status, localfreqindex;
    int binloop;

    binloop = 1;
    if (procslots[index].pulsarbin && !procslots[index].scrunchoutput)
        binloop = procslots[index].numpulsarbins;

    // Per-datastream host prep + first GPU half (input H2D, unpack, weights,
    // fringe rotation, FFT). NOTE: on the DEVICE-weights path, zeroAutocorrelations
    // and resetpcal are NOT done here - they clear host mirrors the deferred tail
    // reads, and with tail-overlap pipelining the next subint's issue_tofft runs
    // before this subint's tail, so they move to completegpudata (just before
    // finishWeights / copyPCalTones).
    //
    // The DIFX_GPU_WEIGHTS_HOST fallback is different: it fills weights[][] here
    // (in set_weights, += per window) and copies autocorrelations[][] in afterfft,
    // both BEFORE the tail - so on that path zeroAutocorrelations must run HERE,
    // ahead of set_weights, not in the tail (where it would wipe them). That path
    // is forced non-pipelined (see the constructor), so this pre-zero is safe.
    if (!GPUMode::useGpuWeights()) {
        for (int j = 0; j < numdatastreams; j++)
            modes[j]->zeroAutocorrelations();
    }
    for (int j = 0; j < numdatastreams; j++) {
        modes[j]->setValidFlags(&(procslots[index].controlbuffer[j][3]));
        modes[j]->setData(procslots[index].databuffer[j], procslots[index].datalengthbytes[j],
                          procslots[index].controlbuffer[j][0], procslots[index].controlbuffer[j][1],
                          procslots[index].controlbuffer[j][2]);
        modes[j]->setOffsets(procslots[index].offsets[0], procslots[index].offsets[1], procslots[index].offsets[2]);
        modes[j]->setDumpKurtosis(scratchspace->dumpkurtosis);
        if (scratchspace->dumpkurtosis)
            modes[j]->zeroKurtosis();

        // Tell the Mode which procslot this subint occupies so it stages its
        // per-subint host uploads into the matching RING-deep host slot - the
        // overlap issues this tofft while the previous subint's async H2Ds may
        // still be draining (see GPUMode::enableHostRing / setProcSlot).
        ((GPUMode *) modes[j])->setProcSlot(index);

        // First half of station processing (see process_gpu_afterfft for the rest).
        ((GPUMode *) modes[j])->process_gpu_tofft(0, numblocks, startblock, numblocks);

        // Capture this subint's validity NOW, before the next subint's issue
        // overwrites datalengthbytes/offsetseconds; the deferred tail reads it.
        gpuprocslots[index].validsubint[j] = ((GPUMode *) modes[j])->isSubintValid() ? 1 : 0;
    }

    //zero the results for this thread (unused on the GPU path, but kept in step
    //with the CPU path; the fused XMAC writes results_gpu directly)
    status = vectorZero_cf32(scratchspace->threadcrosscorrs, procslots[index].threadresultlength);
    if (status != vecNoErr)
        csevere << startl << "Error trying to zero threadcrosscorrs!!!" << endl;

    //zero the baselineweights and baselineshiftdecorrs for this thread (consumed
    //by the tail's host-weights-fallback fold, so zero them before that runs)
    for (int i = 0; i < config->getFreqTableLength(); i++) {
        if (config->isFrequencyUsed(procslots[index].configindex, i)) {
            for (int b = 0; b < binloop; b++) {
                for (int j = 0; j < numbaselines; j++) {
                    localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, i);
                    if (localfreqindex >= 0) {
                        status = vectorZero_f32(scratchspace->baselineweight[i][b][j],
                                                config->getBNumPolProducts(procslots[index].configindex, j,
                                                                           localfreqindex));
                        if (status != vecNoErr)
                            csevere << startl << "Error trying to zero baselineweight[" << i << "][" << b << "][" << j
                                    << "]!!!" << endl;
                    }
                }
            }
            if (model->getNumPhaseCentres(procslots[index].offsets[0]) > 1) {
                for (int j = 0; j < numbaselines; j++) {
                    localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, i);
                    if (localfreqindex >= 0) {
                        status = vectorZero_f32(scratchspace->baselineshiftdecorr[i][j],
                                                model->getNumPhaseCentres(procslots[index].offsets[0]));
                        if (status != vecNoErr)
                            csevere << startl << "Error trying to zero baselineshiftdecorr[" << i << "][" << j << "]!!!"
                                    << endl;
                    }
                }
            }
        }
    }
#endif
}

void
GPUCore::issue_afterfft_xmac_drain(int index, int threadid, int startblock, int numblocks, Mode **modes,
                                   Polyco *currentpolyco, threadscratchspace *scratchspace) {
    ++core_calls;

    auto start = high_resolution_clock::now();

    //std::cout << "called GPUCore::processgpudata for the " << core_calls << " time, index: " << index << ", startblock: "
    //          << startblock << ", numblocks: " << numblocks << std::endl;

//following statement used to cut all all processing for "Neutered DiFX"
#ifndef NEUTERED_DIFX
    // Whole subint processed in one FFT batch (numfftloops == 1 on the GPU path).
    int numBufferedFFTs = numblocks;

    // Second half of per-datastream station processing (weight reduction, pcal,
    // fractional rotation + autocorrelations). The first half (input H2D, unpack,
    // gpu_set_weights, fringe rotation, FFT) ran in issue_tofft; single-compute-
    // stream ordering keeps the fftd/gDataWeights buffers this reads valid.
    DIFX_NVTX_PUSH("station_processing");
    for (int j = 0; j < numdatastreams; j++) {
        ((GPUMode *) modes[j])->process_gpu_afterfft(0, numBufferedFFTs, startblock, numblocks);
    }
    DIFX_NVTX_POP(); // station_processing

    {
//
//        //All baseline freq indices into the freq table are determined by the *first* datastream
//        //in the event of correlating USB with LSB data.  Hence all Nyquist offsets/channels etc
//        //are determined by the freq corresponding to the *first* datastream
//        auto xmacpasses = config->getNumXmacStrides(procslots[index].configindex, 0);
//
//        //do the cross multiplication - gets messy for the pulsar binning
//        for (int j = 0; j < numbaselines; j++) {
//            //get the two modes that contribute to this baseline
//            auto ds1index = config->getBOrderedDataStream1Index(procslots[index].configindex, j);
//            auto ds2index = config->getBOrderedDataStream2Index(procslots[index].configindex, j);
//            auto m1 = modes[ds1index];
//            auto m2 = modes[ds2index];
//
//            auto _f = m1->getGpuFreqs();
//            auto _c = m2->getGpuConjugatedFreqs();
//            checkCuda(cudaMemcpyAsync(&gpuM1Freqs[j], &_f, sizeof(cuFloatComplex *), cudaMemcpyHostToDevice, cuStream));
//            checkCuda(cudaMemcpyAsync(&gpuM2Freqs[j], &_c, sizeof(cuFloatComplex *), cudaMemcpyHostToDevice, cuStream));
//
//            for (int f = 0; f < config->getFreqTableLength(); f++) {
//                if (config->isFrequencyUsed(procslots[index].configindex, f)) {
//
//                    if (numPolarisationProducts != config->getBNumPolProducts(procslots[index].configindex, j, f)) {
//                        NOT_SUPPORTED("Different values for numPolarisationProducts");
//                    }
//
//                    //do the baseline-based processing for this batch of FFT chunks
//                    for (int fftsubloop = 0; fftsubloop < numBufferedFFTs; fftsubloop++) {
//                        auto i = fftloop * numBufferedFFTs + fftsubloop + startblock;
//                        if (i >= startblock + numblocks)
//                            break; //may not have to fully complete last fftloop
//
//                        localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, f);
//                        if (localfreqindex >= 0)
//                        {
//                            //add the desired results into the resultsbuffer, for each polarisation pair [and pulsar bin]
//                            //loop through each polarisation for this frequency
//                            for (int p = 0; p < numPolarisationProducts; p++) {
//                                stream1BandIndexes[(fftsubloop * numPolarisationProducts * numbaselines) +
//                                                   (j * numPolarisationProducts) + p] =
//                                        config->getBDataStream1BandIndex(
//                                                procslots[index].configindex,
//                                                j,
//                                                localfreqindex,
//                                                p
//                                        );
//
//                                stream2BandIndexes[(fftsubloop * numPolarisationProducts * numbaselines) +
//                                                   (j * numPolarisationProducts) + p] =
//                                        config->getBDataStream2BandIndex(
//                                                procslots[index].configindex,
//                                                j,
//                                                localfreqindex,
//                                                p
//                                        );
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        checkCuda(cudaMemcpyAsync(stream1BandIndexes_gpu, stream1BandIndexes, sizeof(char) * numPolarisationProducts * numBufferedFFTs * numbaselines, cudaMemcpyHostToDevice, cuStream));
//        checkCuda(cudaMemcpyAsync(stream2BandIndexes_gpu, stream2BandIndexes, sizeof(char) * numPolarisationProducts * numBufferedFFTs * numbaselines, cudaMemcpyHostToDevice, cuStream));
//
//        processBaselineBased(
//                gpuM1Freqs,
//                gpuM2Freqs,
//                stream1BandIndexes_gpu,
//                stream2BandIndexes_gpu,
//                threadcrosscorrs_gpu,
//                xmacstridelength,
//                numPolarisationProducts,
//                numBufferedFFTs,
//                xmacpasses,
//                fftloop,
//                startblock,
//                numblocks,
//                config->getFNumChannels(0) * 2,
//                config->getDNumRecordedBands(0, 0),
//                cuStream
//        );
//
//        checkCuda(cudaMemcpyAsync(scratchspace->threadcrosscorrs, threadcrosscorrs_gpu, sizeof(cuFloatComplex) * maxthreadresultlength, cudaMemcpyDeviceToHost, cuStream));
//
//        checkCuda(cudaStreamSynchronize(cuStream));


        // ---------------------------------------------------------------------
        // FUSED XMAC AND AVERAGE SETUP
        // ---------------------------------------------------------------------
        DIFX_NVTX_PUSH("xmac_launch");
        // Ensure this slot's device-side results buffer is cleanly zeroed for this subint
        checkCuda(cudaMemsetAsync(gpuprocslots[index].results_gpu, 0, maxcoreresultlength * sizeof(cuFloatComplex), cuStream));

        // The kernel launch metadata (band indexes, result offsets, channel/pol
        // counts and the FFT buffer pointers) is invariant for a given
        // configuration, so build/upload it once and reuse it across subints.
        // It only needs (re)building on the first subint or a config change.
        if (xmacPlanConfigIndex != procslots[index].configindex) {
            buildXmacPlans(procslots[index].configindex, modes);
        }

        // Launch one fused kernel per used frequency using the cached metadata.
        for (const auto &plan : xmacPlans) {
            // Spread the averaged-channel dimension across the grid's z axis so
            // the launch fills the device rather than producing only
            // numbaselines*numPolarisationProducts blocks. Shrink the block when
            // there are fewer channels than the default so idle threads don't
            // hold SM slots (down to one warp).
            int threadsPerBlock = 256;
            if (plan.num_averaged_channels < threadsPerBlock)
                threadsPerBlock = ((plan.num_averaged_channels + 31) / 32) * 32;
            int numChanBlocks =
                (plan.num_averaged_channels + threadsPerBlock - 1) / threadsPerBlock;

            // If baselines*pols*channels alone can't occupy the device (the
            // geodesy regime: 1-6 baselines, often 1 pol, modest channel
            // counts), also split the FFT/time integration across the grid, in
            // chunks of fftsPerChunk FFTs whose partial sums the kernel combines
            // with an atomic add. Otherwise keep a single chunk, which preserves
            // the plain-store (no atomics) fast path for the many-channel case.

            const long long launchThreads =
                (long long)numbaselines * plan.numPolarisationProducts *
                numChanBlocks * threadsPerBlock;


            const long long targetThreads = (long long)cudaMultiProcessorCount * 2048;
            int numFftChunks = 1;
            if (launchThreads < targetThreads) {
                

                           
                
                numFftChunks = (int)((targetThreads + launchThreads - 1) / launchThreads);
                if (numFftChunks > numBufferedFFTs)
                    numFftChunks = numBufferedFFTs;
                // Respect the 65535 gridDim.z hardware limit.

                if (numFftChunks > 65535 / numChanBlocks)
                    numFftChunks = 65535 / numChanBlocks; 
                if (numFftChunks < 1)
                    numFftChunks = 1;
            }
            int fftsPerChunk = (numBufferedFFTs + numFftChunks - 1) / numFftChunks;
            if (fftsPerChunk < 1) // degenerate numBufferedFFTs == 0 slot
                fftsPerChunk = 1;
            // Recompute so the grid has no empty trailing chunks.
            numFftChunks = (numBufferedFFTs + fftsPerChunk - 1) / fftsPerChunk;
            if (numFftChunks < 1)
                numFftChunks = 1;

            dim3 threads(threadsPerBlock);
            dim3 blocks(numbaselines, plan.numPolarisationProducts,
                        numChanBlocks * numFftChunks);

            gpu_fuse_xmac_and_average<<<blocks, threads, 0, cuStream>>>(
                d_m1_ptrs, d_m2_ptrs,
                d_v1_ptrs, d_v2_ptrs,
                plan.d_stream1BandIndexes, plan.d_stream2BandIndexes,
                plan.d_coreResultBaselineOffsets,
                gpuprocslots[index].results_gpu,
                numbaselines, plan.numPolarisationProducts, numBufferedFFTs,
                fftsPerChunk,
                plan.num_averaged_channels, plan.channelstoaverage,
                plan.d_stream1BandStride, plan.d_stream1WindowStride,
                plan.d_stream2BandStride, plan.d_stream2WindowStride
            );
        }

        // Reduce the per-window baseline weights on the device (Increment 2):
        // sum dataweight1[w]*dataweight2[w] over the subint's windows for every
        // (freq, baseline, polproduct), then async-copy the small flat result
        // back for the host finalize fold. Enqueued on cuStream after the XMAC,
        // so the drain below guarantees both the kernel and its D2H completed
        // before the fold reads h_bweightResults. Device-weights path only;
        // the DIFX_GPU_WEIGHTS_HOST fallback keeps the host per-window loop.
        // A single (non-ringed) result buffer suffices: unlike the deferred
        // visibility D2H, this copy is fully drained here, before this subint's
        // fold and the next subint's reduction.
        if (GPUMode::useGpuWeights() && bweightNumAccum > 0) {
            const int tpb = 128;
            gpu_baseline_weights<<<(bweightNumAccum + tpb - 1) / tpb, tpb, 0, cuStream>>>(
                d_bwDw1, d_bwDw2, d_bweightResults, bweightNumAccum, numBufferedFFTs);
            checkCuda(cudaMemcpyAsync(h_bweightResults, d_bweightResults,
                                      bweightNumAccum * sizeof(float),
                                      cudaMemcpyDeviceToHost, cuStream));
        }

        // ---------------------------------------------------------------------
        // DRAIN (event-based, non-blocking) - overlaps the NEXT subint's compute
        // ---------------------------------------------------------------------
        // Record compute-done on cuStream after all of this subint's device work
        // (afterfft outputs + XMAC + the baseline-weight reduction and their
        // cuStream D2Hs). This REPLACES the old cudaStreamSynchronize: the host
        // no longer blocks here, so loopprocess enqueues the NEXT subint's
        // issue_tofft on cuStream right after this - running during this subint's
        // drain and its host tail (completegpudata).
        checkCuda(cudaEventRecord(gpuprocslots[index].evComputeDone, cuStream));
        DIFX_NVTX_POP(); // xmac_launch

        // Visibility D2H on the dedicated d2h stream, gated on evComputeDone so
        // it waits for the XMAC output without draining cuStream (letting it
        // overlap the next subint's compute and this subint's host tail).
        // completegpudata(index) awaits gpuprocslots[index].d2hDone; because d2hStream first
        // waits evComputeDone, that wait transitively covers the cuStream output
        // D2Hs too (autocorr / gTotalWeight / pcal / baseline-weights). The whole
        // host tail (finishWeights, autocorr + baseline-weight fold, pcal, and
        // the visibility memcpy) moves to completegpudata.
        int xcorrslength = config->getCoreResultXcorrsLength(procslots[index].configindex);
        checkCuda(cudaStreamWaitEvent(d2hStream, gpuprocslots[index].evComputeDone, 0));
        checkCuda(cudaMemcpyAsync(gpuprocslots[index].results_host, gpuprocslots[index].results_gpu,
                                  xcorrslength * sizeof(cuFloatComplex),
                                  cudaMemcpyDeviceToHost, d2hStream));
        checkCuda(cudaEventRecord(gpuprocslots[index].d2hDone, d2hStream));
    }


    //std::cout << "Ended fft loop" << std::endl;

    // Mark completion of this subint's input host->device copies. The copies
    // are enqueued on cuStream interleaved with the station-processing kernels,
    // so a single record here (after the last fftloop pass) necessarily covers
    // them all. completegpudata(index) waits on this before the slot lock is
    // released, so the manager cannot refill procslots[index].databuffer[]
    // while a direct (pinned-input) async copy from it is still in flight.
    // NOTE: this therefore fences everything enqueued on cuStream so far, not
    // just the copies; if the input copies ever move to a dedicated H2D stream
    // (to overlap with compute), record this event on that stream instead so
    // slot release does not serialize against the subint's compute.
    checkCuda(cudaEventRecord(gpuprocslots[index].h2dInputDone, cuStream));

    // cuStream is the Core's persistent stream now; it is NOT destroyed here
    // (nor in ~GPUCore - see the no-CUDA-calls-at-teardown note there). The host
    // tail (autocorrelation + baseline-weight fold, pcal, visibility memcpy) is
    // deferred to completegpudata so it overlaps the next subint's compute.
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    avg_postprocess += duration.count();
#endif
}

void GPUCore::completegpudata(int index, int threadid, int startblock, int numblocks, Mode **modes,
                              Polyco *currentpolyco, threadscratchspace *scratchspace) {
    (void) currentpolyco; // pulsar binning / uvshift not supported on the GPU path
    // Wait for this subint's device->host copies to land: gpuprocslots[index].d2hDone (the
    // visibility D2H on d2hStream, which first waited evComputeDone - so this
    // transitively covers the cuStream output D2Hs: autocorr / gTotalWeight /
    // pcal / baseline-weights) and gpuprocslots[index].h2dInputDone (the input copies, so the
    // slot's databuffer can be recycled by the manager).
    DIFX_NVTX_RANGE("complete_d2h_wait");
    checkCuda(cudaEventSynchronize(gpuprocslots[index].d2hDone));
    checkCuda(cudaEventSynchronize(gpuprocslots[index].h2dInputDone));

#ifndef NEUTERED_DIFX
    int perr, localfreqindex, numfftsprocessed;
    int binloop, maxacblocks, acblockcount, acshiftcount;
    double blockns;

    binloop = 1;
    if (procslots[index].pulsarbin && !procslots[index].scrunchoutput)
        binloop = procslots[index].numpulsarbins;

    // Whole subint is one FFT batch on the GPU path (numfftloops == 1), so one
    // AC-average pass covers it. Recompute the AC cadence params here (they were
    // issue-time locals in the old monolithic issuegpudata).
    numfftsprocessed = numblocks;
    acblockcount = 0;
    acshiftcount = 0;
    blockns = ((double) (config->getSubintNS(procslots[index].configindex))) /
              ((double) (config->getBlocksPerSend(procslots[index].configindex)));
    maxacblocks = ((int) (model->getMaxNSBetweenACAvg(procslots[index].offsets[0]) / blockns));
    maxacblocks -= maxacblocks % numblocks;
    if (maxacblocks == 0)
        maxacblocks = numblocks;

    DIFX_NVTX_PUSH("host_finalize");

    // Zero the host autocorrelation/weight mirrors for every datastream, then
    // fold each VALID datastream's device-computed weights + autocorrelations
    // into them. On the device-weights path zeroAutocorrelations() moved here
    // from the old pre-GPU prep: finishWeights overwrites the autocorr mirror and
    // accumulates (+=) into weights[][], so the mirrors must be zeroed here, just
    // before it (and doing it here rather than at issue time is required by tail-
    // overlap pipelining - the next subint's issue_tofft runs before this tail).
    // gpuprocslots[index].validsubint[j] (captured at issue time) drives the invalid-subint
    // skip - an invalid datastream's autocorr/weights stay zeroed. On the
    // DIFX_GPU_WEIGHTS_HOST fallback the mirrors were already filled before the
    // tail (weights in set_weights, autocorr in afterfft) and were pre-zeroed in
    // issue_tofft, so they must NOT be re-zeroed here.
    if (GPUMode::useGpuWeights()) {
        for (int j = 0; j < numdatastreams; j++)
            modes[j]->zeroAutocorrelations();
    }
    for (int j = 0; j < numdatastreams; j++)
        ((GPUMode *) modes[j])->finishWeights(gpuprocslots[index].validsubint[j] != 0);

    // NOTE: unlike Core::processdata we must NOT call uvshiftAndAverage here.
    // The fused XMAC kernel has already written the final averaged cross
    // correlations directly into gpuprocslots[index].results_gpu (staged to results_host and
    // memcpy'd into procslots below), and nothing on the GPU path fills
    // scratchspace->threadcrosscorrs - so the CPU averaging would add
    // uninitialised memory on top of the correct visibilities. (This also means
    // multiple phase centres / pulsar binning, which rely on uvshiftAndAverage,
    // are not supported on the GPU path.)

    // Autocorrelation averaging into procslots results at the AC cadence, then
    // the leftover flush (mirrors the old host_accumulate + host_finalize; with
    // numfftloops == 1 exactly one of the two averageAndSendAutocorrs runs).
    acblockcount += numfftsprocessed;
    if (acblockcount == maxacblocks) {
        averageAndSendAutocorrs(index, threadid,
                                (startblock + acshiftcount * maxacblocks + ((double) maxacblocks) / 2.0) * blockns,
                                maxacblocks * blockns, modes, scratchspace);
        acblockcount = 0;
        acshiftcount++;
        for (int j = 0; j < numdatastreams; j++)
            modes[j]->zeroAutocorrelations();
    }
    if (acblockcount != 0) {
        averageAndSendAutocorrs(index, threadid,
                                (startblock + acshiftcount * maxacblocks + ((double) acblockcount) / 2.0) * blockns,
                                acblockcount * blockns, modes, scratchspace);
    }
    if (scratchspace->dumpkurtosis) {
        averageAndSendKurtosis(index, threadid, (startblock + numblocks / 2.0) * blockns, numblocks * blockns,
                               numblocks, modes, scratchspace);
    }

    // Baseline-weight per-window host loop (DIFX_GPU_WEIGHTS_HOST fallback only;
    // the device-weights path reduced these on the GPU into h_bweightResults,
    // folded below). numfftloops == 1, so the FFT index is just startblock+sub.
    if (!procslots[index].pulsarbin && !GPUMode::useGpuWeights()) {
        for (int fftsubloop = 0; fftsubloop < numblocks; fftsubloop++) {
            auto i = fftsubloop + startblock;
            if (i >= startblock + numblocks)
                break;
            for (int f = 0; f < config->getFreqTableLength(); f++) {
                if (config->isFrequencyUsed(procslots[index].configindex, f)) {
                    for (int j = 0; j < numbaselines; j++) {
                        localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, f);
                        if (localfreqindex >= 0) {
                            auto ds1index = config->getBOrderedDataStream1Index(procslots[index].configindex, j);
                            auto ds2index = config->getBOrderedDataStream2Index(procslots[index].configindex, j);
                            auto m1 = modes[ds1index];
                            auto m2 = modes[ds2index];
                            for (int p = 0; p < config->getBNumPolProducts(procslots[index].configindex, j,
                                                                           localfreqindex); p++) {
                                int ds1recordbandindex = config->getBDataStream1RecordBandIndex(
                                        procslots[index].configindex, j, localfreqindex, p);
                                int ds2recordbandindex = config->getBDataStream2RecordBandIndex(
                                        procslots[index].configindex, j, localfreqindex, p);
                                if (ds1recordbandindex < 0 || ds2recordbandindex < 0) {
                                    cerror << startl
                                           << "Error: Core::processdata(): one of the record band indices could not be found: ds1recordbandindex = "
                                           << ds1recordbandindex << " ds2recordbandindex = " << ds2recordbandindex
                                           << endl;
                                } else {
                                    auto weight1 = m1->getDataWeight(ds1recordbandindex, fftsubloop);
                                    auto weight2 = m2->getDataWeight(ds2recordbandindex, fftsubloop);
                                    scratchspace->baselineweight[f][0][j][p] += weight1 * weight2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Fold the baseline weights into the results. Device-weights path: the flat,
    // self-describing h_bweightResults (each accumulator carries its floatresults
    // destination). Fallback: the nested config walk (plus multiple-phase-centre
    // baselineshiftdecorr, which stays zero on the GPU path).
    perr = pthread_mutex_lock(&(procslots[index].bweightcopylock));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock bweight copy mutex!!!"
                << endl;
    if (GPUMode::useGpuWeights()) {
        for (int a = 0; a < bweightNumAccum; a++)
            procslots[index].floatresults[bwDestOffset[a]] += h_bweightResults[a];
    } else {
        for (int f = 0; f < config->getFreqTableLength(); f++) {
            if (config->isFrequencyUsed(procslots[index].configindex, f)) {
                for (int l = 0; l < numbaselines; l++) {
                    localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, l, f);
                    if (localfreqindex >= 0) {
                        auto resultindex = config->getCoreResultBWeightOffset(procslots[index].configindex, f, l) * 2;
                        for (int b = 0; b < binloop; b++) {
                            for (int j = 0;
                                 j < config->getBNumPolProducts(procslots[index].configindex, l, localfreqindex); j++) {
                                procslots[index].floatresults[resultindex] += scratchspace->baselineweight[f][b][l][j];
                                resultindex++;
                            }
                        }
                    }
                }
                if (model->getNumPhaseCentres(procslots[index].offsets[0]) > 1) {
                    for (int l = 0; l < numbaselines; l++) {
                        localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, l, f);
                        if (localfreqindex >= 0) {
                            auto resultindex = config->getCoreResultBShiftDecorrOffset(procslots[index].configindex, f, l) * 2;
                            for (int s = 0; s < model->getNumPhaseCentres(procslots[index].offsets[0]); s++) {
                                procslots[index].floatresults[resultindex] += scratchspace->baselineshiftdecorr[f][l][s];
                                resultindex++;
                            }
                        }
                    }
                }
            }
        }
    }
    perr = pthread_mutex_unlock(&(procslots[index].bweightcopylock));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock copy mutex!!!"
                << endl;

    // pcal: reset the host extractors (moved from the old pre-GPU prep) then copy
    // the pcal tones. On the GPU path finalisepcal re-sets the extractor data
    // from the pinned pcal_output before reading, so resetting here (in the tail,
    // just before copyPCalTones) is equivalent to the old prep-time reset and
    // avoids the next subint's reset clobbering this subint's pcal.
    for (int j = 0; j < numdatastreams; j++) {
        if (config->getDPhaseCalIntervalMHz(procslots[index].configindex, j) > 0)
            modes[j]->resetpcal();
    }
    copyPCalTones(index, threadid, modes);
    DIFX_NVTX_POP(); // host_finalize
#endif

    // Copy the staged visibilities across into procslots (the visibility prefix,
    // disjoint from the autocorr/weight/pcal trailing regions folded above,
    // which the main thread pre-zeroed before handing us this slot).
    int xcorrslength = config->getCoreResultXcorrsLength(procslots[index].configindex);
    memcpy(procslots[index].results, gpuprocslots[index].results_host, xcorrslength * sizeof(cuFloatComplex));
}
// vim: shiftwidth=2:softtabstop=2:expandtab
