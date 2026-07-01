#include "gpucore.cuh"
#include "gpumode.cuh"
#include "alert.h"
#include "gpumode_kernels.cuh"
#include <thread>
//#include <iostream>
#include <chrono>

using namespace std::chrono;

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
 * blocks of the previous grid-stride design. Each thread owns exactly one
 * output channel and, since every (baseline, pol, avg_chan) maps to a unique
 * slot in the pre-zeroed results buffer within a launch, it writes with a
 * plain store instead of an atomic add.
 */
__global__ void gpu_fuse_xmac_and_average(
        const cuFloatComplex* const * const gpuM1Freqs,
        const cuFloatComplex* const * const gpuM2Freqs,
        const int* const stream1BandIndexes,
        const int* const stream2BandIndexes,
        const int* const coreResultBaselineOffsets,
        cuFloatComplex* const results_gpu,
        int numbaselines,
        int numPolarisationProducts,
        int numBufferedFFTs,
        int num_averaged_channels,
        int channelstoaverage,
        int fftchannels,
        int numrecordedbands
) {
    // gridDim.x  = numbaselines
    // gridDim.y  = numPolarisationProducts
    // gridDim.z  = ceil(num_averaged_channels / blockDim.x)
    // blockDim.x = output averaged channels processed per block (e.g. 256)

    int baseline = blockIdx.x;
    int pol = blockIdx.y;

    // Each thread owns exactly one output averaged channel, indexed across both
    // the block and the grid's z dimension.
    int avg_chan = blockIdx.z * blockDim.x + threadIdx.x;
    if (avg_chan >= num_averaged_channels) return;

    // 1. Resolve Band Indexes
    // We pre-flattened the 2D [baseline][pol] config arrays on the host.
    int b1 = stream1BandIndexes[baseline * numPolarisationProducts + pol];
    int b2 = stream2BandIndexes[baseline * numPolarisationProducts + pol];

    // If the baseline doesn't participate in this frequency, it was flagged as -1.
    if (b1 < 0 || b2 < 0) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    // 2. Time Integration
    // Summation over all FFTs currently residing in the GPU buffer.
    for (int fft = 0; fft < numBufferedFFTs; fft++) {

        // Calculate the absolute starting index for this specific FFT and Band
        int base_idx1 = (fft * numrecordedbands + b1) * fftchannels;
        int base_idx2 = (fft * numrecordedbands + b2) * fftchannels;

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

    // This (baseline, pol, avg_chan) triple is unique within the launch and the
    // results buffer was pre-zeroed, so a plain store is correct and avoids the
    // serialisation of an atomic add.
    results_gpu[final_index] = sum;
}

GPUCore::GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm)
        : Core(id, conf, dids, rcomm) {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    if(numprocessthreads > 1) {
      cerr << "GPU DiFX must have 1 thread per Core process - had " << numprocessthreads << endl;
      exit(1);
    }

    // Initialize the long-lived stream
    checkCuda(cudaStreamCreate(&cuStream));

    // Allocate the main results buffer
    checkCuda(cudaMalloc(&results_gpu, maxcoreresultlength * sizeof(cuFloatComplex)));

    // Allocate the shared, frequency-independent FFT buffer pointer arrays (one
    // entry per baseline). These are populated once per configuration by
    // buildXmacPlans(). The per-frequency band-index/offset arrays are allocated
    // lazily inside buildXmacPlans() and cached in xmacPlans.
    checkCuda(cudaMalloc(&d_m1_ptrs, numbaselines * sizeof(cuFloatComplex*)));
    checkCuda(cudaMalloc(&d_m2_ptrs, numbaselines * sizeof(cuFloatComplex*)));
}

static unsigned long long avg_preprocess;
static unsigned long long avg_postprocess;
int core_calls;
GPUCore::~GPUCore() {
    cout << "Average core pre: " << avg_preprocess / core_calls << endl;
    cout << "Average core post: " << avg_postprocess / core_calls << endl;

    // Clean up device memory
    freeXmacPlans();
    checkCuda(cudaFree(results_gpu));
    checkCuda(cudaFree(d_m1_ptrs));
    checkCuda(cudaFree(d_m2_ptrs));

    // Destroy the stream last
    checkCuda(cudaStreamDestroy(cuStream));
}

void GPUCore::freeXmacPlans() {
    for (auto &plan : xmacPlans) {
        checkCuda(cudaFree(plan.d_stream1BandIndexes));
        checkCuda(cudaFree(plan.d_stream2BandIndexes));
        checkCuda(cudaFree(plan.d_coreResultBaselineOffsets));
    }
    xmacPlans.clear();
    xmacPlanConfigIndex = -1;
}

// Precompute, once per configuration, all the DiFX config-tree metadata that the
// fused XMAC kernel needs. This is invariant for a given configindex, so caching
// it removes the per-subintegration host-side config walk and the redundant
// device uploads that used to happen on every call to processgpudata().
void GPUCore::buildXmacPlans(int configindex, Mode **modes) {
    // Drop any previously-cached plans (e.g. on a configuration change).
    freeXmacPlans();

    const int sampling_multiplier =
        (config->getDSampling(configindex, 0) == Configuration::COMPLEX) ? 1 : 2;
    const int numrecordedbands = config->getDNumRecordedBands(configindex, 0);

    // The FFT buffer pointers depend only on the baseline's datastreams (not on
    // frequency) and the GPUMode buffers never move, so they are gathered and
    // uploaded once here and shared across all per-frequency launches.
    const cuFloatComplex* h_m1_ptrs[numbaselines];
    const cuFloatComplex* h_m2_ptrs[numbaselines];
    for (int j = 0; j < numbaselines; j++) {
        int ds1index = config->getBOrderedDataStream1Index(configindex, j);
        int ds2index = config->getBOrderedDataStream2Index(configindex, j);
        h_m1_ptrs[j] = ((GPUMode*)modes[ds1index])->fftd_gpu->gpuPtr();
        h_m2_ptrs[j] = ((GPUMode*)modes[ds2index])->conj_fftd_gpu->gpuPtr();
    }
    checkCuda(cudaMemcpyAsync(d_m1_ptrs, h_m1_ptrs, numbaselines * sizeof(cuFloatComplex*),
                             cudaMemcpyHostToDevice, cuStream));
    checkCuda(cudaMemcpyAsync(d_m2_ptrs, h_m2_ptrs, numbaselines * sizeof(cuFloatComplex*),
                             cudaMemcpyHostToDevice, cuStream));

    for (int f = 0; f < config->getFreqTableLength(); f++) {
        if (!config->isFrequencyUsed(configindex, f)) continue;

        int freqchannels = config->getFNumChannels(f);
        int channelstoaverage = config->getFChannelsToAverage(f);
        int numPolarisationProducts = config->getBNumPolProducts(
            configindex, 0, config->getBLocalFreqIndex(configindex, 0, f));

        XmacFreqPlan plan;
        plan.numPolarisationProducts = numPolarisationProducts;
        plan.num_averaged_channels = freqchannels / channelstoaverage;
        plan.channelstoaverage = channelstoaverage;
        plan.fftchannels = freqchannels * sampling_multiplier;
        plan.numrecordedbands = numrecordedbands;

        // Gather the per-baseline band indexes and result offsets for this frequency.
        int h_stream1BandIndexes[numbaselines * numPolarisationProducts];
        int h_stream2BandIndexes[numbaselines * numPolarisationProducts];
        int h_coreResultBaselineOffsets[numbaselines];
        for (int j = 0; j < numbaselines; j++) {
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
        checkCuda(cudaMemcpyAsync(plan.d_stream1BandIndexes, h_stream1BandIndexes,
                                 numbaselines * numPolarisationProducts * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_stream2BandIndexes, h_stream2BandIndexes,
                                 numbaselines * numPolarisationProducts * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));
        checkCuda(cudaMemcpyAsync(plan.d_coreResultBaselineOffsets, h_coreResultBaselineOffsets,
                                 numbaselines * sizeof(int),
                                 cudaMemcpyHostToDevice, cuStream));

        xmacPlans.push_back(plan);
    }

    // The uploads above source from host stack buffers that go out of scope when
    // this function returns; synchronise so the (one-off) copies complete first.
    checkCuda(cudaStreamSynchronize(cuStream));

    xmacPlanConfigIndex = configindex;
}

void GPUCore::loopprocess(int threadid) {
    int perr, numprocessed, startblock, numblocks, lastconfigindex, numpolycos, maxchan, maxpolycos, stadumpchannels, strideplussteplen, maxrotatestrideplussteplength, maxxmaclength, slen;
    double sec;
    bool pulsarbin, somepulsarbin, somescrunch, dumpingsta, nowdumpingsta;
    processslot *currentslot;
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

    //while valid, process data
    while (procslots[(numprocessed) % RECEIVE_RING_LENGTH].keepprocessing) {
        currentslot = &(procslots[numprocessed % RECEIVE_RING_LENGTH]);
        if (pulsarbin) {
            sec = double(startseconds + model->getScanStartSec(currentslot->offsets[0], startmjd, startseconds) +
                         currentslot->offsets[1]) + ((double) currentslot->offsets[2]) / 1000000000.0;
            //get the correct Polyco for this time range and set it up correctly
            currentpolyco = Polyco::getCurrentPolyco(currentslot->configindex, startmjd, sec / 86400.0, polycos,
                                                     numpolycos, false);
            if (currentpolyco == NULL) {
                cfatal << startl << "Could not locate a polyco to cover time " << startmjd + sec / 86400.0
                       << " - aborting!!!" << endl;
                currentpolyco = Polyco::getCurrentPolyco(currentslot->configindex, startmjd, sec / 86400.0, polycos,
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

        //process our section of responsibility for this time range
        processgpudata(numprocessed++ % RECEIVE_RING_LENGTH, threadid, startblock, numblocks, modes, currentpolyco,
                       scratchspace);

        if (threadid == 0)
            numcomplete++;

        currentslot = &(procslots[numprocessed % RECEIVE_RING_LENGTH]);
        //if the configuration changes from this segment to the next, change our setup accordingly
        if (currentslot->configindex != lastconfigindex) {
            cinfo << startl << "Core " << mpiid << " threadid " << threadid << ": changing config to "
                  << currentslot->configindex << endl;
            updateconfig(lastconfigindex, currentslot->configindex, threadid, startblock, numblocks, numpolycos,
                         pulsarbin, modes, polycos, false);
            cinfo << startl << "Core " << mpiid << " threadid " << threadid
                  << ": config changed successfully - pulsarbin is now " << pulsarbin << endl;
            createPulsarVaryingSpace(scratchspace->pulsaraccumspace, &(scratchspace->bins), currentslot->configindex,
                                     lastconfigindex, threadid);
            allocateConfigSpecificThreadArrays(scratchspace->baselineweight, scratchspace->baselineshiftdecorr,
                                               currentslot->configindex, lastconfigindex, threadid);
            lastconfigindex = currentslot->configindex;
        }
    }

    //fallen out of loop, so must be finished.  Unlock held mutex
//  cinfo << startl << "PROCESS " << mpiid << "/" << threadid << " process thread about to free resources and exit" << endl;
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
GPUCore::processgpudata(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                        threadscratchspace *scratchspace) {
    ++core_calls;

    auto start = high_resolution_clock::now();

    //std::cout << "called GPUCore::processgpudata for the " << core_calls << " time, index: " << index << ", startblock: "
    //          << startblock << ", numblocks: " << numblocks << std::endl;

#ifndef NEUTERED_DIFX
    int status, numfftloops, numfftsprocessed;
    int binloop;
    int xcblockcount, maxxcblocks, xcshiftcount;
    int acblockcount, maxacblocks, acshiftcount;
    int xmacstridelength, localfreqindex;
    double blockns;
    int numBufferedFFTs;
#endif
    int perr;

//following statement used to cut all all processing for "Neutered DiFX"
#ifndef NEUTERED_DIFX
    xmacstridelength = config->getXmacStrideLength(procslots[index].configindex);
    binloop = 1;
    if (procslots[index].pulsarbin && !procslots[index].scrunchoutput)
        binloop = procslots[index].numpulsarbins;

    //set up the mode objects that will do the station-based processing
    for (int j = 0; j < numdatastreams; j++) {
        //zero the autocorrelations and set delays
        modes[j]->zeroAutocorrelations();
        modes[j]->setValidFlags(&(procslots[index].controlbuffer[j][3]));
        //std::cout << "Setting data for datastream " << j << " with datalengthbytes: " << procslots[index].datalengthbytes[j]
        //          << " and controlbuffer: " << procslots[index].controlbuffer[j][0] << ", "
       //           << procslots[index].controlbuffer[j][1] << ", " << procslots[index].controlbuffer[j][2]
       //           << std::endl;
        modes[j]->setData(procslots[index].databuffer[j], procslots[index].datalengthbytes[j],
                          procslots[index].controlbuffer[j][0], procslots[index].controlbuffer[j][1],
                          procslots[index].controlbuffer[j][2]);
        modes[j]->setOffsets(procslots[index].offsets[0], procslots[index].offsets[1], procslots[index].offsets[2]);
        modes[j]->setDumpKurtosis(scratchspace->dumpkurtosis);
        if (scratchspace->dumpkurtosis)
            modes[j]->zeroKurtosis();

        //reset pcal
        if (config->getDPhaseCalIntervalMHz(procslots[index].configindex, j) > 0) {
            modes[j]->resetpcal();
        }
    }

    //zero the results for this thread
    status = vectorZero_cf32(scratchspace->threadcrosscorrs, procslots[index].threadresultlength);
    if (status != vecNoErr)
        csevere << startl << "Error trying to zero threadcrosscorrs!!!" << endl;

    //zero the baselineweights and baselineshiftdecorrs for this thread
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

    //set up variables which control the number of loops through buffered FFT results
    xcblockcount = 0;
    xcshiftcount = 0;
    acblockcount = 0;
    acshiftcount = 0;
    // Force a single data transfer to the GPU
    numBufferedFFTs = numblocks; // Do all the FFTs in one go
    numfftloops = numblocks / numBufferedFFTs; // so this will always be 1
    if (numblocks % numBufferedFFTs != 0)
        numfftloops++;
    blockns = ((double) (config->getSubintNS(procslots[index].configindex))) /
              ((double) (config->getBlocksPerSend(procslots[index].configindex)));

    maxxcblocks = ((int) (model->getMaxNSBetweenXCAvg(procslots[index].offsets[0]) / blockns));
    maxxcblocks -= maxxcblocks % numBufferedFFTs;
    if (maxxcblocks == 0) {
        maxxcblocks = numBufferedFFTs;
        cverbose << startl << "Requested cross-correlation shift/average time of "
                 << model->getMaxNSBetweenXCAvg(procslots[index].offsets[0]) << " ns cannot be met with "
                 << numBufferedFFTs << " FFTs being buffered; the time resolution which will be attained is "
                 << maxxcblocks * blockns << " ns" << endl;
    }

    maxacblocks = ((int) (model->getMaxNSBetweenACAvg(procslots[index].offsets[0]) / blockns));
    maxacblocks -= maxacblocks % numBufferedFFTs;
    if (maxacblocks == 0) {
        maxacblocks = numBufferedFFTs;
        cverbose << startl << "Requested autocorrelation shift/average time of "
                 << model->getMaxNSBetweenACAvg(procslots[index].offsets[0]) << " ns cannot be met with "
                 << numBufferedFFTs << " FFTs being buffered; the time resolution which will be attained is "
                 << maxacblocks * blockns << " ns" << endl;
    }

    checkCuda(cudaStreamCreate(&cuStream));
//    cuFloatComplex* threadcrosscorrs_gpu;
//
//    checkCuda(cudaStreamCreate(&cuStream));
//
//    checkCuda(cudaMallocAsync(&threadcrosscorrs_gpu, sizeof(cuFloatComplex) * maxthreadresultlength, cuStream));
//    checkCuda(cudaMemsetAsync(threadcrosscorrs_gpu, 0, sizeof(cuFloatComplex) * maxthreadresultlength, cuStream));
//    checkCuda(cudaHostRegister(scratchspace->threadcrosscorrs, sizeof(cuFloatComplex) * maxthreadresultlength, cudaHostRegisterPortable));
//
//    cuFloatComplex** gpuM1Freqs;
//    cuFloatComplex** gpuM2Freqs;
//    checkCuda(cudaMallocAsync(&gpuM1Freqs, sizeof(cuFloatComplex*) * numbaselines, cuStream));
//    checkCuda(cudaMallocAsync(&gpuM2Freqs, sizeof(cuFloatComplex*) * numbaselines, cuStream));
//
//    auto numPolarisationProducts = config->getBNumPolProducts(procslots[index].configindex, 0, 0);
//
//    char* stream1BandIndexes_gpu;
//    char* stream2BandIndexes_gpu;
//    checkCuda(cudaMallocAsync(&stream1BandIndexes_gpu, sizeof(char) * numPolarisationProducts * numBufferedFFTs * numbaselines, cuStream));
//    checkCuda(cudaMallocAsync(&stream2BandIndexes_gpu, sizeof(char) * numPolarisationProducts * numBufferedFFTs * numbaselines, cuStream));
//
//    auto stream1BandIndexes = new char[numPolarisationProducts * numBufferedFFTs * numbaselines];
//    auto stream2BandIndexes = new char[numPolarisationProducts * numBufferedFFTs * numbaselines];
//
//    checkCuda(cudaHostRegister(stream1BandIndexes, numPolarisationProducts * numBufferedFFTs * numbaselines, cudaHostRegisterPortable));
//    checkCuda(cudaHostRegister(stream2BandIndexes, numPolarisationProducts * numBufferedFFTs * numbaselines, cudaHostRegisterPortable));
//
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    avg_preprocess += duration.count();

    // process each chunk of FFTs in turn
    //std::cout << "numfftloops = " << numfftloops << std::endl;
    //std::cout << "numsdatastreams = " << numdatastreams << std::endl;
    for (int fftloop = 0; fftloop < numfftloops; fftloop++) {
        start = high_resolution_clock::now();

        numfftsprocessed = 0;   // not strictly needed, but to prevent compiler warning

        // do the station-based processing for this batch of FFT chunks.
        vector<std::thread> streamThreads;

        for (int j = 0; j < numdatastreams; j++) {
            //std::cout << "datastream " << j << " processing fftloop " << fftloop << std::endl;  
          //  if (procslots[index].datalengthbytes[j] > 1) {
    	//    streamThreads.emplace_back([&numfftsprocessed, &modes, j, fftloop, numBufferedFFTs, startblock, numblocks] {
            //std::cout << "before process_gpu: fftloop = " << fftloop << ", startblock = " << startblock << std::endl;    
        //    numfftsprocessed = ((GPUMode *) modes[j])->process_gpu(fftloop, numBufferedFFTs, startblock, numblocks);
        //    });
             numfftsprocessed = ((GPUMode *) modes[j])->process_gpu(fftloop, numBufferedFFTs, startblock, numblocks);

        //    }
        }

        for (auto &t: streamThreads) {
            t.join();
        }

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        //cout << "total processing: " << duration.count() << endl;
        start = high_resolution_clock::now();
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
        // Ensure the device-side results buffer is cleanly zeroed for this subint
        checkCuda(cudaMemsetAsync(results_gpu, 0, maxcoreresultlength * sizeof(cuFloatComplex), cuStream));

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
            // numbaselines*numPolarisationProducts blocks.
            //
            // TODO (next perf task): promote the FFT/time dimension into the grid.
            // This grid only has good occupancy when there are many averaged
            // channels. In the geodesy regime (typically 1/3/6 baselines and
            // often 1 pol) numbaselines*numPolarisationProducts is tiny and the
            // channel count may be modest, so the device is under-occupied while
            // each thread serially integrates over numBufferedFFTs (which is
            // large). Add a grid dimension over FFTs (or FFT chunks), have
            // threads accumulate partial sums, then reduce across the FFT
            // dimension per output channel (this will need an atomic add or a
            // two-pass/shared-memory reduction, since multiple blocks would then
            // write the same output slot - unlike the single-store fast path
            // below). Consider branching by regime to keep this fast path for the
            // many-channel case.
            dim3 threads(256); // output channels processed per block
            dim3 blocks(numbaselines, plan.numPolarisationProducts,
                        (plan.num_averaged_channels + threads.x - 1) / threads.x);

            gpu_fuse_xmac_and_average<<<blocks, threads, 0, cuStream>>>(
                d_m1_ptrs, d_m2_ptrs,
                plan.d_stream1BandIndexes, plan.d_stream2BandIndexes,
                plan.d_coreResultBaselineOffsets,
                results_gpu,
                numbaselines, plan.numPolarisationProducts, numBufferedFFTs,
                plan.num_averaged_channels, plan.channelstoaverage,
                plan.fftchannels, plan.numrecordedbands
            );
        }

        // ---------------------------------------------------------------------
        // FINAL HOST TRANSFER
        // ---------------------------------------------------------------------
        // Instead of dragging un-averaged cross-correlations back, we do a single, 
        // massive PCIe transfer of the final, properly formatted `results` array.
        checkCuda(cudaMemcpyAsync(procslots[index].results, results_gpu, procslots[index].coreresultlength * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, cuStream));
        checkCuda(cudaStreamSynchronize(cuStream));

        //finally, update the baselineweight if not doing any pulsar stuff
        if (!procslots[index].pulsarbin) {
            for (int fftsubloop = 0; fftsubloop < numBufferedFFTs; fftsubloop++) {
                auto i = fftloop * numBufferedFFTs + fftsubloop + startblock;
                if (i >= startblock + numblocks)
                    break; //may not have to fully complete last fftloop
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
                                    int ds1recordbandindex, ds2recordbandindex;

                                    ds1recordbandindex = config->getBDataStream1RecordBandIndex(
                                            procslots[index].configindex, j, localfreqindex, p);
                                    ds2recordbandindex = config->getBDataStream2RecordBandIndex(
                                            procslots[index].configindex, j, localfreqindex, p);

                                    if (ds1recordbandindex < 0 || ds2recordbandindex < 0) {
                                        cerror << startl
                                               << "Error: Core::processdata(): one of the record band indices could not be found: ds1recordbandindex = "
                                               << ds1recordbandindex << " ds2recordbandindex = " << ds2recordbandindex
                                               << endl;
                                    } else {
                                        auto weight1 = m1->getDataWeight(ds1recordbandindex, fftsubloop);
                                        auto weight2 = m2->getDataWeight(ds2recordbandindex, fftsubloop);

                                        auto bweight = weight1 * weight2;

                                        scratchspace->baselineweight[f][0][j][p] += bweight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        //cout << "baseline weight: " << duration.count() << endl;
    }
   

    //std::cout << "Ended fft loop" << std::endl; 
    start = high_resolution_clock::now();

//    checkCuda(cudaHostUnregister(stream1BandIndexes));
//    checkCuda(cudaHostUnregister(stream2BandIndexes));
//    checkCuda(cudaFreeAsync(stream1BandIndexes_gpu, cuStream));
//    checkCuda(cudaFreeAsync(stream2BandIndexes_gpu, cuStream));
//
//    delete[] stream1BandIndexes;
//    delete[] stream2BandIndexes;
//
//    checkCuda(cudaFreeAsync(threadcrosscorrs_gpu, cuStream));
//    checkCuda(cudaHostUnregister(scratchspace->threadcrosscorrs));
    checkCuda(cudaStreamDestroy(cuStream));

    // The rest of this function uses an insignificant amount of time
    if (xcblockcount != 0) {
        uvshiftAndAverage(index, threadid,
                          (startblock + xcshiftcount * maxxcblocks + ((double) xcblockcount) / 2.0) * blockns,
                          xcblockcount * blockns, currentpolyco, scratchspace);
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

    //lock the bweight copylock, so we're the only one adding to the result array (baseline weight section)
    perr = pthread_mutex_lock(&(procslots[index].bweightcopylock));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock bweight copy mutex!!!"
                << endl;

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

    //unlock the bweight copylock
    perr = pthread_mutex_unlock(&(procslots[index].bweightcopylock));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock copy mutex!!!"
                << endl;



    

    //copy the PCal results
    copyPCalTones(index, threadid, modes);

//end the cutout of processing in "Neutered DiFX"
#endif

    //grab the next slot lock
    perr = pthread_mutex_lock(&(procslots[(index + 1) % RECEIVE_RING_LENGTH].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying lock mutex "
                << (index + 1) % RECEIVE_RING_LENGTH << endl;

    //unlock the one we had
    perr = pthread_mutex_unlock(&(procslots[index].slotlocks[threadid]));
    if (perr != 0)
        csevere << startl << "PROCESSTHREAD " << mpiid << "/" << threadid << " error trying unlock mutex " << index
                << endl;

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    avg_postprocess += duration.count();
    //std::cout << "At the bottom of Core" << std::endl;
}
// vim: shiftwidth=2:softtabstop=2:expandtab
