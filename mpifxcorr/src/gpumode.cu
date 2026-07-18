#include "gpumode.cuh"
#include "alert.h"
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <cufftXt.h>
#include <pcal.h>
#include <algorithm>
#include <cmath>

#include <chrono>
#include <omp.h>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include "mathutil.h"
#include <unistd.h>


using namespace std::chrono;

const int MAX_INDICIES = 10;

cudaStream_t GPUMode::sharedComputeStream = nullptr;
bool GPUMode::inputBuffersPinned = false;

static int weightDebugFrom();   // defined above set_weights, also used by process_gpu

// Env-gated per-window spectral tracing (DIFX_SPEC_DEBUG=<datasec>), the GPU
// twin of the SPECDEBUG lines in CPUMode::process (cpumode.cpp) - identical
// format, so sorted grepped logs diff directly and any CPU-vs-GPU divergence
// localizes to a stage (unpack / fringe rotation+FFT / frac correction).
static int specDebugFrom()
{
    static int from = -2;
    if (from == -2) {
        const char* e = getenv("DIFX_SPEC_DEBUG");
        from = (e != NULL) ? atoi(e) : -1;
    }
    return from;
}

__global__ void gpu_allocate_unpacked(float** arrays, float* data, int nchan, int dlen) {
    // Use arrays to make data into a flattened 2D array
    for (int i = 0; i < nchan; i++) {
        arrays[i] = data + i * dlen;
        //printf("Channel %i starts at %p\n", i, arrays[i]);
    }
}

__global__ void gpu_allocate_unpacked_complex(cuFloatComplex** arrays, cuFloatComplex* data, int nchan, int dlen) {
    // Use arrays to make data into a flattened 2D array
    for (int i = 0; i < nchan; i++) {
        arrays[i] = data + i * dlen;
        //printf("Channel %i starts at %p\n", i, arrays[i]);
    }
}





GPUMode::GPUMode(Configuration *conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend,
                 int gsamples, int nrecordedfreqs, double recordedbw, double *recordedfreqclkoffs,
                 double *recordedfreqclkoffsdelta, double *recordedfreqphaseoffs, double *recordedfreqlooffs,
                 int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling,
                 Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular,
                 int fringerotorder, int arraystridelen, bool cacorrs, double bclock) :
        Mode(conf, confindex, dsindex, recordedbandchan, chanstoavg, bpersend, gsamples, nrecordedfreqs, recordedbw,
             recordedfreqclkoffs, recordedfreqclkoffsdelta, recordedfreqphaseoffs, recordedfreqlooffs, nrecordedbands,
             nzoombands, nbits, sampling, tcomplex, unpacksamp, fbank, linear2circular, fringerotorder, arraystridelen,
             cacorrs, bclock), estimatedbytes_gpu(0) {
    //std::cout << "Constructing a new GPUMode" << std::endl;
    auto start = high_resolution_clock::now();


    

    size_t buffer_payload_bytes = (config->getMaxDataBytes() / config->getMultiplexedFrameBytes(confindex, dsindex)) * config->getMultiplexedFramePayloadBytes(confindex, dsindex);
    std::cout << "buffer_payload_bytes: " << buffer_payload_bytes << std::endl;

    size_t unpacked_size = buffer_payload_bytes * 8 / (config->getDNumBits(confindex, dsindex) * config->getDNumRecordedBands(confindex, dsindex));
    if (usecomplex) {
        unpacked_size /= 2;
    }
    std::cout << "unpacked_size: " << unpacked_size << std::endl;

    // in gpumode.cu, right after unpacked_size is computed:
    size_t maxframes = config->getMaxDataBytes() /
                       config->getMultiplexedFrameBytes(confindex, dsindex);
    size_t payloadsamples = config->getMultiplexedFramePayloadBytes(confindex, dsindex) * 8 /
                            (config->getDNumBits(confindex, dsindex) *
                             config->getDNumRecordedBands(confindex, dsindex));
    if (usecomplex) payloadsamples /= 2;   // mirror Mk5's complex framesamples
    
    //std::cout << "ctor unpacked_size=" << unpacked_size
    //          << " maxframes=" << maxframes
    //          << " payloadsamples(=framesamples)=" << payloadsamples
    //          << " maxframes*payloadsamples=" << maxframes * payloadsamples
    //          << " usecomplex=" << usecomplex << std::endl;


    // Per-band capacity of the unpacked sample buffers; also the stride
    // between bands within unpackeddata_gpu / complex_unpackeddata_gpu.
    unpackedarrays_elem_count = unpacked_size;

    // What's the largest number of FFTs we can fit?
    cfg_numBufferedFFTs = (unpacked_size + fftchannels - 1) / fftchannels;
    //std::cout << "Working on " << cfg_numBufferedFFTs << " FFTs" << std::endl;
    // cfg_numBufferedFFTs = config->getNumBufferedFFTs(confindex);
    vectorFree(dataweight);
    dataweight = vectorAlloc_f32(cfg_numBufferedFFTs);
    for (size_t i = 0; i < cfg_numBufferedFFTs; i++)
        dataweight[i] = 0.0;


    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties( &prop, 0));

    // Use the compute stream GPUCore installed so all modes' station
    // processing and the XMAC share one in-order queue; fall back to a
    // private stream only if none was installed (standalone use).
    if (sharedComputeStream != nullptr) {
        cuStream = sharedComputeStream;
        ownsStream = false;
    } else {
        checkCuda(cudaStreamCreate(&cuStream));
        ownsStream = true;
    }



    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
 
    // Pre-allocate packed data buffer at max possible size. When this mode
    // will use the direct pinned-input path (pinned buffers + shared stream,
    // the same condition process_gpu tests), the host staging half is never
    // touched - skip it (gpuOnly) rather than page-locking getMaxDataBytes
    // per mode that would sit idle.
    packeddata_gpu = new GpuMemHelper<char>(config->getMaxDataBytes(), cuStream,
                                            inputBuffersPinned && !ownsStream);
    checkCuda(cudaStreamSynchronize(cuStream));

    complex_fringe_rotated_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, true);
    estimatedbytes_gpu += complex_fringe_rotated_gpu->size();

    fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
    estimatedbytes_gpu += fftd_gpu->size();

    conj_fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
    estimatedbytes_gpu += conj_fftd_gpu->size();

    temp_autocorrelations_gpu = new GpuMemHelper<cuFloatComplex>(autocorrwidth * numrecordedbands * recordedbandchannels, cuStream);
    estimatedbytes_gpu += temp_autocorrelations_gpu->size();


    // Unpacked data only allocated on GPU
    if (usecomplex) {
        complex_unpackedarrays_gpu = new GpuMemHelper<cuFloatComplex*>(numrecordedbands, cuStream, true);
        complex_unpackeddata_gpu = new GpuMemHelper<cuFloatComplex>(numrecordedbands * unpacked_size, cuStream, true);
    } else {
        unpackedarrays_gpu = new GpuMemHelper<float*>(numrecordedbands, cuStream, true);
        unpackeddata_gpu = new GpuMemHelper<float>(numrecordedbands * unpacked_size, cuStream, true);
    }
    //std::cout << "Unpacked data size: " << numrecordedbands * unpacked_size << std::endl;
    

    // Make sure these are allocated
    if (usecomplex) {
        complex_unpackeddata_gpu->sync();
    } else {
        unpackeddata_gpu->sync();
    }

    // Now launch a kernel to set up the arrays on the GPU
    if (usecomplex) {
        gpu_allocate_unpacked_complex<<<1, 1, 0, cuStream>>>(complex_unpackedarrays_gpu->gpuPtr(), complex_unpackeddata_gpu->gpuPtr(), numrecordedbands, unpacked_size);
    } else {
        gpu_allocate_unpacked<<<1, 1, 0, cuStream>>>(unpackedarrays_gpu->gpuPtr(), unpackeddata_gpu->gpuPtr(), numrecordedbands, unpacked_size);
    }
    if (usecomplex) {
        estimatedbytes_gpu += complex_unpackedarrays_gpu->size();
        estimatedbytes_gpu += complex_unpackeddata_gpu->size();
    } else {
        estimatedbytes_gpu += unpackedarrays_gpu->size();
        estimatedbytes_gpu += unpackeddata_gpu->size();
    }
    gSampleIndexes = new GpuMemHelper<int>(cfg_numBufferedFFTs, cuStream);
    gValidSamples = new GpuMemHelper<bool>(cfg_numBufferedFFTs, cuStream);


    gInterpolator = new GpuMemHelper<double>(interpolator, 3, cuStream);
    gFracSampleError = new GpuMemHelper<float>(cfg_numBufferedFFTs, cuStream);
    gLoFreqs = new GpuMemHelper<double>(numrecordedbands, cuStream);
    counts_gpu = new GpuMemHelper<int>(numrecordedfreqs, cuStream); 

    int max_framestounpack = config->getMaxDataBytes() / config->getMultiplexedFrameBytes(confindex, dsindex);
    valid_frames = new GpuMemHelper<bool>(max_framestounpack, cuStream);
    
    indices = new GpuMemHelper<unsigned int>((MAX_INDICIES * numrecordedfreqs), cuStream);
    for (auto i = 0; i < (MAX_INDICIES * numrecordedfreqs); i++) {
        indices->ptr()[i] = 0xffffffff;
    }
    grecordedfreqclockoffsets = new GpuMemHelper<double>(numrecordedbands, cuStream);
    grecordedfreqclockoffsetsdelta = new GpuMemHelper<double>(numrecordedbands, cuStream);
    grecordedfreqlooffsets = new GpuMemHelper<double>(numrecordedbands, cuStream);
    // Copy the lofreq and freq clock offset values to the GPU
    for (auto i = 0; i < numrecordedbands; i++) {
        int localfreqindex = config->getDLocalRecordedFreqIndex(configindex, datastreamindex, i);
        gLoFreqs->ptr()[i] = config->getDRecordedFreq(configindex, datastreamindex, localfreqindex);
        grecordedfreqclockoffsets->ptr()[i] = recordedfreqclockoffsets[localfreqindex];
        grecordedfreqclockoffsetsdelta->ptr()[i] = recordedfreqclockoffsetsdelta[localfreqindex];
        grecordedfreqlooffsets->ptr()[i] = recordedfreqlooffsets[localfreqindex];
    }

    gLoFreqs->copyToDevice();
    grecordedfreqclockoffsets->copyToDevice();
    grecordedfreqclockoffsetsdelta->copyToDevice();
    grecordedfreqlooffsets->copyToDevice();


    // The below has be moved from pcal extraction to here 
    if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) { 
        pcal_offsets_hz = new GpuMemHelper<int>(numrecordedbands, cuStream);
        N_pcal_bins = new GpuMemHelper<int>(numrecordedbands, cuStream);
        double bandwidth_hz = 1e6*recordedbandwidth;
        double fs_hz = 2 * bandwidth_hz;
        double pcal_spacing_hz = 1e6*config->getDPhaseCalIntervalMHz(configindex, datastreamindex);
        int N_pcal_bins_max=0;    
     
        for (int ii=0; ii<numrecordedbands; ii++) { 
            int localfreqindex = config->getDLocalRecordedFreqIndex(configindex, datastreamindex, ii);
            pcal_offsets_hz->ptr()[ii] = config->getDRecordedFreqPCalOffsetsHz(configindex, datastreamindex, localfreqindex);
            N_pcal_bins->ptr()[ii] = (int)(fs_hz/gcd(fs_hz,pcal_offsets_hz->ptr()[ii]));
            if (N_pcal_bins->ptr()[ii] > N_pcal_bins_max) {
                N_pcal_bins_max = N_pcal_bins->ptr()[ii]; 
            }  
        }
        pcal_bin_stride_length = N_pcal_bins_max*2;
        pcal_offsets_hz->copyToDevice();
        N_pcal_bins->copyToDevice();
        if (usecomplex) {
            pcal_output_complex = new GpuMemHelper<cuFloatComplex>(numrecordedbands*pcal_bin_stride_length, cuStream);
            for (size_t ii=0; ii<numrecordedbands*pcal_bin_stride_length; ii++)
                pcal_output_complex->ptr()[ii] = make_cuFloatComplex(0.0f, 0.0f);
            pcal_output_complex->copyToDevice();
        } else {
            pcal_output_real = new GpuMemHelper<float>(numrecordedbands*pcal_bin_stride_length, cuStream);
            for (size_t ii=0; ii<numrecordedbands*pcal_bin_stride_length; ii++)
                pcal_output_real->ptr()[ii] = 0.0;
            pcal_output_real->copyToDevice();
        }
    }




    int n[] = {fftchannels};
    int istride = 1;
    int ostride = 1;
    int idist = fftchannels;
    int odist = fftchannels;

    int inembed[] = {0};
    int onembed[] = {0};

    checkCufft(
            cufftPlanMany(
                    &fft_plan,
                    1,
                    (int *) &n,
                    (int *) &inembed,
                    istride,
                    idist,
                    (int *) &onembed,
                    ostride,
                    odist,
                    CUFFT_C2C,
                    numrecordedbands * cfg_numBufferedFFTs
            )
    );
    checkCufft(cufftSetStream(fft_plan, cuStream));


    // precalc
    nearestSamples = new GpuMemHelper<int>(cfg_numBufferedFFTs, cuStream);
    checkCuda(cudaStreamSynchronize(cuStream));

    // Cross-check the start-up VRAM estimator against what was actually
    // tallied (the tally covers only the large buffers, so it must never
    // exceed the estimate - if it does, estimateDeviceBytes is stale).
    size_t estimate = estimateDeviceBytes(config, confindex, dsindex);
    if (estimatedbytes_gpu > estimate)
        cwarn << startl << "GPUMode::estimateDeviceBytes (" << estimate
              << " bytes) is smaller than the allocated-buffer tally (" << estimatedbytes_gpu
              << " bytes) for datastream " << dsindex << " - the estimator is out of date" << endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //cout << "GPUMode(): " << duration.count() << endl;



    constructor_time = high_resolution_clock::now();
}

// static long long avg_unpack = 0;
// static long long avg_copyto = 0;
// static long long avg_pcal = 0;
// static long long avg_rotate = 0;
// static long long avg_fft = 0;
// static long long avg_fracrotate = 0;
// static long long avg_postprocess = 0;
// static long long processing_time = 0;
// 
// int calls = 0;
//static int debug_dataweight_gpu_prints = 0;
//static int debug_dataweight_gpu_invalid_prints = 0;

GPUMode::~GPUMode() {
    auto start = high_resolution_clock::now();
    std::cout << "Starting destructor" << std::endl;
    delete packeddata_gpu;
    delete complex_fringe_rotated_gpu;
    delete fftd_gpu;
    delete conj_fftd_gpu;
    delete temp_autocorrelations_gpu;
    if (usecomplex) {
        delete complex_unpackedarrays_gpu;
        delete complex_unpackeddata_gpu;
    } else {
        delete unpackeddata_gpu;
        delete unpackedarrays_gpu;
    }

    delete gSampleIndexes;
    delete gValidSamples;
    delete gInterpolator;
    delete gFracSampleError;

    delete nearestSamples;
    delete counts_gpu;
    delete valid_frames;


    if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) { 
        delete pcal_offsets_hz;
        delete N_pcal_bins;
         if (usecomplex) {
            delete pcal_output_complex;
        } else {
            delete pcal_output_real;
        }       
    }
    printf("pcal_output_real_gpu_mode \n");
    if (pcal_output_real_gpu_mode != nullptr) {
        pcal_output_real_gpu_mode = nullptr;
    }
    printf("freed pcal_output_real_gpu_mode \n");
    if (pcal_output_complex_gpu_mode != nullptr) {
        pcal_output_complex_gpu_mode = nullptr;
    }

    printf("done \n");

    if (ev_start) {
        cudaEventDestroy(ev_start);  cudaEventDestroy(ev_copy1);
        cudaEventDestroy(ev_unpack); cudaEventDestroy(ev_copy2);
        cudaEventDestroy(ev_pcal);   cudaEventDestroy(ev_rotate);
        cudaEventDestroy(ev_fft);    cudaEventDestroy(ev_frac);
    }


    checkCufft(cufftDestroy(fft_plan));
    // Never destroy the shared compute stream - GPUCore owns it, and modes
    // are destroyed/recreated mid-run on configuration changes.
    if (ownsStream)
        checkCuda(cudaStreamDestroy(cuStream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    if (calls > 0) {
        cout << "GPUMode pid=" << getpid()
        // << " tid=" << std::this_thread::get_id()
         << " DS=" << datastreamindex
         << " (" << calls << " calls):" << endl;
        cout << "  copyto:      " << t_copyto      / calls << " us" << endl;
        cout << "  unpack:      " << t_unpack      / calls << " us" << endl;
        cout << "  rotate:      " << t_rotate      / calls << " us" << endl;
        cout << "  fft:         " << t_fft         / calls << " us" << endl;
        cout << "  fracrotate:  " << t_fracrotate  / calls << " us" << endl;
        cout << "  pcal:        " << t_pcal        / calls << " us" << endl;
        cout << "  postprocess: " << t_postprocess / calls << " us" << endl;
        cout << "  total:       " << (double)t_total / 1e6 << " s" << endl;
    }


}

size_t GPUMode::estimateDeviceBytes(Configuration *config, int configindex, int dsindex)
{
    // Mirrors the constructor's device allocations (GpuMemHelper device
    // buffers plus the cuFFT plan work area) using only Configuration
    // lookups. If the constructor gains/loses allocations this must be
    // updated to match - the constructor cross-checks its actual tally
    // against this estimate and warns on divergence.
    const int nbands = config->getDNumRecordedBands(configindex, dsindex);
    const int nfreqs = config->getDNumRecordedFreqs(configindex, dsindex);
    const int nbits = config->getDNumBits(configindex, dsindex);
    const bool complexsampled = (config->getDSampling(configindex, dsindex) == Configuration::COMPLEX);
    const int freqindex = config->getDRecordedFreqIndex(configindex, dsindex, 0);
    const int recordedbandchan = config->getFNumChannels(freqindex);
    int fftchannels = recordedbandchan * 2;
    if (complexsampled)
        fftchannels /= 2;

    const size_t maxdatabytes = config->getMaxDataBytes();
    const int framebytes = config->getMultiplexedFrameBytes(configindex, dsindex);
    if (framebytes <= 0 || nbands <= 0 || fftchannels <= 0)
        return 0; // not a format the GPU path can run; getMode's gate reports it
    const size_t maxframes = maxdatabytes / framebytes;
    size_t unpacked_size = maxframes * config->getMultiplexedFramePayloadBytes(configindex, dsindex) * 8 /
                           (nbits * nbands);
    if (complexsampled)
        unpacked_size /= 2;
    const size_t nFFTs = (unpacked_size + fftchannels - 1) / fftchannels; // = cfg_numBufferedFFTs

    size_t bytes = 0;
    bytes += maxdatabytes;                                                          // packeddata_gpu
    bytes += 3 * (size_t)fftchannels * nFFTs * nbands * sizeof(cuFloatComplex);     // fringe rotated + fftd + conj_fftd
    const int acwidth = config->writeAutoCorrs(configindex) ? 2 : 1;
    bytes += (size_t)acwidth * nbands * recordedbandchan * sizeof(cuFloatComplex);  // temp_autocorrelations_gpu
    bytes += (size_t)nbands * unpacked_size *
             (complexsampled ? sizeof(cuFloatComplex) : sizeof(float));             // unpacked data
    bytes += nbands * sizeof(void*);                                                // unpacked pointer array
    bytes += nFFTs * (2 * sizeof(int) + sizeof(bool) + sizeof(float));              // gSampleIndexes+nearestSamples+gValidSamples+gFracSampleError
    bytes += 3 * sizeof(double);                                                    // gInterpolator
    bytes += (size_t)nbands * 4 * sizeof(double);                                   // gLoFreqs + 3 clock/lo offset arrays
    bytes += (size_t)nfreqs * sizeof(int);                                          // counts_gpu
    bytes += maxframes * sizeof(bool);                                              // valid_frames
    bytes += (size_t)MAX_INDICIES * nfreqs * sizeof(unsigned int);                  // indices

    if (config->getDPhaseCalIntervalMHz(configindex, dsindex) > 0) {
        // mirror the constructor's N_pcal_bins_max / pcal_bin_stride_length
        double bandwidth_hz = 1e6 * config->getFreqTableBandwidth(freqindex);
        double fs_hz = 2 * bandwidth_hz;
        int N_pcal_bins_max = 0;
        for (int b = 0; b < nbands; b++) {
            int localfreqindex = config->getDLocalRecordedFreqIndex(configindex, dsindex, b);
            int offset_hz = config->getDRecordedFreqPCalOffsetsHz(configindex, dsindex, localfreqindex);
            int nbins = (int)(fs_hz / gcd(fs_hz, (double)offset_hz));
            if (nbins > N_pcal_bins_max)
                N_pcal_bins_max = nbins;
        }
        const size_t stride = (size_t)N_pcal_bins_max * 2;
        bytes += 2 * (size_t)nbands * sizeof(int);                                  // pcal_offsets_hz + N_pcal_bins
        bytes += (size_t)nbands * stride *
                 (complexsampled ? sizeof(cuFloatComplex) : sizeof(float));         // pcal_output
    }

    // cuFFT C2C plan work area, worst case (same shape as the constructor's
    // cufftPlanMany: rank-1 length-fftchannels batched over bands*windows)
    size_t fftwork = 0;
    if (cufftEstimate1d(fftchannels, CUFFT_C2C, nbands * (int)nFFTs, &fftwork) == CUFFT_SUCCESS)
        bytes += fftwork;

    return bytes;
}

__global__ void check_unpack(float** array, int nchan, int nsamp) {
    printf("Unpacked data:\n");
    for (int o = 0; o < 10; o++) {
        for (int c = 0; c < nchan; c++) {
            printf("%f\t", array[c][o]);
        }
        printf("\n");
    }
}

__global__ void debug_print_unpack_window(float **array, int sample_index, int max_samples,
                                          int ds, int idx, int sub, int band,
                                          int nearest, int unpackstart, int sampleoffset, int datasamples) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    if (sample_index < 0 || sample_index + 7 >= max_samples) {
        return;
    }

    const float *src = array[band];
    printf("DEBUG_UNPACK_GPU_WINDOW ds=%d idx=%d sub=%d band=%d nearest=%d unpackstart=%d sampleIndex=%d sampleoffset=%d datasamples=%d values=%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
           ds,
           idx,
           sub,
           band,
           nearest,
           unpackstart,
           sample_index,
           sampleoffset,
           datasamples,
           src[sample_index + 0],
           src[sample_index + 1],
           src[sample_index + 2],
           src[sample_index + 3],
           src[sample_index + 4],
           src[sample_index + 5],
           src[sample_index + 6],
           src[sample_index + 7]);
}



// Little kernel to print out results of the FFT for debugging/checking
__global__ void print_fft_window(cuFloatComplex* fftd_data, int nchan, int fftchannels, int nffts) {
    for (int win = 0; win < nffts; win++) {
        printf("---\tSample %i\t---\n", win);
        for (int s = 0; s < fftchannels; s++) {
            for (int c = 0; c < nchan; c++) {
                int index = win * nchan * fftchannels + c * fftchannels + s;
                printf("%f\t%f\t|\t", fftd_data[index].x, fftd_data[index].y);
            }
            printf("\n");
        }
        printf("\n---\t---\t---\n");
    }

}

int GPUMode::process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                         int numblocks)  //frac sample error is in microseconds
{


    auto begin_time = high_resolution_clock::now();
    calls += 1;
    
    // Sanity checks
    //std::cout << "numBufferedFFTs: " << numBufferedFFTs << std::endl;
    //std::cout << "cfg_numBufferedFFTs: " << cfg_numBufferedFFTs << std::endl;

    assert(numblocks == numBufferedFFTs); // All FFTs are always done in one go
    assert(numBufferedFFTs <= cfg_numBufferedFFTs); // The value calculated in the constructor should be the same as we are told now

//    if (config->getDPhaseCalIntervalMHz(configindex, datastreamindex) != 0) {
//        NOT_SUPPORTED("DPhaseCal");
//    }

    if (fringerotationorder != 1) { // linear only
        NOT_SUPPORTED("fringerotationorder = " + to_string(fringerotationorder));
    }

    if (usedouble) {
        NOT_SUPPORTED("usedouble branch");
    }

    for (auto i = 0; i < numrecordedfreqs; i++) {
        if (recordedfreqlooffsets[i] > 0.0 || recordedfreqlooffsets[i] < 0.0) {
            NOT_SUPPORTED("lo offsets");
        }
    }

    if (usecomplex && usedouble) {
        NOT_SUPPORTED("complex double-sideband data");
    } //else if (usecomplex) {
     //   NOT_SUPPORTED("complex data");
    //}

    if (deltapoloffsets) {
        NOT_SUPPORTED("deltapoloffsets");
    }

    if (config->getDRecordedLowerSideband(configindex, datastreamindex, 0)) {
        NOT_SUPPORTED("lower sideband");
    }

    if (dumpkurtosis) {
        NOT_SUPPORTED("dump_kurtosis branch");
    }

    if (linear2circular) {
        NOT_SUPPORTED("linear to circular polarisation conversion");
    } else if (phasepoloffset) {
        NOT_SUPPORTED("phase polarisation offset");
    }

    auto start = high_resolution_clock::now();


    if (ev_start == nullptr) {
        checkCuda(cudaEventCreate(&ev_start));
        checkCuda(cudaEventCreate(&ev_copy1));
        checkCuda(cudaEventCreate(&ev_unpack));
        checkCuda(cudaEventCreate(&ev_copy2));
        checkCuda(cudaEventCreate(&ev_pcal));
        checkCuda(cudaEventCreate(&ev_rotate));
        checkCuda(cudaEventCreate(&ev_fft));
        checkCuda(cudaEventCreate(&ev_frac));
    }



    // Copy packed data to device, needed to refactor this since we moved packed data allocation to the constructor.
    cudaEventRecord(ev_start, cuStream);

    // Input H2D. When GPUCore has page-locked the procslots receive buffers
    // (see DIFX_GPU_PIN_INPUT), DMA directly from the delivered buffer -
    // no host staging copy. GPUCore's h2dInputDone event guarantees the
    // buffer is not recycled while this async copy is in flight. Otherwise
    // fall back to staging through packeddata_gpu's pinned host half
    // (pageable -> pinned -> device); there the host memcpy dominates this
    // range, the H2D itself is small.
    DIFX_NVTX_PUSH("h2d_stage");
    // The !ownsStream condition guards the direct path's reuse fence:
    // GPUCore's h2dInputDone event is recorded on the SHARED stream, so a
    // mode running on a private stream (standalone use) must stage instead.
    if (inputBuffersPinned && !ownsStream) {
        checkCuda(cudaMemcpyAsync(packeddata_gpu->gpuPtr(), data,
                                  datalengthbytes, cudaMemcpyHostToDevice, cuStream));
    } else {
        memcpy(packeddata_gpu->ptr(), data, datalengthbytes);
        checkCuda(cudaMemcpyAsync(packeddata_gpu->gpuPtr(), packeddata_gpu->ptr(),
                                  datalengthbytes, cudaMemcpyHostToDevice, cuStream));
    }
    DIFX_NVTX_POP();

    // Figure out how many frames in the packed data
    int framestounpack = datalengthbytes / config->getMultiplexedFrameBytes(configindex, datastreamindex);
    //std::cout << "framestounpack: " << framestounpack << std::endl;
    if (datalengthbytes > 1) {  // datalengthbytes <= 1 means an invalid sub int which should be handled....
        //std::cout << "datalengthbytes: " << datalengthbytes << " getMultiplexedFrameBytes: " << config->getMultiplexedFrameBytes(configindex, datastreamindex) << std::endl;
        assert(datalengthbytes % config->getMultiplexedFrameBytes(configindex, datastreamindex) == 0);     // Buffer contains fraction of a frame :(. This shouldn't happen!
        
    } else {

      // set everything to zero and return
        checkCuda(cudaMemsetAsync(fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex), cuStream));
	    checkCuda(cudaMemsetAsync(conj_fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex), cuStream));

        // We return before set_weights() runs, so explicitly invalidate every
        // FFT window and zero its data weight - otherwise stale values from
        // the previous subintegration survive, and GPUCore's baseline-weight
        // accumulation (and the XMAC kernel's validity flags) would count a
        // subint that contains no data. The CPU path sets dataweight = 0 for
        // every FFT of such a subint (Mode::process validity branch).
        for (int i = 0; i < cfg_numBufferedFFTs; i++) {
            dataweight[i] = 0.0;
            if (perbandweights) {
                for (int b = 0; b < numrecordedbands; b++)
                    perbandweights[i][b] = 0.0;
            }
            gValidSamples->ptr()[i] = false;
        }
        gValidSamples->copyToDevice();

        checkCuda(cudaStreamSynchronize(cuStream));
	    return numBufferedFFTs;
    }

    //valid_frames = new GpuMemHelper<bool>(framestounpack, cuStream, false); 

    // Reset pcal accumulation only once at the start of a subintegration.
     if (!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0) &&
            (datasec != pcalResetDataSec || datans != pcalResetDataNs)) {
        if (usecomplex) {
            checkCuda(cudaMemsetAsync(pcal_output_complex->gpuPtr(), 0,
                                      sizeof(cuFloatComplex) * numrecordedbands * pcal_bin_stride_length, cuStream));
        } else {
            checkCuda(cudaMemsetAsync(pcal_output_real->gpuPtr(), 0,
                                      sizeof(float) * numrecordedbands * pcal_bin_stride_length, cuStream));
        }
        pcalResetDataSec = datasec;
        pcalResetDataNs = datans;
    }
 
 
 
    //if (!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0) &&
    //        (datasec != pcalResetDataSec || datans != pcalResetDataNs)) {
    //    checkCuda(cudaMemsetAsync(pcal_output_real->gpuPtr(), 0,
    //                              sizeof(float) * numrecordedbands * pcal_bin_stride_length, cuStream));
    //        pcalResetDataSec = datasec;
    //        pcalResetDataNs = datans;
    //}

    // Reset the autocorrelations
    checkCuda(cudaMemsetAsync(temp_autocorrelations_gpu->gpuPtr(), 0,
                              sizeof(cf32) * numrecordedbands * recordedbandchannels * autocorrwidth, cuStream));

    // Update the interpolator
    gInterpolator->copyToDevice();
    cudaEventRecord(ev_copy1, cuStream);




    // ==========================================
    // 2. UNPACK
    // ==========================================
    // Host-side per-FFT-window delay-polynomial evaluation (nearestSamples etc.).
    DIFX_NVTX_PUSH("calculatePre_cpu");
    calculatePre_cpu(fftloop, numBufferedFFTs, startblock, numblocks);
    DIFX_NVTX_POP();

    // HDRDEBUG (gated by DIFX_WEIGHT_DEBUG, same as the CPU twin in
    // cpumode.cpp): dump the delivered buffer's VDIF frame-class transitions
    // (real / all-zero / fill-pattern / invalid-bit) once per subint. The
    // delivered tail past the true end of data is unspecified buffer content
    // and can differ from run to run, so each run's log needs its own dump.
    if (weightDebugFrom() >= 0 && datasec >= weightDebugFrom()) {
        static const char* dbg_names[4] = {"zero", "invalidbit", "fill", "real"};
        int dbg_framebytes = config->getMultiplexedFrameBytes(configindex, datastreamindex);
        int prevclass = -1;
        for (int f = 0; f < framestounpack; f++) {
            // Read from the delivered buffer, not packeddata_gpu's host half -
            // the latter is no longer filled when the input buffers are pinned.
            const unsigned int *hdr = (const unsigned int *)((const unsigned char *)data + (size_t)f * dbg_framebytes);
            const unsigned int *pay = hdr + 8;
            int cls;
            if (hdr[0] == 0 && hdr[1] == 0 && hdr[2] == 0 && hdr[3] == 0)
                cls = 0;
            else if ((hdr[0] >> 31) & 0x1)
                cls = 1;
            else if (pay[0] == 0x11223344 || (hdr[2] & 0xFFFFFF) == 0)
                cls = 2;
            else
                cls = 3;
            if (cls != prevclass || f == framestounpack - 1)
                fprintf(stderr, "HDRDEBUG ds=%d datasec=%d datans=%d frame=%d/%d hdr=%08x %08x %08x %08x pay0=%08x class=%s\n",
                        datastreamindex, datasec, datans, f, framestounpack,
                        hdr[0], hdr[1], hdr[2], hdr[3], pay[0], dbg_names[cls]);
            prevclass = cls;
        }
    }

    packeddata_gpu->sync();
    unpack_all(framestounpack);

    // The unpack kernel only decoded framestounpack frames; beyond that the
    // unpacked buffers still hold samples from the previous subintegration.
    // Zero the tail so an FFT window that straddles the end of the delivered
    // data (processed with a fractional weight, see set_weights) sees zeros
    // for its missing samples - exactly what the CPU path's unpacker leaves
    // there. Mid-scan the buffers are full and this is a no-op.
    {
        int tail_framesamples = config->getMultiplexedFramePayloadBytes(configindex, datastreamindex)*8 /
            (config->getDNumBits(configindex, datastreamindex)*config->getDNumRecordedBands(configindex, datastreamindex)*config->getDDecimationFactor(configindex, datastreamindex));
        if (usecomplex)
            tail_framesamples /= 2;
        size_t loadedsamples = (size_t)framestounpack * tail_framesamples;
        if (loadedsamples < unpackedarrays_elem_count) {
            size_t tail = unpackedarrays_elem_count - loadedsamples;
            for (int b = 0; b < numrecordedbands; b++) {
                if (usecomplex) {
                    checkCuda(cudaMemsetAsync(complex_unpackeddata_gpu->gpuPtr() + b*unpackedarrays_elem_count + loadedsamples,
                                              0, tail * sizeof(cuFloatComplex), cuStream));
                } else {
                    checkCuda(cudaMemsetAsync(unpackeddata_gpu->gpuPtr() + b*unpackedarrays_elem_count + loadedsamples,
                                              0, tail * sizeof(float), cuStream));
                }
            }
        }
    }

    cudaEventRecord(ev_unpack, cuStream);
   
    // ==========================================
    // 3. COPY TO (Weights & Indices)
    // ==========================================
    int counts[numrecordedfreqs] = {0};
    // Set up the FFT window indices and weights
    // Ideally this will move to the GPU but it's a bit tricky. Isn't *too* time intensive anyway I think
    
    // CRITICAL: nearestSamples->copyToDevice() in calculatePre_cpu() is async.
    // Sync before reading nearestSamples->ptr() to avoid reading stale data from previous iteration.
    nearestSamples->sync();

    // Host-side per-FFT-window weight/validity calculation (reads valid_frames
    // back from the device via nearestSamples->sync() above).
    DIFX_NVTX_PUSH("set_weights");
    for (int fftwin = 0; fftwin < numBufferedFFTs; fftwin++) {
        set_weights(fftwin, framestounpack, counts, numBufferedFFTs);
    }
    DIFX_NVTX_POP();

    // Indices are now calculated, so we can copy them to the gpu
    indices->copyToDevice();

    // We need to copy the sample indexes to the gpu
    gSampleIndexes->copyToDevice();
    gValidSamples->copyToDevice();

    cudaEventRecord(ev_copy2, cuStream);

    // ==========================================
    // 4. PCAL EXTRACTION
    // ==========================================

    if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) { 
        pcalExtraction(fftloop, numBufferedFFTs, startblock, numblocks);
        // point pcal_output_real_gpu_mode
        if (usecomplex) {
            pcal_output_complex->copyToHost();
            pcal_output_complex_gpu_mode = reinterpret_cast<cf32 *>(pcal_output_complex->ptr());
        } else {
            pcal_output_real->copyToHost();
            pcal_output_real_gpu_mode = pcal_output_real->ptr();
        }
    }    
    cudaEventRecord(ev_pcal, cuStream);  

    
    // ==========================================
    // 5. FRINGE ROTATION
    // ==========================================
    fringeRotation(fftloop, numBufferedFFTs, startblock, numblocks);
    cudaEventRecord(ev_rotate, cuStream);
    
    // ==========================================
    // 6. FFT
    // ==========================================

    runFFT();
    cudaEventRecord(ev_fft, cuStream);
   

    

    // ==========================================
    // 7. FRACTIONAL ROTATION
    // ==========================================
    // do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f) 
    fractionalRotation(fftloop, numBufferedFFTs, startblock, numblocks, calccrosspolautocorrs, counts);
    cudaEventRecord(ev_frac, cuStream);
    cudaEventSynchronize(ev_frac);

    // GPU twin of the CPU path's SPECDEBUG tracing (see specDebugFrom above).
    // All device work for this batch is complete at this point, so small
    // synchronous copies of the traced windows are safe.
    if (specDebugFrom() >= 0 && datasec >= specDebugFrom()) {
        for (int sub = 0; sub < numBufferedFFTs; sub++) {
            int idx = fftloop * numBufferedFFTs + startblock + sub;
            if (idx % 128 != 5 || idx >= startblock + numblocks)
                continue;
            if (!gValidSamples->ptr()[sub])
                continue;
            int off = gSampleIndexes->ptr()[sub];
            if (off < 0 || (size_t)off + 516 > unpackedarrays_elem_count)
                continue;
            for (int b = 0; b < numrecordedbands; b++) {
                // The mid-window fields (unp2/rot, offsets 510-515) bracket the
                // single-corrupted-sample-at-offset-512 signature seen in the
                // complex-complex test: spec differences of the form d*(-1)^k.
                float ur[4], ui[4], u2r[6], u2i[6];
                cuFloatComplex rot[6], spec[4];
                if (usecomplex) {
                    cuFloatComplex unp[4], unp2[6];
                    checkCuda(cudaMemcpy(unp, complex_unpackeddata_gpu->gpuPtr() + (size_t)b*unpackedarrays_elem_count + off,
                                         4*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                    checkCuda(cudaMemcpy(unp2, complex_unpackeddata_gpu->gpuPtr() + (size_t)b*unpackedarrays_elem_count + off + 510,
                                         6*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                    for (int k = 0; k < 4; k++) { ur[k] = unp[k].x; ui[k] = unp[k].y; }
                    for (int k = 0; k < 6; k++) { u2r[k] = unp2[k].x; u2i[k] = unp2[k].y; }
                } else {
                    float unp[4], unp2[6];
                    checkCuda(cudaMemcpy(unp, unpackeddata_gpu->gpuPtr() + (size_t)b*unpackedarrays_elem_count + off,
                                         4*sizeof(float), cudaMemcpyDeviceToHost));
                    checkCuda(cudaMemcpy(unp2, unpackeddata_gpu->gpuPtr() + (size_t)b*unpackedarrays_elem_count + off + 510,
                                         6*sizeof(float), cudaMemcpyDeviceToHost));
                    for (int k = 0; k < 4; k++) { ur[k] = unp[k]; ui[k] = 0.0f; }
                    for (int k = 0; k < 6; k++) { u2r[k] = unp2[k]; u2i[k] = 0.0f; }
                }
                checkCuda(cudaMemcpy(rot, complex_fringe_rotated_gpu->gpuPtr() + (size_t)sub*fftchannels*numrecordedbands + (size_t)b*fftchannels + 510,
                                     6*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                checkCuda(cudaMemcpy(spec, fftd_gpu->gpuPtr() + (size_t)sub*fftchannels*numrecordedbands + (size_t)b*fftchannels,
                                     4*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
                fprintf(stderr, "SPECDEBUG ds=%d datasec=%d datans=%d index=%d band=%d nearest=%d unp=%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f unp2=%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f rot=%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f spec=%.6f,%.6f;%.6f,%.6f;%.6f,%.6f;%.6f,%.6f\n",
                        datastreamindex, datasec, datans, idx, b, nearestSamples->ptr()[sub],
                        ur[0], ui[0], ur[1], ui[1], ur[2], ui[2], ur[3], ui[3],
                        u2r[0], u2i[0], u2r[1], u2i[1], u2r[2], u2i[2], u2r[3], u2i[3], u2r[4], u2i[4], u2r[5], u2i[5],
                        rot[0].x, rot[0].y, rot[1].x, rot[1].y, rot[2].x, rot[2].y,
                        rot[3].x, rot[3].y, rot[4].x, rot[4].y, rot[5].x, rot[5].y,
                        spec[0].x, spec[0].y, spec[1].x, spec[1].y, spec[2].x, spec[2].y, spec[3].x, spec[3].y);
            }
        }
    }
    
    float ms_copy1 = 0, ms_unpack = 0, ms_copy2 = 0, ms_pcal = 0, ms_rotate = 0, ms_fft = 0, ms_frac = 0;

    cudaEventElapsedTime(&ms_copy1, ev_start, ev_copy1);
    cudaEventElapsedTime(&ms_unpack, ev_copy1, ev_unpack);
    cudaEventElapsedTime(&ms_copy2, ev_unpack, ev_copy2);
    cudaEventElapsedTime(&ms_pcal, ev_copy2, ev_pcal);
    cudaEventElapsedTime(&ms_rotate, ev_pcal, ev_rotate);
    cudaEventElapsedTime(&ms_fft, ev_rotate, ev_fft);
    cudaEventElapsedTime(&ms_frac, ev_fft, ev_frac);

    // Filter out garbage/negative thread-clash timings BEFORE casting to long long
    t_copyto += (long long)((std::max(0.0f, ms_copy1) + std::max(0.0f, ms_copy2)) * 1000.0f);
    t_unpack += (long long)(std::max(0.0f, ms_unpack) * 1000.0f);
    t_pcal   += (long long)(std::max(0.0f, ms_pcal) * 1000.0f);
    t_rotate += (long long)(std::max(0.0f, ms_rotate) * 1000.0f);
    t_fft    += (long long)(std::max(0.0f, ms_fft) * 1000.0f);
    t_fracrotate += (long long)(std::max(0.0f, ms_frac) * 1000.0f);



    // ==========================================
    // 8. POST-PROCESSING (Host Autocorrelations)
    // ==========================================
    // This synchronise is really needed, as we need the GPU processing/memcpys to finish before we read the result
    // data in to the autocorrelation vectors
    auto post_start = high_resolution_clock::now();
    temp_autocorrelations_gpu->sync();

    

    // Copy over the autocorrs
    for (int i = 0; i < autocorrwidth; i++) {
        for (int j = 0; j < numrecordedbands; j++) {
            vectorCopy_cf32(
                    reinterpret_cast<const cf32 *>(&temp_autocorrelations_gpu->ptr()[(i * numrecordedbands * recordedbandchannels) + (j * recordedbandchannels)]),
                    autocorrelations[i][j],
                    recordedbandchannels
            );
        }
    }

    auto post_stop = high_resolution_clock::now();
    t_postprocess += duration_cast<microseconds>(post_stop - post_start).count();

    // 9. TOTAL PROCESSING TIME
    auto end_time = high_resolution_clock::now();
    t_total += duration_cast<microseconds>(end_time - begin_time).count();

    // TODO: the return value might need to change? Not sure how its used
    //return numfftsprocessed;
    return numBufferedFFTs;
}


//int GPUMode::set_invalid_data(int fftloop, int numBufferedFFTs, int startblock,
//                         int numblocks) {
//
//    size_t unpacked_size = buffer_payload_bytes * 8 / (config->getDNumBits(confindex, dsindex) * config->getDNumRecordedBands(confindex, dsindex));
//    // What's the largest number of FFTs we can fit?
//    cfg_numBufferedFFTs = (unpacked_size + fftchannels - 1) / fftchannels;	
//    fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
//    conj_fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
//    return numBufferedFFTs;
//}	


bool GPUMode::is_dataweight_valid(int subloopindex) {
    int status;

    if (dataweight[subloopindex] <= 0.0) {
        //printf("Data weight for subloopindex %d is %f, which is invalid. Setting fft outputs to 0.\n", subloopindex, dataweight[subloopindex]);
    //    for (int i = 0; i < numrecordedbands; i++) {
    //        status = vectorZero_cf32(fftd_gpu[i][subloopindex], recordedbandchannels);
    //        if (status != vecNoErr)
    //            csevere << startl << "Error trying to zero fftd_gpu when data is bad!" << endl;
    //        status = vectorZero_cf32(conjfftd_gpu[i][subloopindex], recordedbandchannels);
    //        if (status != vecNoErr)
    //            csevere << startl << "Error trying to zero fftd_gpu when data is bad!" << endl;
    //    }
        return false;
    }

    return true;
}

bool GPUMode::is_data_valid(int index, int subloopindex) {
    int status;
    const int validflagwordindex = index / FLAGS_PER_INT;
    const int validflagbitindex = index % FLAGS_PER_INT;
    const unsigned int validflagword = static_cast<unsigned int>(validflags[validflagwordindex]);
    const int validflagbit = ((validflagword >> validflagbitindex) & 0x01);
    const int reason_datalen = (datalengthbytes <= 1);
    const int reason_subint = (offsetseconds == INVALID_SUBINT);
    const int reason_validflag = (validflagbit == 0);

    // Check the data is valid for this index
    if (reason_datalen || reason_subint || reason_validflag) {
   
//        for (int i = 0; i < numrecordedbands; i++) {
//            status = vectorZero_cf32(fftd_gpu[i][subloopindex], recordedbandchannels);
//            if (status != vecNoErr)
//                csevere << startl << "Error trying to zero fftd_gpuhen data is bad!" << endl;
//            status = vectorZero_cf32(conjfftd_gpu[i][subloopindex], recordedbandchannels);
//            if (status != vecNoErr)
//                csevere << startl << "Error trying to zero fftd_gpu when data is bad!" << endl;
//        }
//        cerr << "Mode for DS " << datastreamindex << " is bailing out of index " << index << "/" << subloopindex << " which is scan " << currentscan << ", sec " << offsetseconds << ", ns " << offsetns << " because datalengthbytes is " << datalengthbytes << " and validflag was " << ((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) << endl;
        return false; //don't process crap data
    }

    // Check that the nearest sample is valid
    if (nearestSamples->ptr()[subloopindex] < -1 ||
        (((nearestSamples->ptr()[subloopindex] + fftchannels) / samplesperblock) * bytesperblocknumerator) / bytesperblockdenominator >
        datalengthbytes) {
 
//        for (int i = 0; i < numrecordedbands; i++) {
//            status = vectorZero_cf32(fftd_gpu[i][subloopindex], recordedbandchannels);
//            if (status != vecNoErr)
//                csevere << startl << "Error trying to zero fftd_gpu when data is bad!" << endl;
//            status = vectorZero_cf32(conjfftd_gpu[i][subloopindex], recordedbandchannels);
//            if (status != vecNoErr)
//                csevere << startl << "Error trying to zero fftd_gpu when data is bad!" << endl;
//        }
        return false;
    }

    return true;
}

// Env-gated per-FFT-window weight tracing for CPU-vs-GPU debugging: set
// DIFX_WEIGHT_DEBUG=<datasec> to emit one WDEBUG line per FFT window for all
// subints with datasec >= <datasec>. The format is identical to the one in
// CPUMode::process (cpumode.cpp) so the grepped logs diff directly.
static int weightDebugFrom()
{
    static int from = -2;
    if (from == -2) {
        const char* e = getenv("DIFX_WEIGHT_DEBUG");
        from = (e != NULL) ? atoi(e) : -1;
    }
    return from;
}

void GPUMode::set_weights(int subloopindex, int nframes, int *counts, int numBufferedFFTs) {

    int framesamples = config->getMultiplexedFramePayloadBytes(configindex, datastreamindex)*8/(config->getDNumBits(configindex, datastreamindex)*config->getDNumRecordedBands(configindex, datastreamindex)*config->getDDecimationFactor(configindex, datastreamindex));
    if (usecomplex)
      framesamples /= 2;

    // Not sure if this is still needed. Set to zero for now.
     unpackstartsamples = 0;
    // unpackstartsamples = nearestSamples->ptr()[subloopindex] - (nearestSamples->ptr()[subloopindex]% samplegranularity);
    //std::cout << "subloopindex = " << subloopindex << ", unpackstartsamples = " << unpackstartsamples <<  ", samplegranularity = " << samplegranularity << std::endl;
    //std::cout << "subloopindex = " << subloopindex << ", cfg_numBufferedFFTs = " << cfg_numBufferedFFTs << std::endl;
    // Clear the perbandweights for this subloopindex

    //fprintf(stderr, "set_weights DS=%d sub=%d numBufferedFFTs(arg via caller)=? cfg=%d configNBF=%d perbandweights=%p\n",
    //    datastreamindex, subloopindex, cfg_numBufferedFFTs,
    //   config->getNumBufferedFFTs(configindex), (void*)perbandweights);    
    //std::cout << "numrecordedbands = " << numrecordedbands << std::endl;
    if(perbandweights)
    {
        for(int b = 0; b < numrecordedbands; ++b)
        {
            perbandweights[subloopindex][b] = 0.0;
        }
    }

    //std::cout << "fftloop = " << fftloop << ", subloopindex = " << subloopindex << ", numBufferedFFTs = " << numBufferedFFTs << ", startblock = " << startblock << std::endl;
    //int validity_index =  subloopindex + startblock;
    //if (startblock != 0) {
        //std::cout << "subloopindex = " << subloopindex << ", validity_index = " << validity_index << std::endl;
    //}
    if (!is_data_valid(subloopindex, subloopindex)) {

        // since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
        dataweight[subloopindex] = 0.0;

        gValidSamples->ptr()[subloopindex] = false;
        if (weightDebugFrom() >= 0 && datasec >= weightDebugFrom())
            fprintf(stderr, "WDEBUG ds=%d datasec=%d datans=%d index=%d nearest=%d weight=0.000000000 valid=0 reason=rejected\n",
                    datastreamindex, datasec, datans, subloopindex, nearestSamples->ptr()[subloopindex]);
        return;
    }
    //std::cout << "Data is valid for subloopindex " << subloopindex << std::endl;
    gValidSamples->ptr()[subloopindex] = true;

    // Frames at or beyond nframes were not delivered in this subintegration:
    // valid_frames (and the unpacked buffers) hold stale values from the
    // previous subint there, so such frames must count as invalid. This gives
    // an FFT window straddling the end of the delivered data the same
    // fractional weight the CPU path's unpacker returns (its missing samples
    // are zeroed after unpacking, also matching the CPU).
    auto frame_ok = [&](int frame) -> float {
        return (frame >= 0 && frame < nframes && valid_frames->ptr()[frame]) ? 1.0f : 0.0f;
    };

    if (nearestSamples->ptr()[subloopindex] == -1) {
        nearestSamples->ptr()[subloopindex] = 0;
        dataweight[subloopindex] = 1.0;
        cerr << "Why is this happening?" << std::endl;      // I'm not sure what case this branch is for
        abort();
    } else if (nearestSamples->ptr()[subloopindex] < unpackstartsamples ||
               nearestSamples->ptr()[subloopindex] > unpackstartsamples + unpacksamples - fftchannels ||
               subloopindex + 1 == numBufferedFFTs) {
        //std::cout << "Entered standard path subloopindex = " << subloopindex << ", nearestSamples = " << nearestSamples->ptr()[subloopindex] << ", unpackstartsamples = " << unpackstartsamples << ", unpacksamples = " << unpacksamples << ", fftchannels = " << fftchannels << std::endl;
        // Standard path. TODO: above condition can be simplified I think
        int start_frame = nearestSamples->ptr()[subloopindex] / framesamples;
        // The last window of the batch has no successor to consult, so use
        // its own extent to locate its final frame.
        int end_frame;
        if (subloopindex + 1 == numBufferedFFTs)
            end_frame = (nearestSamples->ptr()[subloopindex] + fftchannels - 1) / framesamples;
        else
            end_frame = (nearestSamples->ptr()[subloopindex + 1] - 1) / framesamples;
        if (start_frame == end_frame) {
            // This FFT window does not cross a frame boundary
            dataweight[subloopindex] = frame_ok(start_frame);
        } else if (start_frame + 1 == end_frame) {
            // Crosses frame boundary: set weight proportional to occupancy in each frame
            float frac_first_frame = (float)(end_frame * framesamples - nearestSamples->ptr()[subloopindex]) / (float)fftchannels;
            dataweight[subloopindex] = (frac_first_frame) * frame_ok(start_frame) + (1 - frac_first_frame) * frame_ok(end_frame);
        } else {
            cerr << "FFT window somehow spans more than two frames. This is suspicious to me but maybe allowed?" << std::endl;
            abort();
        };
    }
    // Need to access samplegranularity which was out of scope
    
    
    gSampleIndexes->ptr()[subloopindex] = nearestSamples->ptr()[subloopindex] - unpackstartsamples;

    if (!is_dataweight_valid(subloopindex)) {
        //std::cout << "Data weight is not valid for subloopindex " << subloopindex << std::endl;
        gValidSamples->ptr()[subloopindex] = false;
    } else {
	//printf("numrecordedfreqs = %d \n",numrecordedfreqs);
        //printf("numrecordedbands  = %d \n",numrecordedbands);
        //exit(0);	
        // Todo: This can definitely be cleaned up and improved
        for (int i = 0; i < numrecordedfreqs; i++) {
            // PWCR numrecordedbands = 2 for the test; but e.g. 8 is very realistical
            // Loop over all recorded bands looking for the matching frequency we should be dealing with
            int count = 0;
            for (int j = 0; j < numrecordedbands; j++) {
                // For upper sideband bands, normally just need to copy the fftd channels.
                // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back

                if (config->matchingRecordedBand(configindex, datastreamindex, i, j)) {
                    indices->ptr()[(i * MAX_INDICIES) + count++] = j;
		            counts[i] = count;
                    // At this point in the code the array fftd_gpu[j] contains complex-valued voltage spectra with the following properties:
                    //
                    // 1. The zero element corresponds to the lowest sky frequency.  That is:
                    //    fftd_gpu[j][0] = Local Oscillator Frequency              (for Upper Sideband)
                    //    fftd_gpu[j][0] = Local Oscillator Frequency - bandwidth  (for Lower Sideband)
                    //    fftd_gpu[j][0] = Local Oscillator Frequency - bandwidth  (for Complex Lower Sideband)
                    //    fftd_gpu[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Upper Sideband)
                    //    fftd_gpu[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Lower Sideband)
                    //
                    // 2. The frequency increases monotonically with index
                    //
                    // 3. The last element of the array corresponds to the highest sky frequency minus the spectral resolution.
                    //    (i.e., the first element beyond the array bound corresponds to the highest sky frequency)

                    //store the weight for the autocorrelations
                    //fprintf(stderr, "set_weights ds=%d sub=%d perbandweights=%p\n",
                    //        datastreamindex, subloopindex, (void*)perbandweights);
//fprintf(stderr, "set_weights ds=%d sub=%d j=%d nbands=%d acw=%d w0=%p pbw=%p pbw[sub]=%p\n",
//        datastreamindex, subloopindex, j, numrecordedbands, autocorrwidth,
//        (void*)(weights ? weights[0] : nullptr),
//        (void*)perbandweights,
//        (void*)(perbandweights ? perbandweights[subloopindex] : nullptr));
                    if (perbandweights) {
                        weights[0][j] += perbandweights[subloopindex][j];
                    } else {
                        weights[0][j] += dataweight[subloopindex];
                    }
                }
            }

            if (count > 1) {
                //store the weights
                if (perbandweights) {
                    weights[1][indices->ptr()[(i * MAX_INDICIES)]] += perbandweights[subloopindex][indices->ptr()[(i * MAX_INDICIES)]] *
                                                     perbandweights[subloopindex][indices->ptr()[(i * MAX_INDICIES) + 1]];
                    weights[1][indices->ptr()[(i * MAX_INDICIES) + 1]] += perbandweights[subloopindex][indices->ptr()[(i * MAX_INDICIES)]] *
                                                     perbandweights[subloopindex][indices->ptr()[(i * MAX_INDICIES) + 1]];
                } else {
                    weights[1][indices->ptr()[(i * MAX_INDICIES)]] += dataweight[subloopindex];
                    weights[1][indices->ptr()[(i * MAX_INDICIES) + 1]] += dataweight[subloopindex];
                }
            }
        }
    }

    if (weightDebugFrom() >= 0 && datasec >= weightDebugFrom())
        fprintf(stderr, "WDEBUG ds=%d datasec=%d datans=%d index=%d nearest=%d weight=%.9f valid=%d reason=ok\n",
                datastreamindex, datasec, datans, subloopindex, nearestSamples->ptr()[subloopindex],
                dataweight[subloopindex], gValidSamples->ptr()[subloopindex] ? 1 : 0);
}

void GPUMode::calculatePre_cpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
    int startIndex = fftloop * numBufferedFFTs + startblock;
    int endIndex = startblock + numblocks;

    // Always initialize the full batch to avoid carrying stale values
    // from a previous process_gpu() call when this pass is invalid/short.
    for (int subloopindex = 0; subloopindex < numBufferedFFTs; subloopindex++) {
        nearestSamples->ptr()[subloopindex] = -1;
        gFracSampleError->ptr()[subloopindex] = 0.0f;
    }

    // Invalid subints are filtered later by is_data_valid(); keep nearestSamples
    // in a sentinel state so debug output doesn't show misleading large negatives.
    if (offsetseconds == INVALID_SUBINT) {
        gFracSampleError->copyToDevice();
        nearestSamples->copyToDevice();
        return;
    }

    for (int subloopindex = 0; subloopindex < numBufferedFFTs; subloopindex++) {
        int index = startIndex + subloopindex;
        if (index >= endIndex)
            break; // may not have to fully complete last fftloop

        double fftcentre = index + 0.5;
        double averagedelay = interpolator[0] * fftcentre * fftcentre + interpolator[1] * fftcentre + interpolator[2];
        double fftstartmicrosec = index * fftchannels * sampletime; //CHRIS CHECK
        double starttime = (offsetseconds - datasec) * 1000000.0 +
                (double) (static_cast<long long>(offsetns) - static_cast<long long>(datans)) / 1000.0 + fftstartmicrosec -
                           averagedelay;
        nearestSamples->ptr()[subloopindex] = int(starttime / sampletime + 0.5);

        double nearestsampletime = nearestSamples->ptr()[subloopindex] * sampletime;
        gFracSampleError->ptr()[subloopindex] = float(starttime - nearestsampletime);
    }

    // Start copying the fracSampleErrors and nearestSamples to the gpu
    gFracSampleError->copyToDevice();
    nearestSamples->copyToDevice();
}

__global__ void gpu_fringeRotation(
        cuFloatComplex* const dest,
        float **const src,
        const double* const interpolator,
        const int* const sampleIndexes,
        const bool* const validSamples,
        const double* const lofreqs,
        const double* const recordedfreqlooffsets,
        double sampletime,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels
    ) {
    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numrecordedbands index [0 .. numrecordedbands]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numrecordedbands size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t bandindex = threadIdx.x;
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
    const size_t numrecordedbands = blockDim.x;

    if (channelindex >= fftchannels) {
        return;
    }

    
    // Calculate the destination index
    const size_t destIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

    // Calculate the source index and get the source value
    const size_t srcIndex = bandindex;
    const float srcVal = src[srcIndex][sampleIndexes[subloopindex] + channelindex];

    /* The actual calculation that is going on for the linear case is as follows:

     Calculate complexrotator[j]  (for j = 0 to fftchanels-1) as:

     complexrotator[j] = exp( 2 pi i * (A*j + B) )

     where:

     A = a*lofreq/fftchannels - sampletime*1.0e-6*recordedfreqlooffsets[i]
     B = b*lofreq + fraclofreq*integerdelay - recordedfreqlooffsets[i]*fracwalltime - fraclooffset*intwalltime

     And a, b are computed outside the recordedfreq loop (variable i)
    */

    // Calculate littleA/B
    double d0 = interpolator[0] * (double) index * (double) index + interpolator[1] * (double) index + interpolator[2];
    double d1 = interpolator[0] * ((double) index + 0.5) * ((double) index + 0.5) + interpolator[1] * ((double) index + 0.5) + interpolator[2];
    double d2 = interpolator[0] * ((double) index + 1) * ((double) index + 1) + interpolator[1] * ((double) index + 1) + interpolator[2];

    double a = d2 - d0;
    double b = d0 + (d1 - (a * 0.5 + d0)) / 3.0;

    // Calculate BigA/B
//    double bigAval = a * lofreqs[numrecordedfreq] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[numrecordedfreq];
//    double bigBval = b * lofreqs[numrecordedfreq];

//    double bigAval = a * lofreqs[0] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[0];
//    double bigBval = b * lofreqs[0];

    double bigAval = a * lofreqs[bandindex] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[bandindex];
    double bigBval = b * lofreqs[bandindex];


    // Calculate
    double bigB_reduced = bigBval - int(bigBval);
    double exponent = (bigAval * (double) channelindex + bigB_reduced);
    exponent -= int(exponent);
    cuFloatComplex cr;
    __sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
    dest[destIndex] = cuCmulf(c, cr);
  

//    if (srcVal != 0) {
       //printf("lofreqs = %lf \n",lofreqs[0]);
       //printf("Using src[%lu][%lu] = %f to get dest[%lu] = %f + %fi  and lofreq = %lf\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal, destIndex, dest[destIndex].x, dest[destIndex].y, lofreqs[bandindex]);
       //printf("src[%lu][%lu] = %f\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal);

//    }

}


__global__ void gpu_complex_fringeRotation(
        cuFloatComplex* const dest,
        cuFloatComplex **const src,
        const double* const interpolator,
        const int* const sampleIndexes,
        const bool* const validSamples,
        const double* const lofreqs,
        const double* const recordedfreqlooffsets,
        double sampletime,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels
    ) {
    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numrecordedbands index [0 .. numrecordedbands]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numrecordedbands size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t bandindex = threadIdx.x;
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
    const size_t numrecordedbands = blockDim.x;

    if (channelindex >= fftchannels) {
        return;
    }

    
    // Calculate the destination index
    const size_t destIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

    // Calculate the source index and get the source value
    const size_t srcIndex = bandindex;
    const cuFloatComplex srcVal = src[srcIndex][sampleIndexes[subloopindex] + channelindex];

    /* The actual calculation that is going on for the linear case is as follows:

     Calculate complexrotator[j]  (for j = 0 to fftchanels-1) as:

     complexrotator[j] = exp( 2 pi i * (A*j + B) )

     where:

     A = a*lofreq/fftchannels - sampletime*1.0e-6*recordedfreqlooffsets[i]
     B = b*lofreq + fraclofreq*integerdelay - recordedfreqlooffsets[i]*fracwalltime - fraclooffset*intwalltime

     And a, b are computed outside the recordedfreq loop (variable i)
    */

    // Calculate littleA/B
    double d0 = interpolator[0] * (double) index * (double) index + interpolator[1] * (double) index + interpolator[2];
    double d1 = interpolator[0] * ((double) index + 0.5) * ((double) index + 0.5) + interpolator[1] * ((double) index + 0.5) + interpolator[2];
    double d2 = interpolator[0] * ((double) index + 1) * ((double) index + 1) + interpolator[1] * ((double) index + 1) + interpolator[2];

    double a = d2 - d0;
    double b = d0 + (d1 - (a * 0.5 + d0)) / 3.0;

    // Calculate BigA/B
//    double bigAval = a * lofreqs[numrecordedfreq] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[numrecordedfreq];
//    double bigBval = b * lofreqs[numrecordedfreq];

//    double bigAval = a * lofreqs[0] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[0];
//    double bigBval = b * lofreqs[0];

    double bigAval = a * lofreqs[bandindex] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[bandindex];
    double bigBval = b * lofreqs[bandindex];


    // Calculate
    double bigB_reduced = bigBval - int(bigBval);
    double exponent = (bigAval * (double) channelindex + bigB_reduced);
    exponent -= int(exponent);
    cuFloatComplex cr;
    __sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    dest[destIndex] = cuCmulf(srcVal, cr);
  

//    if (srcVal != 0) {
       //printf("lofreqs = %lf \n",lofreqs[0]);
       //printf("Using src[%lu][%lu] = %f to get dest[%lu] = %f + %fi  and lofreq = %lf\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal, destIndex, dest[destIndex].x, dest[destIndex].y, lofreqs[bandindex]);
       //printf("src[%lu][%lu] = %f\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal);

//    }

}


// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
__device__ void atomicAddFloatComplex(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    auto *x = (float*) a;
    auto *y = x + 1;
    //use atomicAdd for float variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}


__global__ void gpu_pcalextraction(
    cuFloatComplex* const complexunpacked,
    float **const unpackedarrays,
    const double* const interpolator,
    const int* const sampleIndexes,
    const bool* const validSamples,
    const double* const lofreqs,
    const double* const recordedfreqlooffsets,
    double sampletime,
    int fftloop,
    int startblock,
    int numblocks,
    size_t fftchannels,
    const int* const nearestSamples,
    int datasamples,
    double bandwidth_hz,
    double pcal_spacing_hz,
    const int* pcal_offsets_hz,
    float* pcal_output_real,
    int pcal_bin_stride_length,
    int* N_pcal_bins
) {
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) return;

    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) return;

    const int sample_index = sampleIndexes[subloopindex];
    if (sample_index < 0) return;

    int bandindex = blockIdx.y; 
    int n_bins = N_pcal_bins[bandindex];

    int sampleoffset = datasamples + sample_index;
    int pcal_index = sampleoffset % n_bins;
    
    // Resolve 2D array offset once per thread
    float const* src = unpackedarrays[bandindex] + sample_index;
    float* dst = pcal_output_real + (bandindex * pcal_bin_stride_length);

    // Each thread computes EXACTLY ONE bin. 
    // ZERO shared memory, ZERO atomics in the loop!
    for (int bin = threadIdx.x; bin < n_bins; bin += blockDim.x) {
        float sum = 0.0f;
        
        // Find the starting channel index for this specific bin
        int start_c = (bin - pcal_index + n_bins) % n_bins;
        
        // Stride through the array picking up ONLY this thread's bin data.
        // Because threads request adjacent start_c, this is a PERFECTLY coalesced warp read!
        for (size_t c = start_c; c < fftchannels; c += n_bins) {
            sum += src[c];
        }
        
        // Only ONE global memory write per bin per block!
        atomicAdd(&dst[bin], sum); 
    }
}



__global__ void gpu_pcalextraction_complex(
    cuFloatComplex* const complexunpacked,
    cuFloatComplex **const unpackedarrays,
    const double* const interpolator,
    const int* const sampleIndexes,
    const bool* const validSamples,
    const double* const lofreqs,
    const double* const recordedfreqlooffsets,
    double sampletime,
    int fftloop,
    int startblock,
    int numblocks,
    size_t fftchannels,
    const int* const nearestSamples,
    int datasamples,
    double bandwidth_hz,
    double pcal_spacing_hz,
    const int* pcal_offsets_hz,
    cuFloatComplex* pcal_output_complex,
    int pcal_bin_stride_length,
    int* N_pcal_bins
) {
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) return;

    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) return;

    const int sample_index = sampleIndexes[subloopindex];
    if (sample_index < 0) return;

    int bandindex = blockIdx.y; 
    int n_bins = N_pcal_bins[bandindex];

    int sampleoffset = datasamples + sample_index;
    int pcal_index = sampleoffset % n_bins;
    
    // Resolve 2D array offset once per thread
    cuFloatComplex const* src = unpackedarrays[bandindex] + sample_index;
    cuFloatComplex* dst = pcal_output_complex + (bandindex * pcal_bin_stride_length);

    // Each thread computes EXACTLY ONE bin. 
    // ZERO shared memory, ZERO atomics in the loop!
    for (int bin = threadIdx.x; bin < n_bins; bin += blockDim.x) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        // Find the starting channel index for this specific bin
        int start_c = (bin - pcal_index + n_bins) % n_bins;
        
        // Stride through the array picking up ONLY this thread's bin data.
        // Because threads request adjacent start_c, this is a PERFECTLY coalesced warp read!
        for (size_t c = start_c; c < fftchannels; c += n_bins) {
            sum = cuCaddf(sum, src[c]);
        }
        
        // Only ONE global memory write per bin per block!
        atomicAddFloatComplex(&dst[bin], sum); 
    }
}




void GPUMode::pcalExtraction(int fftloop, int numBufferedFFTs, int startblock, int numblocks) { 


    double bandwidth_hz = 1e6*recordedbandwidth;
    double pcal_spacing_hz = 1e6*config->getDPhaseCalIntervalMHz(configindex, datastreamindex);

    // N_pcal_bins_max is half of the stride length. Calculate the required shared memory size.
    //int max_bins = pcal_bin_stride_length / 2;
    //size_t shared_mem_size = max_bins * sizeof(float);


 //   gpu_pcalextraction<<<
 //       dim3(numBufferedFFTs, 1),
 //       dim3(numrecordedbands, 1),
 //       0, cuStream
 //   >>>
    int threads_per_block = 256;  
    if (!usecomplex) {
        gpu_pcalextraction<<<
            dim3(numBufferedFFTs, numrecordedbands), 
            dim3(threads_per_block),                 
            0, cuStream // No shared memory needed!
        >>>             
                (
                        complex_fringe_rotated_gpu->gpuPtr(),
                        unpackedarrays_gpu->gpuPtr(),
                        gInterpolator->gpuPtr(),
                        gSampleIndexes->gpuPtr(),
                        gValidSamples->gpuPtr(),
                        gLoFreqs->gpuPtr(),
                        grecordedfreqlooffsets->gpuPtr(),
                        sampletime,
                        fftloop,
                        startblock,
                        numblocks,
                        fftchannels,
                        nearestSamples->gpuPtr(),
                        datasamples,
                        bandwidth_hz,
                        pcal_spacing_hz,
                        pcal_offsets_hz->gpuPtr(),
                        pcal_output_real->gpuPtr(),
                        pcal_bin_stride_length,
                        N_pcal_bins->gpuPtr()
               );
    } else {
        gpu_pcalextraction_complex<<<
            dim3(numBufferedFFTs, numrecordedbands), 
            dim3(threads_per_block),                 
            0, cuStream // No shared memory needed!
        >>>             
                (
                        complex_fringe_rotated_gpu->gpuPtr(),
                        complex_unpackedarrays_gpu->gpuPtr(),
                        gInterpolator->gpuPtr(),
                        gSampleIndexes->gpuPtr(),
                        gValidSamples->gpuPtr(),
                        gLoFreqs->gpuPtr(),
                        grecordedfreqlooffsets->gpuPtr(),
                        sampletime,
                        fftloop,
                        startblock,
                        numblocks,
                        fftchannels,
                        nearestSamples->gpuPtr(),
                        datasamples,
                        bandwidth_hz,
                        pcal_spacing_hz,
                        pcal_offsets_hz->gpuPtr(),
                        pcal_output_complex->gpuPtr(),
                        pcal_bin_stride_length,
                        N_pcal_bins->gpuPtr()
               );
            }
    
}






void GPUMode::fringeRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {

    // At this point we have
    // * Unpacked data on GPU
    // * Output buffer on GPU ready to go
    // * Sample indexes in the unpacked data
    // * BigA and BigB
    // * Which samples are valid - ie that we need to operate on

    // numBufferedFFTs(blockIdx.x) * (numrecordedbands(threadIdx.x) * fftchannels(threadIdx.y))
    size_t fftchannels_block = fftchannels;
    size_t fftchannels_grid = 1;

    size_t divisor = cudaMaxThreadsPerBlock / numrecordedbands;
    if (fftchannels > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = (fftchannels / divisor);

        if (fftchannels % divisor != 0) {
            fftchannels_grid++;
        }
    }
    //  For LSB data, gLoFreqs needs to have already been corrected for the fact that we will convert to 
    // USB in unpacking. This means than loFreq = loFreq - bandwidth.
    if (usecomplex) {
        gpu_complex_fringeRotation<<<
            dim3(numBufferedFFTs, fftchannels_grid),
            dim3(numrecordedbands,fftchannels_block),
            0, cuStream
        >>>
            (
                    complex_fringe_rotated_gpu->gpuPtr(),
                    complex_unpackedarrays_gpu->gpuPtr(),
                    gInterpolator->gpuPtr(),
                    gSampleIndexes->gpuPtr(),
                    gValidSamples->gpuPtr(),
                    gLoFreqs->gpuPtr(),
                    grecordedfreqlooffsets->gpuPtr(),
                    sampletime,
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels
            );
    } else {
    gpu_fringeRotation<<<
        dim3(numBufferedFFTs, fftchannels_grid),
        dim3(numrecordedbands,fftchannels_block),
        0, cuStream
    >>>
            (
                    complex_fringe_rotated_gpu->gpuPtr(),
                    unpackedarrays_gpu->gpuPtr(),
                    gInterpolator->gpuPtr(),
                    gSampleIndexes->gpuPtr(),
                    gValidSamples->gpuPtr(),
                    gLoFreqs->gpuPtr(),
                    grecordedfreqlooffsets->gpuPtr(),
                    sampletime,
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels
            );
    }
}



__global__ void gpu_resultsrotatorMultiply(
        cuFloatComplex* const gpufftd_gpu,
        cuFloatComplex* const gpuconjfftd_gpu,
        cuFloatComplex* const autocorrelations,
        const float* const fracSampleError,
        const bool* const validSamples,
        const unsigned int* const indices,
        const double* const recordedfreqclockoffsets,
        const double* const recordedfreqclockoffsetsdelta,
        const double recordedbandwidth,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels,
        size_t recordedbandchannels,
        size_t numrecordedbands,
        size_t numrecordedfreqs,
	bool calccrosspolautocorrs,
	int* counts_gpu
    ) {

    //for (int ii=0; ii<numrecordedfreqs; ii++){
    //    printf("counts_gpu[%d] = %d\n",ii,counts_gpu[ii]);
    //}

    // numBufferedFFTs(blockIdx.x) * fftchannels(threadIdx.x)

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t channelindex = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (channelindex >= recordedbandchannels) {
        return;
    }

    for (size_t bandindex = 0; bandindex < numrecordedbands; bandindex++) {
        /* Creating a fractional sample rotation array
         *  The actual calculation being performed is as follows:
         *  Assume we know the frequency of every FFT output channel, and it is stored in an array of length fftchannels, called channelfreq
         *  then for every frequency subband f (in the range 0 … recordedbandchannels), calculate the slope as:
         *  A = fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
         *  (for the second polarisation, a is identical except subtracting recordedfreqclockoffsetsdelta[f]/2)
         * then calculate complexrotator[j]  (for j = 0 to fftchannels-1) as:
         * complexrotator[j] = exp( 2 pi i * (A*fftchannels[j]) )
         *
         * So how is fftchannels calculated? For “regular data” it is as follows (for j = 0 to fftchannels-1)
         * fftchannels[j] = recordedbandwidth * j / fftchannels
         * For lower sideband data it is:
         * fftchannels[j] = -recordedbandwidth * j / fftchannels
         * For double sideband data it is:
         * fftchannels[j] = recordedbandwidth * j / fftchannels - recordedbandwidth/2.0
        */

        // todo: Move recorded freq out of the kernel as a dim?
        const size_t dataIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

        // Calculate fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
        double bigAval = fracSampleError[subloopindex] - recordedfreqclockoffsets[bandindex] + recordedfreqclockoffsetsdelta[bandindex] / 2;

        // Calculate fftchannels[j] = recordedbandwidth * j / fftchannels
        double subFreq = recordedbandwidth * (double) channelindex / (double) recordedbandchannels;

        // Calculate
        double exponent = bigAval * subFreq;
        exponent -= int(exponent);
        cuFloatComplex cr;
        __sincosf(TWO_PI * exponent, &cr.y, &cr.x);
        gpufftd_gpu[dataIndex] = cuCmulf(gpufftd_gpu[dataIndex], cr);

        // do the conjugation
        gpuconjfftd_gpu[dataIndex] = cuConjf(gpufftd_gpu[dataIndex]);

        // do the autocorrelation (skipping Nyquist channel)
        // Calculate the destination index
        const size_t autocorrIndex = (bandindex * recordedbandchannels) + channelindex;
        atomicAddFloatComplex(&autocorrelations[autocorrIndex], cuCmulf(gpufftd_gpu[dataIndex], gpuconjfftd_gpu[dataIndex]));
    }

    for (size_t recordedfreq = 0; recordedfreq < numrecordedfreqs; recordedfreq++) {
        if (calccrosspolautocorrs && counts_gpu[recordedfreq] > 1) {

            size_t fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;

	        size_t conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;

	        atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex], cuCmulf(gpufftd_gpu[fftIndex], gpuconjfftd_gpu[conjIndex]));
	    
            fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;
            conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;

            atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 1] * recordedbandchannels) + channelindex], cuCmulf(gpufftd_gpu[fftIndex], gpuconjfftd_gpu[conjIndex]));
        }
    }
}

void GPUMode::fractionalRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks, bool calccrosspolautocorrs, int* counts) {
    // At this point we have
    // * FFT results on GPU
    // * subchannelfreqs
    // * Which samples are valid - ie that we need to operate on

    // numBufferedFFTs(blockIdx.x) * fftchannels(threadIdx.x)
    size_t fftchannels_block = recordedbandchannels;
    size_t fftchannels_grid = 1;

    size_t divisor = cudaMaxThreadsPerBlock;
    if (recordedbandchannels > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = recordedbandchannels / divisor;

        if (recordedbandchannels % divisor != 0) {
            fftchannels_grid++;
        }
    }
        //printf("fracRot indices[0] = %u \n",indices->ptr()[0]);
	//printf("fracRot indices[1] = %u \n",indices->ptr()[1]);
	//exit(0);
        //gpu_resultsrotatorMultiply<<<1, 1, 0, cuStream>>>
    //GpuMemHelper<int> *counts_gpu;
    //counts_gpu = new GpuMemHelper<int>(numrecordedfreqs, cuStream); 
    for (int ii=0; ii < numrecordedfreqs; ii++) {
        counts_gpu->ptr()[ii] = counts[ii];
        //printf("counts[%d] = %d\n",ii,counts[ii]);
        //printf("counts_gpu[%d] = %d\n",ii,counts_gpu->ptr()[ii]);
	}
	counts_gpu->copyToDevice(); // Check what these are for and if they are necessary!

	gpu_resultsrotatorMultiply<<<dim3(numBufferedFFTs, fftchannels_grid), dim3(fftchannels_block), 0, cuStream>>>
           (
                    fftd_gpu->gpuPtr(),
                    conj_fftd_gpu->gpuPtr(),
                    temp_autocorrelations_gpu->gpuPtr(),
                    gFracSampleError->gpuPtr(),
                    gValidSamples->gpuPtr(),
                    indices->gpuPtr(),
                    grecordedfreqclockoffsets->gpuPtr(),
                    grecordedfreqclockoffsetsdelta->gpuPtr(),
                    recordedbandwidth,
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels,
                    recordedbandchannels,
                    numrecordedbands,
                    numrecordedfreqs,
		    calccrosspolautocorrs,
		    counts_gpu->gpuPtr()
            );

    // We should zero the first channel (lowest frequency) of any LSB bands. Of both ffd_gpu and conj_fftd_gpu
    // In future, would be more efficient to do this at visibility.cpp, just prior to writing to disk (if either band in the baseline is LSB)

    // Start copying the autocorrelations back to the host
    temp_autocorrelations_gpu->copyToHost();
    // This delete prevents a memory but it introduces a huge performance overhead
    //delete counts_gpu;
}

void GPUMode::runFFT() {
    checkCufft(cufftExecC2C(fft_plan, complex_fringe_rotated_gpu->gpuPtr(), fftd_gpu->gpuPtr(), CUFFT_FORWARD));
}
