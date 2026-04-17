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
#include "mathutil.h"

using namespace std::chrono;

const int MAX_INDICIES = 10;

__global__ void gpu_allocate_unpacked(float** arrays, float* data, int nchan, int dlen) {
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
    std::cout << "Constructing a new GPUMode" << std::endl;
    auto start = high_resolution_clock::now();

    size_t buffer_payload_bytes = (config->getMaxDataBytes() / config->getFrameBytes(confindex, dsindex)) * config->getFramePayloadBytes(confindex, dsindex);
    size_t unpacked_size = buffer_payload_bytes * 8 / (config->getDNumBits(confindex, dsindex) * config->getDNumRecordedBands(confindex, dsindex));
    // What's the largest number of FFTs we can fit?
    cfg_numBufferedFFTs = (unpacked_size + fftchannels - 1) / fftchannels;
    //std::cout << "Working on " << cfg_numBufferedFFTs << " FFTs" << std::endl;
    // cfg_numBufferedFFTs = config->getNumBufferedFFTs(confindex);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties( &prop, 0));

    checkCuda(cudaStreamCreate(&cuStream));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    //std::cout << "fftchannels " << fftchannels << std::endl;

    complexunpacked_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, true);
    estimatedbytes_gpu += complexunpacked_gpu->size();

    fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
    estimatedbytes_gpu += fftd_gpu->size();

    conj_fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
    estimatedbytes_gpu += conj_fftd_gpu->size();

    temp_autocorrelations_gpu = new GpuMemHelper<cuFloatComplex>(autocorrwidth * numrecordedbands * recordedbandchannels, cuStream);
    estimatedbytes_gpu += temp_autocorrelations_gpu->size();


    // Unpacked data only allocated on GPU
    unpackedarrays_gpu = new GpuMemHelper<float*>(numrecordedbands, cuStream, true);
    unpackeddata_gpu = new GpuMemHelper<float>(numrecordedbands * unpacked_size, cuStream, true);
    //std::cout << "Unpacked data size: " << numrecordedbands * unpacked_size << std::endl;
    

    // Make sure these are allocated
    unpackeddata_gpu->sync();

    // Now launch a kernel to set up the arrays on the GPU
    gpu_allocate_unpacked<<<1, 1, 0, cuStream>>>(unpackedarrays_gpu->gpuPtr(), unpackeddata_gpu->gpuPtr(), numrecordedbands, unpacked_size);

    estimatedbytes_gpu += unpackedarrays_gpu->size();
    estimatedbytes_gpu += unpackeddata_gpu->size();

    gSampleIndexes = new GpuMemHelper<int>(cfg_numBufferedFFTs, cuStream);
    gValidSamples = new GpuMemHelper<bool>(cfg_numBufferedFFTs, cuStream);


    gInterpolator = new GpuMemHelper<double>(interpolator, 3, cuStream);
    gFracSampleError = new GpuMemHelper<float>(cfg_numBufferedFFTs, cuStream);
    gLoFreqs = new GpuMemHelper<double>(numrecordedbands, cuStream);


    //printf("MAX_INDICIES = %d \n", MAX_INDICIES);
    //printf("numrecordedfreqs %d \n",numrecordedfreqs);
    //exit(0);    
    indices = new GpuMemHelper<unsigned int>((MAX_INDICIES * numrecordedfreqs), cuStream);
    for (auto i = 0; i < (MAX_INDICIES * numrecordedfreqs); i++) {
        indices->ptr()[i] = 0xffffffff;
    }
    grecordedfreqclockoffsets = new GpuMemHelper<double>(numrecordedbands, cuStream);
    grecordedfreqclockoffsetsdelta = new GpuMemHelper<double>(numrecordedbands, cuStream);
    grecordedfreqlooffsets = new GpuMemHelper<double>(numrecordedbands, cuStream);
    // Copy the lofreq and freq clock offset values to the GPU
    std::cout << "numrecordedfreqs = " << numrecordedfreqs << std::endl;
    std::cout << "numrecordedbands = " << numrecordedbands << std::endl;
    //    exit(0);
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
        pcal_bin_stride_length = N_pcal_bins_max*2;  // *2 to avoid buffer wraps in the long-term integration method
        pcal_offsets_hz->copyToDevice();
        N_pcal_bins->copyToDevice();
        pcal_output_real = new GpuMemHelper<float>(numrecordedbands*pcal_bin_stride_length, cuStream);     
        for (size_t ii=0; ii<numrecordedbands*pcal_bin_stride_length; ii++) {
            pcal_output_real->ptr()[ii] = 0.0;
        }
        pcal_output_real->copyToDevice();
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

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //cout << "GPUMode(): " << duration.count() << endl;
    constructor_time = high_resolution_clock::now();
}

static unsigned long long avg_unpack;
static unsigned long long avg_copyto;
static unsigned long long avg_rotate;
static unsigned long long avg_fft;
static unsigned long long avg_fracrotate;
static unsigned long long avg_postprocess;
static unsigned long long processing_time;
static bool printed_unpack_debug = false;

int calls = 0;
static int debug_dataweight_gpu_prints = 0;
static int debug_dataweight_gpu_invalid_prints = 0;

GPUMode::~GPUMode() {
    auto start = high_resolution_clock::now();
    std::cout << "Starting destructor" << std::endl;
    delete complexunpacked_gpu;
    delete fftd_gpu;
    delete conj_fftd_gpu;
    delete temp_autocorrelations_gpu;
    delete unpackeddata_gpu;
    delete unpackedarrays_gpu;

    delete gSampleIndexes;
    delete gValidSamples;
    delete gInterpolator;
    delete gFracSampleError;

    delete nearestSamples;


    if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) { 
        delete pcal_offsets_hz;
        delete pcal_output_real;
        delete N_pcal_bins;
    }
    printf("pcal_output_real_gpu_mode \n");
    if (pcal_output_real_gpu_mode != nullptr) {
        //delete pcal_output_real_gpu_mode;
        pcal_output_real_gpu_mode = nullptr;
    }
    printf("done \n");

    checkCufft(cufftDestroy(fft_plan));
    checkCuda(cudaStreamDestroy(cuStream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Average unpack: " << avg_unpack / calls << endl;
    cout << "Average copyto: " << avg_copyto / calls << endl;
    cout << "Average rotate: " << avg_rotate / calls << endl;
    cout << "Average fft: " << avg_fft / calls << endl;
    cout << "Average fracrotate: " << avg_fracrotate / calls << endl;
    cout << "Average postprocess: " << avg_postprocess / calls << endl;
    cout << "Actual time processing (seconds): " << (double) processing_time / 1000. / 1000. / 3 << endl; 
    duration = duration_cast<microseconds>(stop - constructor_time);
    cout << "GPUMode lifetime: " << duration.count() / 1000. / 1000. << endl;
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

    //printf("In process_gpu \n");
    //std::cout << "cfg_numBufferedFFTs = " << cfg_numBufferedFFTs << " numBufferdFFTs = "<< numBufferedFFTs << std::endl;
    //exit(0);
    //printf("calccrosspolautocorrs = %d \n",calccrosspolautocorrs);
    auto begin_time = high_resolution_clock::now();
    calls += 1;
    //std::cout << "Doing the thing. fftloop: " << fftloop << ", numBufferedFFTs: " << numBufferedFFTs << ", numblocks: " << numblocks << ", startblock: " << startblock << std::endl;
    
    // Sanity checks
//    assert(numblocks == config->getNumBufferedFFTs(configindex));     // If this fails then check the input file and change "NUM BUFFERED FFTS"
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
    } else if (usecomplex) {
        NOT_SUPPORTED("complex data");
    }

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

    //std::cout << "Copy bytes to GPU " << datalengthbytes << std::endl;
    packeddata_gpu = new GpuMemHelper<char>((char*)data, datalengthbytes, cuStream);
    // Copy packed data to device
    packeddata_gpu->copyToDevice();
    //std::cout <<" Data copied to GPU" << std::endl;
    // Figure out how many frames in the packed data
    int framestounpack = datalengthbytes / config->getFrameBytes(configindex, datastreamindex);
    //std::cout << "Frames to unpack: " << framestounpack << std::endl;

    //if (datalengthbytes % config->getFrameBytes(configindex, datastreamindex) != 0) {
     // std::cout << "**************************************************" << std::endl;
     // std::cout << "datalengthbytes = " << datalengthbytes << std::endl;
     // std::cout << "frame bytes = " << config->getFrameBytes(configindex, datastreamindex) << std::endl;
     // std::cout << "modulo = " << datalengthbytes % config->getFrameBytes(configindex, datastreamindex) << std::endl; 
     // std::cout << "**************************************************" << std::endl;
    //}

    if (datalengthbytes > 1) {  // datalengthbytes <= 1 means an invalid sub int which should be handled....
        
        assert(datalengthbytes % config->getFrameBytes(configindex, datastreamindex) == 0);     // Buffer contains fraction of a frame :(. This shouldn't happen!
        
    } else { 
      //fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);
      //conj_fftd_gpu = new GpuMemHelper<cuFloatComplex>(fftchannels * cfg_numBufferedFFTs * numrecordedbands, cuStream, false);

      // set everything to zero and return

      //std::cout << "fftchannels * cfg_numBufferedFFTs * numrecordedbands = " << fftchannels * cfg_numBufferedFFTs * numrecordedbands << std::endl;    
      //std::cout << "setting fftd_gpu and conj_fftd_gpu to 0." << std::endl;
      //cudaMemset(fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex));
      //cudaMemset(conj_fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex));
      //std::cout << "done setting fftd_gpu and conj_fftd_gpu didn't segfault huzzah!" << std::endl;
        checkCuda(cudaMemsetAsync(fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex), cuStream));
	    checkCuda(cudaMemsetAsync(conj_fftd_gpu->gpuPtr(), 0.0, fftchannels * cfg_numBufferedFFTs * numrecordedbands * sizeof(cuFloatComplex), cuStream));
        checkCuda(cudaStreamSynchronize(cuStream));
	    fftd_gpu->copyToHost();
        conj_fftd_gpu->copyToHost();
	    delete packeddata_gpu;
        //std::cout << "Invalide subint returning" << std::endl;
	    return numBufferedFFTs;
    }
    //std::cout <<"not invalid subint" << std::endl;

    valid_frames = new GpuMemHelper<bool>(framestounpack, cuStream, false); 

    // Reset pcal accumulation only once at the start of a subintegration.
        if (!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0) &&
            (datasec != pcalResetDataSec || datans != pcalResetDataNs)) {
        checkCuda(cudaMemsetAsync(pcal_output_real->gpuPtr(), 0,
                                  sizeof(float) * numrecordedbands * pcal_bin_stride_length, cuStream));
            pcalResetDataSec = datasec;
            pcalResetDataNs = datans;
    }
    // Reset the autocorrelations
    //std::cout << "Reset autocorrelations" << std::endl;



    checkCuda(cudaMemsetAsync(temp_autocorrelations_gpu->gpuPtr(), 0,
                              sizeof(cf32) * numrecordedbands * recordedbandchannels * autocorrwidth, cuStream));

    // Update the interpolator
    gInterpolator->copyToDevice();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    avg_copyto += duration.count();

    start = high_resolution_clock::now();
    //std::cout << "calculatePre_cpu" << std::endl;
    calculatePre_cpu(fftloop, numBufferedFFTs, startblock, numblocks);

    int samplegranularity = 0;
    //std::cout << "unpack data" << std::endl;
    packeddata_gpu->sync();
    //printf("About to call unpack_all with framestounpack=%d\n", framestounpack);
    //fflush(stdout);
    unpack_all(framestounpack, samplegranularity);
    //std::cout << "unpack data done" << std::endl;

//    if (!printed_unpack_debug && numrecordedbands >= 2) {
//        std::cout << "Launching check_unpack kernel to print unpacked data for debugging" << std::endl;
//        check_unpack<<<1, 1, 0, cuStream>>>(unpackedarrays_gpu->gpuPtr(), numrecordedbands, unpacksamples);
//        checkCuda(cudaStreamSynchronize(cuStream));
//        printed_unpack_debug = true;
//    }

    //printf("samplegranularity = %d \n", samplegranularity);
    
    //if (unpackstartsamples != 0) {
    //    printf("unpackstartsamples = %d\n", unpackstartsamples);
    //} 
    

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_unpack += duration.count();
    //std::cout << "end unpack data" << std::endl;
   

    int counts[numrecordedfreqs] = {0};
    // Set up the FFT window indices and weights
    // Ideally this will move to the GPU but it's a bit tricky. Isn't *too* time intensive anyway I think
    //std::cout << "set_weights" << std::endl;
    
    // CRITICAL: nearestSamples->copyToDevice() in calculatePre_cpu() is async.
    // Sync before reading nearestSamples->ptr() to avoid reading stale data from previous iteration.
    nearestSamples->sync();
/// FOR DEBUGGING
    if (datans == 36500000 && datastreamindex == 4 && numBufferedFFTs > 0) {
        int min_nearest_sample = INT_MAX;
        int max_nearest_sample = INT_MIN;
        for (int ii = 0; ii < numBufferedFFTs; ii++) {
            const int ns = nearestSamples->ptr()[ii];
            if (ns >= 0) {
                if (ns < min_nearest_sample) {
                    min_nearest_sample = ns;
                }
                if (ns > max_nearest_sample) {
                    max_nearest_sample = ns;
                }
            }
        }

        if (min_nearest_sample == INT_MAX) {
            min_nearest_sample = -1;
        }
        if (max_nearest_sample == INT_MIN) {
            max_nearest_sample = -1;
        }

        const int frame_samples = config->getFrameSamples(configindex, datastreamindex);
        const int unpacked_size_batch = framestounpack * frame_samples;
        const int max_nearest_plus_fft = (max_nearest_sample >= 0) ? (max_nearest_sample + (int)fftchannels) : -1;

    //    printf("DEBUG_PCAL_GPU_SPAN call=%d ds=%d datasec=%d datans=%d numBufferedFFTs=%d min_nearest=%d max_nearest=%d max_nearest_plus_fft=%d unpacked_size=%d\n",
    //           calls,
    //           datastreamindex,
    //           datasec,
    //           datans,
    //           numBufferedFFTs,
    //           min_nearest_sample,
    //           max_nearest_sample,
    //           max_nearest_plus_fft,
    //           unpacked_size_batch);
    }
/// END DEBUGGING    
    for (int fftwin = 0; fftwin < numBufferedFFTs; fftwin++) {
        
        //std::cout << "before set_weights: fftloop = " << fftloop << ", startblock = " << startblock << std::endl;
        set_weights(fftwin, framestounpack, counts);

        //if (debug_dataweight_gpu_prints < 64) {
//        if (datans == 36500000) {
// //           if (datastreamindex == 4) { 
//                  printf("DEBUG_DATAWEIGHT_GPU call=%d ds=%d idx=%d nearestSamples=%d sampleIndex=%d weight=%.9f valid=%d offsetsec=%d datasec=%d datans=%d\n",
//                      calls,
//                      datastreamindex,
//                      fftwin,
//                      nearestSamples->ptr()[fftwin],
//                      gSampleIndexes->ptr()[fftwin],
//                      dataweight[fftwin],
//                      (int)(dataweight[fftwin] > 0.0f),
//                      offsetseconds,
//                      datasec,
//                      datans);
//                       debug_dataweight_gpu_prints++;
//  //          }       
//            
//        }
    }

    // Right after the set_weights loop (line 391)
//    printf("Host array immediately after set_weights:\n");
//    for (int i = 490; i < 510; i++) {
//        printf("gValidSamples[%d] = %d\n", i, gValidSamples->ptr()[i]);
//   }
//    for (int ii=0; ii<numrecordedfreqs; ii++) {
//        printf("counts[%d] = %d\n",ii,counts[ii]);
//    }
  


    start = high_resolution_clock::now();

    // Indices are now calculated, so we can copy them to the gpu
    indices->copyToDevice();

    // We need to copy the sample indexes to the gpu
    gSampleIndexes->copyToDevice();
    gValidSamples->copyToDevice();




    // todo: remove
   //std::cout << "Data copied and unpacked with code: " << cudaGetLastError() << std::endl;
    checkCuda(cudaStreamSynchronize(cuStream));


//    gValidSamples->copyToHost();  // Copy back from device to see what's actually on the GPU
//    for (int i = 490; i < std::min(510, (int)cfg_numBufferedFFTs); i++) {
//        printf("gValidSamples[%d] on GPU = %d\n", i, gValidSamples->ptr()[i]);
//    }
//    checkCuda(cudaStreamSynchronize(cuStream));
//    exit(0);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_copyto += duration.count();


    start = high_resolution_clock::now();

    if(!(config->getDPhaseCalIntervalMHz(configindex, datastreamindex) == 0)) { 

        if (datans == 36500000 && datastreamindex == 4 && numBufferedFFTs > 87) {
        //    printf("DEBUG_PCAL_GPU_HOST call=%d ds=%d idx=%d sampleIndex=%d nearest=%d datasamples=%d offsetsec=%d datasec=%d datans=%d\n",
        //           calls,
        //           datastreamindex,
        //           87,
        //           gSampleIndexes->ptr()[87],
        //           nearestSamples->ptr()[87],
        //           datasamples,
        //           offsetseconds,
        //           datasec,
        //           datans);

            const int unpack_sample_index_87 = gSampleIndexes->ptr()[87];
            const int unpack_nearest_87 = nearestSamples->ptr()[87];
            const int unpack_sample_offset_87 = datasamples + unpack_sample_index_87;
            const int unpacked_size_batch = framestounpack * config->getFrameSamples(configindex, datastreamindex);
            debug_print_unpack_window<<<1, 1, 0, cuStream>>>(
                unpackedarrays_gpu->gpuPtr(),
                unpack_sample_index_87,
                unpacked_size_batch,
                datastreamindex,
                87,
                87,
                0,
                unpack_nearest_87,
                0,
                unpack_sample_offset_87,
                datasamples);
            checkCuda(cudaGetLastError());
            checkCuda(cudaStreamSynchronize(cuStream));
        }

        pcalExtraction(fftloop, numBufferedFFTs, startblock, numblocks);
        checkCuda(cudaStreamSynchronize(cuStream));
        pcal_output_real->copyToHost();
        checkCuda(cudaStreamSynchronize(cuStream));
        // point pcal_output_real_gpu_mode
        pcal_output_real_gpu_mode = pcal_output_real->ptr();
        if (datans == 36500000 && datastreamindex == 4) {
            for (int band = 0; band < numrecordedbands && band < 2; band++) {
                const float *buf = pcal_output_real_gpu_mode + (band * pcal_bin_stride_length);
                const int n_bins = N_pcal_bins->ptr()[band];
                const int sample_index_87 = (numBufferedFFTs > 87) ? gSampleIndexes->ptr()[87] : -1;
                const int sample_offset_87 = datasamples + sample_index_87;
                int pcal_index_87 = -1;
                if (n_bins > 0 && sample_index_87 >= 0) {
                    pcal_index_87 = sample_offset_87 % n_bins;
                    if (pcal_index_87 < 0) {
                        pcal_index_87 += n_bins;
                    }
                }

                float sum_abs_first_half = 0.0f;
                float sum_abs_second_half = 0.0f;
                if (n_bins > 0 && pcal_bin_stride_length >= (2 * n_bins)) {
                    for (int ii = 0; ii < n_bins; ii++) {
                        sum_abs_first_half += std::fabs(buf[ii]);
                        sum_abs_second_half += std::fabs(buf[n_bins + ii]);
                    }
                }

                printf("DEBUG_PCAL_GPU_RAW_BINS ds=%d band=%d datasec=%d datans=%d bins=%d values=%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                       datastreamindex,
                       band,
                       datasec,
                       datans,
                       pcal_bin_stride_length,
                       buf[0],
                       buf[1],
                       buf[2],
                       buf[3],
                       buf[4],
                       buf[5],
                       buf[6],
                       buf[7]);

                  if (n_bins > 0 && pcal_index_87 >= 0) {
                      const int i0 = (pcal_index_87 + n_bins - 3) % n_bins;
                      const int i1 = (pcal_index_87 + n_bins - 2) % n_bins;
                      const int i2 = (pcal_index_87 + n_bins - 1) % n_bins;
                      const int i3 = pcal_index_87;
                      const int i4 = (pcal_index_87 + 1) % n_bins;
                      const int i5 = (pcal_index_87 + 2) % n_bins;
                      const int i6 = (pcal_index_87 + 3) % n_bins;
                      const int i7 = (pcal_index_87 + 4) % n_bins;

                      printf("DEBUG_PCAL_GPU_RAW_AROUND ds=%d band=%d sampleIndex=%d sampleOffset=%d n_bins=%d pcal_index=%d idx=%d,%d,%d,%d,%d,%d,%d,%d values=%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                          datastreamindex,
                          band,
                          sample_index_87,
                          sample_offset_87,
                          n_bins,
                          pcal_index_87,
                          i0,
                          i1,
                          i2,
                          i3,
                          i4,
                          i5,
                          i6,
                          i7,
                          buf[i0],
                          buf[i1],
                          buf[i2],
                          buf[i3],
                          buf[i4],
                          buf[i5],
                          buf[i6],
                          buf[i7]);

                      printf("DEBUG_PCAL_GPU_RAW_HALVES ds=%d band=%d n_bins=%d sumabs_first=%.9f sumabs_second=%.9f\n",
                          datastreamindex,
                          band,
                          n_bins,
                          sum_abs_first_half,
                          sum_abs_second_half);
                  }
            }
        }
        // DEBUG: summarize full pcal buffers instead of only first few bins.
        if (datasec > 0) {
            for (int band = 0; band < numrecordedbands && band < 2; band++) {
                const float *buf = pcal_output_real_gpu_mode + (band * pcal_bin_stride_length);
                int nonzero_count = 0;
                int first_nonzero = -1;
                float max_abs = 0.0f;
                for (int ii = 0; ii < pcal_bin_stride_length; ii++) {
                    float v = buf[ii];
                    float av = std::fabs(v);
                    if (av > 0.0f) {
                        nonzero_count++;
                        if (first_nonzero < 0) {
                            first_nonzero = ii;
                        }
                        if (av > max_abs) {
                            max_abs = av;
                        }
                    }
                }
                
                //printf("DEBUG gpumode pcal band=%d datasec=%d nonzero=%d first_nonzero=%d max_abs=%.6e\n",
                //       band, datasec, nonzero_count, first_nonzero, max_abs);
            }
        }
    }    



    // Run the fringe rotation
    //std::cout << "Starting processing" << std::endl;
    fringeRotation(fftloop, numBufferedFFTs, startblock, numblocks);

    checkCuda(cudaStreamSynchronize(cuStream));

    // Run PCAL extraction
    



    // todo: remove
    //std::cout << "Fringe rotation ended with code: " << cudaGetLastError() << std::endl;
    

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_rotate += duration.count();

    start = high_resolution_clock::now();
    // Actually run the FFT
    runFFT();

    // todo: remove
    //std::cout << "Data FFTed with code: " << cudaGetLastError() << std::endl;
    checkCuda(cudaStreamSynchronize(cuStream));

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_fft += duration.count();

    start = high_resolution_clock::now();

    // do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f) 
    fractionalRotation(fftloop, numBufferedFFTs, startblock, numblocks, calccrosspolautocorrs, counts);

    // todo: remove
    //std::cout << "Fractional rotate complete with code: " << cudaGetLastError() << std::endl;
    checkCuda(cudaStreamSynchronize(cuStream));
    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_fracrotate += duration.count();

    start = high_resolution_clock::now();

    // This synchronise is really needed, as we need the GPU processing/memcpys to finish before we read the result
    // data in to the autocorrelation vectors
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





//    fftd_gpu->copyToHost();
//    conj_fftd_gpu->copyToHost();
//
//    fftd_gpu->sync();
//    conj_fftd_gpu->sync();
//
//    for (int j = 0; j < numrecordedbands; j++) {
//        int numfftsprocessed = 0;
//        for (; numfftsprocessed < numBufferedFFTs; numfftsprocessed++) {
//            int i = fftloop * numBufferedFFTs + numfftsprocessed + startblock;
//            if (i >= startblock + numblocks)
//                break; // may not have to fully complete last fftloop
//
//            auto status = vectorAddProduct_cf32(
//                    reinterpret_cast<const cf32 *>(&fftd_gpu->ptr()[(j * cfg_numBufferedFFTs) + (i * fftchannels)]),
//                    reinterpret_cast<const cf32 *>(&conj_fftd_gpu->ptr()[(j * cfg_numBufferedFFTs) + (i * fftchannels)]),
//                    autocorrelations[0][j],
//                    recordedbandchannels
//                    );
//            if (status != vecNoErr)
//                csevere << startl << "Error in autocorrelation!!!" << status << endl;
//        }
//    }
//
//    for (int i = 0; i < autocorrwidth; i++) {
//        for (int j = 0; j < numrecordedbands; j++) {
//            for (int k = 0; k < recordedbandchannels; k++) {
//                autocorrelations[i][j][k].im = 0;
//                autocorrelations[i][j][k].re = 0;
//            }
//        }
//    }

//    static auto printed = false;
//    if (!printed) {
//        printed = true;
//
//        for (int i = 0; i < autocorrwidth; i++) {
//            for (int j = 0; j < numrecordedbands; j++) {
//                cout << i << " " << j << " - " << autocorrelations[i][j]->re << " : " << autocorrelations[i][j]->im << endl;
//            }
//        }
//    }

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_postprocess += duration.count();

    processing_time += duration_cast<microseconds>(stop - begin_time).count();

    delete packeddata_gpu;

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
        printf("Data weight for subloopindex %d is %f, which is invalid. Setting fft outputs to 0.\n", subloopindex, dataweight[subloopindex]);
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
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
    //    if (datans == 36500000) {
    //        printf("DEBUG_DATAWEIGHT_GPU_INVALID call=%d ds=%d idx=%d sub=%d datalen=%d offsetsec=%d invalid_subint=%d datasec=%d datans=%d validflag=%d flagword_idx=%d flagbit_idx=%d flagword=0x%08x nearest=%d reason_len=%d reason_subint=%d reason_flag=%d\n",
    //               calls,
    //               datastreamindex,
    //               index,
    //               subloopindex,
    //               datalengthbytes,
    //               offsetseconds,
    //               (int)(offsetseconds == INVALID_SUBINT),
    //               datasec,
    //               datans,
    //               validflagbit,
    //               validflagwordindex,
    //               validflagbitindex,
    //               validflagword,
    //               nearestSamples->ptr()[subloopindex],
    //               reason_datalen,
    //               reason_subint,
    //               reason_validflag);
    //        debug_dataweight_gpu_invalid_prints++;
    //    }
//        std::cerr << "to M::p_g; we are in the weird place with the datalengthbytes" << std::endl;
//        std::cerr << "to M::p_g; numrecorededbands = " << numrecordedbands << std::endl;
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
//        cerr << "Mode for DS " << datastreamindex << " is bailing out of index " << index << "/" << subloopindex << " which is scan " << currentscan << ", sec " << offsetseconds << ", ns " << offsetns << " because datalengthbytes is " << datalengthbytes << " and validflag was " << ((validflags[index/FLAGS_PER_INT] >> (index%FLAGS_PER_INT)) & 0x01) << endl;
        return false; //don't process crap data
    }

    // Check that the nearest sample is valid
    if (nearestSamples->ptr()[subloopindex] < -1 ||
        (((nearestSamples->ptr()[subloopindex] + fftchannels) / samplesperblock) * bytesperblocknumerator) / bytesperblockdenominator >
        datalengthbytes) {
        if (debug_dataweight_gpu_invalid_prints < 128 && datastreamindex == 4) {
            printf("DEBUG_DATAWEIGHT_GPU_RANGE_INVALID call=%d ds=%d idx=%d sub=%d nearest=%d fftchannels=%zu datalen=%d samplesperblock=%d\n",
                   calls,
                   datastreamindex,
                   index,
                   subloopindex,
                   nearestSamples->ptr()[subloopindex],
                   fftchannels,
                   datalengthbytes,
                   samplesperblock);
            debug_dataweight_gpu_invalid_prints++;
        }
//        std::cerr << "to M::p_g; we are in the 'crap data' branch" << std::endl;
//        cerror << startl << "MODE error for datastream " << datastreamindex
//               << " - trying to process data outside range - aborting!!! nearest sample was " << nearestSamples->ptr()[subloopindex]
//               << ", the max bytes should be " << datalengthbytes << " and hence last sample should be "
//               << (datalengthbytes * bytesperblockdenominator) / (bytesperblocknumerator * samplesperblock)
//               << " (fftchannels is " << fftchannels << "), offsetseconds was " << offsetseconds << ", offsetns was "
//               << offsetns << ", index was " << index << ", average delay was " << nearestSamples->ptr()[subloopindex] << ", datasec was "
//               << datasec << ", datans was " << datans << ", fftstartmicrosec was " << fftstartmicrosec << endl;
        for (int i = 0; i < numrecordedbands; i++) {
            status = vectorZero_cf32(fftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
            status = vectorZero_cf32(conjfftoutputs[i][subloopindex], recordedbandchannels);
            if (status != vecNoErr)
                csevere << startl << "Error trying to zero fftoutputs when data is bad!" << endl;
        }
        return false;
    }

    return true;
}

void GPUMode::process_unpack(int index, int subloopindex) {

    //std::cout << "In process_unpack" << std::endl;
    // Clear the perbandweights for this subloopindex
    if(perbandweights)
    {
        for(int b = 0; b < numrecordedbands; ++b)
        {
            perbandweights[subloopindex][b] = 0.0;
        }
    }

    if (!is_data_valid(index, subloopindex)) {
        
        // since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
        dataweight[subloopindex] = 0.0;

        gValidSamples->ptr()[subloopindex] = false;
        return;
    }

    gValidSamples->ptr()[subloopindex] = true;

    if (nearestSamples->ptr()[subloopindex] == -1) {
        nearestSamples->ptr()[subloopindex] = 0;
        dataweight[subloopindex] = unpack(nearestSamples->ptr()[subloopindex], subloopindex);
    } else if (nearestSamples->ptr()[subloopindex] < unpackstartsamples || nearestSamples->ptr()[subloopindex] > unpackstartsamples + unpacksamples - fftchannels)
        //need to unpack more data
        dataweight[subloopindex] = unpack(nearestSamples->ptr()[subloopindex], subloopindex);

    gSampleIndexes->ptr()[subloopindex] = nearestSamples->ptr()[subloopindex] - unpackstartsamples;

    if (!is_dataweight_valid(subloopindex)) {
        gValidSamples->ptr()[subloopindex] = false;
    } else {
        // Todo: This can definitely be cleaned up and improved
        for (int i = 0; i < numrecordedfreqs; i++) {
            int count = 0;
            // PWCR numrecordedbands = 2 for the test; but e.g. 8 is very realistical
            // Loop over all recorded bands looking for the matching frequency we should be dealing with
            for (int j = 0; j < numrecordedbands; j++) {
                // For upper sideband bands, normally just need to copy the fftd channels.
                // However for complex double upper sideband, the two halves of the frequency space are swapped, so they need to be swapped back

                if (config->matchingRecordedBand(configindex, datastreamindex, i, j)) {
		    //printf("setting indices! \n");
		    //exit(0);	    
                    indices->ptr()[(i * MAX_INDICIES) + count++] = j;

                    // At this point in the code the array fftoutputs[j] contains complex-valued voltage spectra with the following properties:
                    //
                    // 1. The zero element corresponds to the lowest sky frequency.  That is:
                    //    fftoutputs[j][0] = Local Oscillator Frequency              (for Upper Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Lower Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Complex Lower Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Upper Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Lower Sideband)
                    //
                    // 2. The frequency increases monotonically with index
                    //
                    // 3. The last element of the array corresponds to the highest sky frequency minus the spectral resolution.
                    //    (i.e., the first element beyond the array bound corresponds to the highest sky frequency)

                    //store the weight for the autocorrelations
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
}

void GPUMode::set_weights(int subloopindex, int nframes, int *counts) {

    // Not sure if this is still needed. Set to zero for now.
     unpackstartsamples = 0;
    // unpackstartsamples = nearestSamples->ptr()[subloopindex] - (nearestSamples->ptr()[subloopindex]% samplegranularity);
    //std::cout << "subloopindex = " << subloopindex << ", unpackstartsamples = " << unpackstartsamples <<  ", samplegranularity = " << samplegranularity << std::endl;
    // Clear the perbandweights for this subloopindex
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
        return;
    }
    //std::cout << "Data is valid for subloopindex " << subloopindex << std::endl;
    gValidSamples->ptr()[subloopindex] = true;

    if (nearestSamples->ptr()[subloopindex] == -1) {
        nearestSamples->ptr()[subloopindex] = 0;
        dataweight[subloopindex] = 1.0;
        cerr << "Why is this happening?" << std::endl;      // I'm not sure what case this branch is for
        abort();
    } else if (subloopindex + 1 == config->getNumBufferedFFTs(configindex)) {
        // We are in the last loop
        if (nearestSamples->ptr()[subloopindex] + fftchannels > nframes * config->getFrameSamples(configindex, datastreamindex)) {
            cerr << "This FFT window is trying to cross into unloaded data" << std::endl;
            abort();
        } else {
            int start_frame = nearestSamples->ptr()[subloopindex] / config->getFrameSamples(configindex, datastreamindex);
            dataweight[subloopindex] = (float)valid_frames->ptr()[start_frame];
        }
    } else if (nearestSamples->ptr()[subloopindex] < unpackstartsamples || nearestSamples->ptr()[subloopindex] > unpackstartsamples + unpacksamples - fftchannels) {
        //std::cout << "Entered standard path subloopindex = " << subloopindex << ", nearestSamples = " << nearestSamples->ptr()[subloopindex] << ", unpackstartsamples = " << unpackstartsamples << ", unpacksamples = " << unpacksamples << ", fftchannels = " << fftchannels << std::endl;
        // Standard path. TODO: above condition can be simplified I think
        int start_frame = nearestSamples->ptr()[subloopindex] / config->getFrameSamples(configindex, datastreamindex);
        int end_frame = (nearestSamples->ptr()[subloopindex + 1] - 1) / config->getFrameSamples(configindex, datastreamindex);
        if (start_frame == end_frame) {
            // This FFT window does not cross a frame boundary
            dataweight[subloopindex] = valid_frames->ptr()[start_frame] * 1.0;
        } else if (start_frame + 1 == end_frame) {
            // Crosses frame boundary: set weight proportional to occupancy in each frame
            float frac_first_frame = (float)(end_frame * config->getFrameSamples(configindex, datastreamindex) - nearestSamples->ptr()[subloopindex]) / (float)fftchannels;
            dataweight[subloopindex] = (frac_first_frame) * valid_frames->ptr()[start_frame] + (1 - frac_first_frame) * valid_frames->ptr()[end_frame];
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
                    // At this point in the code the array fftoutputs[j] contains complex-valued voltage spectra with the following properties:
                    //
                    // 1. The zero element corresponds to the lowest sky frequency.  That is:
                    //    fftoutputs[j][0] = Local Oscillator Frequency              (for Upper Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Lower Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth  (for Complex Lower Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Upper Sideband)
                    //    fftoutputs[j][0] = Local Oscillator Frequency - bandwidth/2(for Complex Double Lower Sideband)
                    //
                    // 2. The frequency increases monotonically with index
                    //
                    // 3. The last element of the array corresponds to the highest sky frequency minus the spectral resolution.
                    //    (i.e., the first element beyond the array bound corresponds to the highest sky frequency)

                    //store the weight for the autocorrelations
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
    sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
    dest[destIndex] = cuCmulf(c, cr);
  

    if (srcVal != 0) {
       //printf("lofreqs = %lf \n",lofreqs[0]);
       //printf("Using src[%lu][%lu] = %f to get dest[%lu] = %f + %fi  and lofreq = %lf\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal, destIndex, dest[destIndex].x, dest[destIndex].y, lofreqs[bandindex]);
       //printf("src[%lu][%lu] = %f\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal);

    }

}

// Copied from mathutil.cpp for gpu kernal implementation 
//__device__ void gpu_gcd(double a, double b, long* result)
//{
//    long a_long = 0;
//    long b_long = 0;
//    a_long = floor(a+0.5);
//    b_long = floor(b+0.5);
//
//    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
//   
//    //if (thread_index == 0) {
//      //  printf("a_long = %ld",a_long);
//       // printf("b_long = %ld",b_long);
//    //}
//    if (a_long == 0 || b_long == 0) {
//        result[0] = max(a_long,b_long);
//        return;
//    } else {
//        while (true) {
//            a_long = a_long%b_long;
//            if (a_long == 0) {
//               result[0] = b_long;
//               return;
//            }
//            b_long = b_long%a_long;
//            if (b_long == 0) {
//               result[0]= a_long;
//               return; 
//            }
//        }
//    } 
//}



// Adopted from the Implicit Shift pcal extraction method in pcal.cpp
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
    //printf("Entered gpu_pcalextraction kernel with fftloop = %d, startblock = %d, numblocks = %d, fftchannels = %lu \n", fftloop, startblock, numblocks, fftchannels);
    
    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the numrecordedbands index [0 .. numrecordedbands]
    // threadIdx.y in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the numrecordedbands size
    // blockDim.y in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    //int numrecordedbands = blockDim.x;
    int bandindex = threadIdx.x;
    //printf("fftchannels, N_pcal_bins[%d] = %ld, %ld \n", bandindex, fftchannels, N_pcal_bins[bandindex]); 

    //int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
 
    const size_t subloopindex = blockIdx.x;
    //printf("subloopindex = %lu \n", subloopindex);
    if (!validSamples[subloopindex]) {
        //printf("Invalid sample at subloopindex = %lu \n", subloopindex);
        // Not valid, so don't do anything
        return;
    }
    //printf("Valid subloopindex in pcal_extraction= %lu \n", subloopindex);

    // May not have to fully complete last fftloop; keep behaviour aligned with other kernels.
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        return;
    }
    
    
    const int sample_index = sampleIndexes[subloopindex];
    if (sample_index < 0) {
        return;
    }
    int sampleoffset = datasamples + sample_index;
   
   


    // Determine fftchannel number
    const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (channelindex >= fftchannels) {
        return;
    }

    // One thread per (subloopindex, bandindex) performs the pcal accumulation.
    // Additional channel threads would duplicate the same sums.
    if (channelindex != 0) {
        return;
    }

//    _fs_hz          = 2 * bandwidth_hz;
//    _pcalspacing_hz = pcal_spacing_hz;
//    _pcaloffset_hz  = pcal_offset_hz;
//    _N_bins         = (int)(_fs_hz / gcd(_fs_hz, _pcaloffset_hz));
//    _N_tones        = calcNumTones(bandwidth_hz, _pcaloffset_hz, _pcalspacing_hz);
//    _cfg = new pcal_config_pimpl();
//    _estimatedbytes = 0;
        

     // Adjust sample offset 
    size_t pcal_index = (sampleoffset)% N_pcal_bins[bandindex];
     
    // Pointer to pcal data within unpacked data 
    //printf("nearestSamples[subloopindex], unpackstartsamples = %d, %d \n",nearestSamples[subloopindex],unpackstartsamples);
    f32 *samples = &(unpackedarrays[bandindex][sample_index]);
     //printf("fftchannels, N_pcal_bins[%d], tail = %ld, %ld, %d \n", bandindex, fftchannels, N_pcal_bins[bandindex],tail);
    size_t tail = (fftchannels % N_pcal_bins[bandindex]);
    size_t end  = fftchannels - tail;   

    float const* src = samples;

    // Pointer to storage location for pcal output taking into account stride length 
    // of pcal_bin_stride_length = N_pcal_bins_max*2
    float* dst = (pcal_output_real + (bandindex * pcal_bin_stride_length)) + pcal_index;

//    if (subloopindex == 87 && bandindex == 0) {
//
//        printf("DEBUG_PCAL_GPU sub=%llu band=%d sample_index=%d sampleoffset=%d datasamples=%d pcal_index=%llu N_bins=%d fftchannels=%llu tail=%llu end=%llu src0=%.9f src1=%.9f\n",
//               (unsigned long long)subloopindex,
//               bandindex,
//               sample_index,
//               sampleoffset,
//               datasamples,
//               (unsigned long long)pcal_index,
//               N_pcal_bins[bandindex],
//               (unsigned long long)fftchannels,
//               (unsigned long long)tail,
//               (unsigned long long)end,
//               src[0],
//               src[1]);
//    }

    for (size_t ii=0; ii<end; ii+=N_pcal_bins[bandindex]) {
        // src[ii+cc] -> dst[cc]
        for (size_t cc=0; cc<N_pcal_bins[bandindex]; cc++) {
            //atomicAdd(&dst[cc], src[ii+cc]);
            atomicAdd(&dst[cc], src[ii+cc]);
        }
    }  
         
    // Handle the tail portion 
    if (0 != tail) {
        for (size_t cc=0; cc<tail; cc++) {
            atomicAdd(&dst[cc], src[end+cc]);
        }
    }
    //if (subloopindex != 499) {
        //printf("bandindex = %d, subloopindex = %lu, nearestsample = %d, unpackstartsamples = %d, \n", bandindex, subloopindex, nearestSamples[subloopindex], unpackstartsamples);
    //}

//    if (src[end+0] != 0.0) {
//       printf("src[end+0], src[end+1], src[end+3] = %f, %f, %f \ndst[0], dst[1], dst[3] = %f, %f, %f \n",src[end+0],src[end+1],src[end+2],dst[0],dst[1],dst[2]);
//    }

    // Below method may be invalid for GPU parallel processing


    /* This method is from Walter Brisken, it works perfectly for smallish 'len'
    * and when offset and tone spacing have suitable properties.
    * Instead of rotating the input to counteract the offset, we bin
    * into a long vector with size of the offset repeat length (again *2 to avoid
    * buffer wraps). After long-term integration, we copy desired FFT bins
    * into PCal. The time-domain PCal can be derived from inverse FFT.
    */
    // Copied from pcal.cpp, will need to implemented in CUDA
    /* Process the first part that fits perfectly */
//    float const* src = samples;
//     printf("samples[0] = %f,  src[0] =  %f\n",samples[0], src[0]);
//    float* dst = &(pcal_output_real[pcal_index]);
//     printf("end = %d \n", end);

 
//    for (size_t n = 0; n < end; n+=N_pcal_bins[bandindex], src+=N_pcal_bins[bandindex]) {
       //vectorAddf32_I(samples, pcal_output_real[i][pcal_index], N_pcal_bins[i]);
//        for (size_t pcal_bin_num = 0; pcal_bin_num <  N_pcal_bins[bandindex]; pcal_bin_num++) {
//            atomicAdd(&dst[pcal_bin_num],src[pcal_bin_num]);
//        }
//    }
     /* Handle any samples that didn't fit */
//    if (tail != 0) {    
//       for (size_t pcal_bin_num = 0; pcal_bin_num <  tail; pcal_bin_num++) {
//            atomicAdd(&dst[pcal_index],src[pcal_bin_num]);
//       }  
//       pcal_index = (pcal_index + tail) % N_pcal_bins[bandindex];
//    }


    /* Done! */
    //_samplecount += len;  
      

   
}


void GPUMode::pcalExtraction(int fftloop, int numBufferedFFTs, int startblock, int numblocks) { 


    //std::cout << "In pcalExtraction" << std::endl;
    // Kernel accumulates across all fftchannels internally, so channel threading is unnecessary.
    size_t fftchannels_block = 1;
    size_t fftchannels_grid = 1;

    double bandwidth_hz = 1e6*recordedbandwidth;
    double pcal_spacing_hz = 1e6*config->getDPhaseCalIntervalMHz(configindex, datastreamindex);

//    Moving the below allocations to the mode constructor  

//    pcal_offsets_hz = new GpuMemHelper<int>(numrecordedbands, cuStream);
//    N_pcal_bins = new GpuMemHelper<int>(numrecordedbands, cuStream);
//    double bandwidth_hz = 1e6*recordedbandwidth;
//    double fs_hz = 2 * bandwidth_hz;
//    double pcal_spacing_hz = 1e6*config->getDPhaseCalIntervalMHz(configindex, datastreamindex);
//    int N_pcal_bins_max=0;    
// 
//    for (int ii=0; ii<numrecordedbands; ii++) { 
//        int localfreqindex = config->getDLocalRecordedFreqIndex(configindex, datastreamindex, ii);
//        pcal_offsets_hz->ptr()[ii] = config->getDRecordedFreqPCalOffsetsHz(configindex, datastreamindex, localfreqindex);
//        N_pcal_bins->ptr()[ii] = (int)(fs_hz/gcd(fs_hz,pcal_offsets_hz->ptr()[ii]));
//        if (N_pcal_bins->ptr()[ii] > N_pcal_bins_max) {
//            N_pcal_bins_max = N_pcal_bins->ptr()[ii]; 
//        }  
//    }
//    pcal_bin_stride_length = N_pcal_bins_max*2;  // *2 to avoid buffer wraps in the long-term integration method
//    pcal_offsets_hz->copyToDevice();
//    N_pcal_bins->copyToDevice();
//    pcal_output_real = new GpuMemHelper<float>(numrecordedbands*pcal_bin_stride_length, cuStream);     
//    for (size_t ii=0; ii<numrecordedbands*pcal_bin_stride_length; ii++) {
//        pcal_output_real->ptr()[ii] = 0.0;
//    }
//    pcal_output_real->copyToDevice();




    gpu_pcalextraction<<<
        dim3(numBufferedFFTs, fftchannels_grid),
        dim3(numrecordedbands,fftchannels_block),
        0, cuStream
    >>>             
            (
                    complexunpacked_gpu->gpuPtr(),
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

    gpu_fringeRotation<<<
        dim3(numBufferedFFTs, fftchannels_grid),
        dim3(numrecordedbands,fftchannels_block),
        0, cuStream
    >>>
            (
                    complexunpacked_gpu->gpuPtr(),
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

// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
__device__ void atomicAddFloatComplex(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    auto *x = (float*) a;
    auto *y = x + 1;
    //use atomicAdd for float variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}

__global__ void gpu_resultsrotatorMultiply(
        cuFloatComplex* const fftoutputs,
        cuFloatComplex* const conjfftoutputs,
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
        sincosf(TWO_PI * exponent, &cr.y, &cr.x);
        fftoutputs[dataIndex] = cuCmulf(fftoutputs[dataIndex], cr);

        // do the conjugation
        conjfftoutputs[dataIndex] = cuConjf(fftoutputs[dataIndex]);

        // do the autocorrelation (skipping Nyquist channel)
        // Calculate the destination index
        const size_t autocorrIndex = (bandindex * recordedbandchannels) + channelindex;
        atomicAddFloatComplex(&autocorrelations[autocorrIndex], cuCmulf(fftoutputs[dataIndex], conjfftoutputs[dataIndex]));
    }

    for (size_t recordedfreq = 0; recordedfreq < numrecordedfreqs; recordedfreq++) {
        if (calccrosspolautocorrs && counts_gpu[recordedfreq] > 1) {
            //printf("calccrosspolautocorrs = %d\n",calccrosspolautocorrs);		
            // if we need to, do the cross-polar autocorrelations
            size_t fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;
            //long indices_index = (recordedfreq * MAX_INDICIES) + 1;
            //printf("indices_index = %lu \n", indices_index); 
	    //int index_conj = indices[(recordedfreq * MAX_INDICIES) + 1];
            //printf("index1 = %d\n", indices[0]);
	    //printf("index2 = %d\n", indices[1]);    
	    //printf("index_conj = %d \n", index_conj);
	    //printf("recordedfreq = %lu \n",recordedfreq);
	    //printf("MAX_INDICIES = %lu \n",MAX_INDICIES);  
	    size_t conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;
            //printf("fftIndex = %lu \n",fftIndex);
	  
	    //printf("conjIndex = %lu \n ",conjIndex);
        //    long autocorr_index = (numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex; 
			     
	    //printf("autocorr_index = %lu \n ",autocorr_index);

            //long long size_autocorrs = sizeof(&autocorrelations)/sizeof(&autocorrelations[0]);
            //long long size_fftoutputs = sizeof(fftoutputs) /sizeof(fftoutputs[0]);
	    //long long size_conjfftoutputs = sizeof(conjfftoutputs) /sizeof(conjfftoutputs[0]);

	    //long long size_autocorrs = sizeof(&autocorrelations);
         //   long long size_fftoutputs = sizeof(fftoutputs);
         //   long long size_conjfftoutputs = sizeof(conjfftoutputs);



      
            //printf("size_autocorrs = %lld \n",size_autocorrs);
            //printf("size_fftoutputs = %lld \n",size_fftoutputs);
            //printf("size_conjfftoutputs = %lld \n",size_conjfftoutputs); 
            //cuFloatComplex * blah1 = &autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex];
	    //cuFloatComplex blah2 = fftoutputs[fftIndex];    
            ////cuFloatComplex blah3 = conjfftoutputs[conjIndex];


            //int size1 = sizeof(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex]);
            //int size2 = sizeof(cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));
            //int input_size1 = sizeof(cuFloatComplex*);
            //int input_size2 = sizeof(cuFloatComplex);




	    
            //printf("autocorr_index = %ld \n",autocorr_index);
        //    cuFloatComplex * xx = &autocorrelations[autocorr_index];
	    //cuFloatComplex ffto = fftoutputs[fftIndex];
	    //printf("fftIndex = %lu \n",fftIndex);
	    //printf("conjIndex = %lu \n",conjIndex);
	    //cuFloatComplex cffto = conjfftoutputs[conjIndex];
	    //cuFloatComplex yy = cuCmulf(ffto, cffto);
        //    printf("yy = %f  %f \n",yy.x,yy.y);
	    //printf("&autocorrelations[autocorr_index][0] = %f	    
		//}
	    //}	
            

	        atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));
	    
            fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;
            conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;

            atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 1] * recordedbandchannels) + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));
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
        GpuMemHelper<int> *counts_gpu;
        counts_gpu = new GpuMemHelper<int>(numrecordedfreqs, cuStream); 
        for (int ii=0; ii < numrecordedfreqs; ii++) {
       	    counts_gpu->ptr()[ii] = counts[ii];
	}
	counts_gpu->copyToDevice();
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

    // Start copying the autocorrelations back to the host
    temp_autocorrelations_gpu->copyToHost();
}

void GPUMode::runFFT() {
    checkCufft(cufftExecC2C(fft_plan, complexunpacked_gpu->gpuPtr(), fftd_gpu->gpuPtr(), CUFFT_FORWARD));
}
