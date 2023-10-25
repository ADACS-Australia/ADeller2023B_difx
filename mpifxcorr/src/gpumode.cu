#include "gpumode.cuh"
#include "alert.h"
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>
#include <cufftXt.h>

#include <chrono>
#include <omp.h>
#include <thread>

using namespace std::chrono;

const int MAX_INDICIES = 10;

__global__ void gpu_allocate_unpacked(float** arrays, float* data, int nchan, int dlen) {
    // Use arrays to make data into a flattened 2D array
    for (int i = 0; i < nchan; i++) {
        arrays[i] = data + i * dlen;
        printf("Channel %i starts at %p\n", i, arrays[i]);
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

    auto start = high_resolution_clock::now();

    size_t buffer_payload_bytes = (config->getMaxDataBytes() / config->getFrameBytes(confindex, dsindex)) * config->getFramePayloadBytes(confindex, dsindex);
    size_t unpacked_size = buffer_payload_bytes * 8 / (config->getDNumBits(confindex, dsindex) * config->getDNumRecordedBands(confindex, dsindex));
    // What's the largest number of FFTs we can fit?
    cfg_numBufferedFFTs = (unpacked_size + fftchannels - 1) / fftchannels;
    std::cout << "Working on " << cfg_numBufferedFFTs << " FFTs" << std::endl;
    // cfg_numBufferedFFTs = config->getNumBufferedFFTs(confindex);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties( &prop, 0));

    checkCuda(cudaStreamCreate(&cuStream));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;

    std::cout << "fftchannels " << fftchannels << std::endl;

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
    std::cout << "Unpacked data size: " << numrecordedbands * unpacked_size << std::endl;

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

    gLoFreqs = new GpuMemHelper<double>(numrecordedfreqs, cuStream);

    indices = new GpuMemHelper<unsigned int>(MAX_INDICIES * numrecordedfreqs, cuStream);
    for (auto i = 0; i < MAX_INDICIES * numrecordedfreqs; i++) {
        indices->ptr()[i] = 0xffffffff;
    }
    grecordedfreqclockoffsets = new GpuMemHelper<double>(numrecordedfreqs, cuStream);
    grecordedfreqclockoffsetsdelta = new GpuMemHelper<double>(numrecordedfreqs, cuStream);
    grecordedfreqlooffsets = new GpuMemHelper<double>(numrecordedfreqs, cuStream);
    // Copy the lofreq and freq clock offset values to the GPU
    for (auto i = 0; i < numrecordedfreqs; i++) {
        gLoFreqs->ptr()[i] = config->getDRecordedFreq(configindex, datastreamindex, i);
        grecordedfreqclockoffsets->ptr()[i] = recordedfreqclockoffsets[i];
        grecordedfreqclockoffsetsdelta->ptr()[i] = recordedfreqclockoffsetsdelta[i];
        grecordedfreqlooffsets->ptr()[i] = recordedfreqlooffsets[i];
    }

    gLoFreqs->copyToDevice();
    grecordedfreqclockoffsets->copyToDevice();
    grecordedfreqclockoffsetsdelta->copyToDevice();
    grecordedfreqlooffsets->copyToDevice();

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
    nearestSamples = new int[cfg_numBufferedFFTs];

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "GPUMode(): " << duration.count() << endl;

    constructor_time = high_resolution_clock::now();
}

static unsigned long long avg_unpack;
static unsigned long long avg_copyto;
static unsigned long long avg_rotate;
static unsigned long long avg_fft;
static unsigned long long avg_fracrotate;
static unsigned long long avg_postprocess;
static unsigned long long processing_time;

int calls = 0;

GPUMode::~GPUMode() {
    auto start = high_resolution_clock::now();

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

    delete[] nearestSamples;

    checkCufft(cufftDestroy(fft_plan));
    checkCuda(cudaStreamDestroy(cuStream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "~GPUMode(): " << duration.count() << endl;

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
    std::cout << "Doing the thing. fftloop: " << fftloop << ", numBufferedFFTs: " << numBufferedFFTs << ", numblocks: " << numblocks << ", startblock: " << startblock << std::endl;

    // Sanity checks
    assert(numblocks == config->getNumBufferedFFTs(configindex));     // If this fails then check the input file and change "NUM BUFFERED FFTS"
    if (config->getDPhaseCalIntervalMHz(configindex, datastreamindex) != 0) {
        NOT_SUPPORTED("DPhaseCal");
    }

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

    std::cout << "Copied bytes to GPU " << datalengthbytes << std::endl;
    packeddata_gpu = new GpuMemHelper<char>((char*)data, datalengthbytes, cuStream);
    // Copy packed data to device
    packeddata_gpu->copyToDevice();

    // Figure out how many frames in the packed data
    int framestounpack = datalengthbytes / config->getFrameBytes(configindex, datastreamindex);
    assert(datalengthbytes % config->getFrameBytes(configindex, datastreamindex) == 0);     // Buffer contains fraction of a frame :(. This shouldn't happen!

    valid_frames = new GpuMemHelper<bool>(framestounpack, cuStream, false); 

    // Reset the autocorrelations
    checkCuda(cudaMemsetAsync(temp_autocorrelations_gpu->gpuPtr(), 0,
                              sizeof(cf32) * numrecordedbands * recordedbandchannels * autocorrwidth, cuStream));

    // Update the interpolator
    gInterpolator->copyToDevice();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    avg_copyto += duration.count();

    start = high_resolution_clock::now();

    calculatePre_cpu(fftloop, numBufferedFFTs, startblock, numblocks);

    packeddata_gpu->sync();
    unpack_all(framestounpack);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_unpack += duration.count();

    // Set up the FFT window indices and weights
    // Ideally this will move to the GPU but it's a bit tricky. Isn't *too* time intensive anyway I think
    for (int fftwin = 0; fftwin < numBufferedFFTs; fftwin++) {
        set_weights(fftwin, framestounpack);
    }

    start = high_resolution_clock::now();

    // Indices are now calculated, so we can copy them to the gpu
    indices->copyToDevice();

    // We need to copy the sample indexes to the gpu
    gSampleIndexes->copyToDevice();
    gValidSamples->copyToDevice();

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));
    std::cout << "Data copied and unpacked with code: " << cudaGetLastError() << std::endl;

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_copyto += duration.count();

    start = high_resolution_clock::now();

    // Run the fringe rotation
    std::cout << "Starting processing" << std::endl;
    fringeRotation(fftloop, numBufferedFFTs, startblock, numblocks);

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));
    std::cout << "Fringe rotation ended with code: " << cudaGetLastError() << std::endl;

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_rotate += duration.count();

    start = high_resolution_clock::now();
    // Actually run the FFT
    runFFT();

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));
    std::cout << "Data FFTed with code: " << cudaGetLastError() << std::endl;

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    avg_fft += duration.count();

    start = high_resolution_clock::now();

    // do the frac sample correct (+ phase shifting if applicable, + fringe rotate if its post-f)
    fractionalRotation(fftloop, numBufferedFFTs, startblock, numblocks);

    // todo: remove
    checkCuda(cudaStreamSynchronize(cuStream));
    std::cout << "Fractional rotate complete with code: " << cudaGetLastError() << std::endl;

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

bool GPUMode::is_dataweight_valid(int subloopindex) {
    int status;

    if (dataweight[subloopindex] <= 0.0) {
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

    // Check the data is valid for this index
    if ((datalengthbytes <= 1) || (offsetseconds == INVALID_SUBINT) ||
        (((validflags[index / FLAGS_PER_INT] >> (index % FLAGS_PER_INT)) & 0x01) == 0)) {
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
    if (nearestSamples[subloopindex] < -1 ||
        (((nearestSamples[subloopindex] + fftchannels) / samplesperblock) * bytesperblocknumerator) / bytesperblockdenominator >
        datalengthbytes) {
//        std::cerr << "to M::p_g; we are in the 'crap data' branch" << std::endl;
//        cerror << startl << "MODE error for datastream " << datastreamindex
//               << " - trying to process data outside range - aborting!!! nearest sample was " << nearestSamples[subloopindex]
//               << ", the max bytes should be " << datalengthbytes << " and hence last sample should be "
//               << (datalengthbytes * bytesperblockdenominator) / (bytesperblocknumerator * samplesperblock)
//               << " (fftchannels is " << fftchannels << "), offsetseconds was " << offsetseconds << ", offsetns was "
//               << offsetns << ", index was " << index << ", average delay was " << nearestSamples[subloopindex] << ", datasec was "
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

    if (nearestSamples[subloopindex] == -1) {
        nearestSamples[subloopindex] = 0;
        dataweight[subloopindex] = unpack(nearestSamples[subloopindex], subloopindex);
    } else if (nearestSamples[subloopindex] < unpackstartsamples || nearestSamples[subloopindex] > unpackstartsamples + unpacksamples - fftchannels)
        //need to unpack more data
        dataweight[subloopindex] = unpack(nearestSamples[subloopindex], subloopindex);

    gSampleIndexes->ptr()[subloopindex] = nearestSamples[subloopindex] - unpackstartsamples;

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

void GPUMode::set_weights(int subloopindex, int nframes) {
    // Not sure if this is still needed. Set to zero for now.
    unpackstartsamples = 0;

    // Clear the perbandweights for this subloopindex
    if(perbandweights)
    {
        for(int b = 0; b < numrecordedbands; ++b)
        {
            perbandweights[subloopindex][b] = 0.0;
        }
    }

    if (!is_data_valid(subloopindex, subloopindex)) {
        // since these data weights can be retreived after this processing ends, reset them to a default of zero in case they don't get updated
        dataweight[subloopindex] = 0.0;

        gValidSamples->ptr()[subloopindex] = false;
        return;
    }

    gValidSamples->ptr()[subloopindex] = true;

    if (nearestSamples[subloopindex] == -1) {
        nearestSamples[subloopindex] = 0;
        dataweight[subloopindex] = 1.0;
        cerr << "Why is this happening?" << std::endl;      // I'm not sure what case this branch is for
        abort();
    } else if (subloopindex + 1 == config->getNumBufferedFFTs(configindex)) {
        // We are in the last loop
        if (nearestSamples[subloopindex] + fftchannels > nframes * config->getFrameSamples(configindex, datastreamindex)) {
            cerr << "This FFT window is trying to cross into unloaded data" << std::endl;
            abort();
        } else {
            int start_frame = nearestSamples[subloopindex] / config->getFrameSamples(configindex, datastreamindex);
            dataweight[subloopindex] = (float)valid_frames->ptr()[start_frame];
        }
    } else if (nearestSamples[subloopindex] < unpackstartsamples || nearestSamples[subloopindex] > unpackstartsamples + unpacksamples - fftchannels) {
        // Standard path. TODO: above condition can be simplified I think
        int start_frame = nearestSamples[subloopindex] / config->getFrameSamples(configindex, datastreamindex);
        int end_frame = (nearestSamples[subloopindex + 1] - 1) / config->getFrameSamples(configindex, datastreamindex);
        if (start_frame == end_frame) {
            // This FFT window does not cross a frame boundary
            dataweight[subloopindex] = valid_frames->ptr()[start_frame] * 1.0;
        } else if (start_frame + 1 == end_frame) {
            // Crosses frame boundary: set weight proportional to occupancy in each frame
            float frac_first_frame = (float)(end_frame * config->getFrameSamples(configindex, datastreamindex) - nearestSamples[subloopindex]) / (float)fftchannels;
            dataweight[subloopindex] = (frac_first_frame) * valid_frames->ptr()[start_frame] + (1 - frac_first_frame) * valid_frames->ptr()[end_frame];
        } else {
            cerr << "FFT window somehow spans more than two frames. This is suspicious to me but maybe allowed?" << std::endl;
            abort();
        };
    }

    gSampleIndexes->ptr()[subloopindex] = nearestSamples[subloopindex] - unpackstartsamples;

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

void GPUMode::calculatePre_cpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
    int startIndex = fftloop * numBufferedFFTs + startblock;
    int endIndex = startblock + numblocks;

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
        nearestSamples[subloopindex] = int(starttime / sampletime + 0.5);

        double nearestsampletime = nearestSamples[subloopindex] * sampletime;
        gFracSampleError->ptr()[subloopindex] = float(starttime - nearestsampletime);
    }

    // Start copying the fracSampleErrors to the gpu
    gFracSampleError->copyToDevice();
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

    double bigAval = a * lofreqs[0] / (double) fftchannels - sampletime * 1.e-6 * recordedfreqlooffsets[0];
    double bigBval = b * lofreqs[0];

    // Calculate
    double bigB_reduced = bigBval - int(bigBval);
    double exponent = (bigAval * (double) channelindex + bigB_reduced);
    exponent -= int(exponent);
    cuFloatComplex cr;
    sincosf(-TWO_PI * exponent, &cr.y, &cr.x);
    cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
    dest[destIndex] = cuCmulf(c, cr);
    if (subloopindex == 5000) {
        //printf("Using src[%lu][%lu] = %f to get dest[%lu] = %f + %fi\n", srcIndex, sampleIndexes[subloopindex] + channelindex, srcVal, destIndex, dest[destIndex].x, dest[destIndex].y);
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
        size_t numrecordedfreqs
    ) {
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
        double bigAval = fracSampleError[subloopindex] - recordedfreqclockoffsets[0] + recordedfreqclockoffsetsdelta[0] / 2;

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
        if (numrecordedbands > 1) {
            // if we need to, do the cross-polar autocorrelations
            size_t fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;
            size_t conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;

            atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 0] * recordedbandchannels) + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));

            fftIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 1] * fftchannels) + channelindex;
            conjIndex = (subloopindex * fftchannels * numrecordedbands) + (indices[(recordedfreq * MAX_INDICIES) + 0] * fftchannels) + channelindex;

            atomicAddFloatComplex(&autocorrelations[(numrecordedbands * recordedbandchannels) + (indices[(recordedfreq * MAX_INDICIES) + 1] * recordedbandchannels) + channelindex], cuCmulf(fftoutputs[fftIndex], conjfftoutputs[conjIndex]));
        }
    }
}

void GPUMode::fractionalRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks) {
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
                    numrecordedfreqs
            );

    // Start copying the autocorrelations back to the host
    temp_autocorrelations_gpu->copyToHost();
}

void GPUMode::runFFT() {
    checkCufft(cufftExecC2C(fft_plan, complexunpacked_gpu->gpuPtr(), fftd_gpu->gpuPtr(), CUFFT_FORWARD));
}
