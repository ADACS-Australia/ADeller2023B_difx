#include "gpucore.cuh"
#include "gpumode.cuh"
#include "alert.h"
#include "gpumode_kernels.cuh"
#include <thread>
//#include <iostream>
#include <chrono>

using namespace std::chrono;

GPUCore::GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm)
        : Core(id, conf, dids, rcomm) {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    cudaMaxThreadsPerBlock = prop.maxThreadsPerBlock;
}

static unsigned long long avg_preprocess;
static unsigned long long avg_postprocess;
int core_calls;
GPUCore::~GPUCore() {
    cout << "Average core pre: " << avg_preprocess / core_calls << endl;
    cout << "Average core post: " << avg_postprocess / core_calls << endl;
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
    for (int j = 0; j < numdatastreams; j++)
        delete modes[j];
    delete[] modes;
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
    vectorFree(scratchspace->threadcrosscorrs);
    vectorFree(scratchspace->chanfreqs);
    vectorFree(scratchspace->rotator);
    vectorFree(scratchspace->rotated);
    vectorFree(scratchspace->channelsums);
    vectorFree(scratchspace->argument);
    if (scratchspace->starecordbuffer != 0) {
        free(scratchspace->starecordbuffer);
    }
    delete scratchspace;

    cinfo << startl << "PROCESS " << mpiid << "/" << threadid << " process thread exiting!!!" << endl;

    extern int calls;
    cout << "process calls: " << calls << endl;
}

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

void GPUCore::processBaselineBased(
        cuFloatComplex** gpuM1Freqs,
        cuFloatComplex** gpuM2Freqs,
        char* stream1BandIndexes_gpu,
        char* stream2BandIndexes_gpu,
        cuFloatComplex* threadcrosscorrs_gpu,
        int xmacstridelength,
        int numPolarisationProducts,
        int numBufferedFFTs,
        int xmacpasses,
        int fftloop,
        int startblock,
        int numblocks,
        int fftchannels,
        int numrecordedbands,
        cudaStream_t cuStream
) {

    size_t fftchannels_block;
    size_t fftchannels_grid;

    size_t divisor = cudaMaxThreadsPerBlock / numPolarisationProducts;
    if (xmacstridelength > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = xmacstridelength / divisor;

        if (xmacstridelength % divisor != 0) {
            fftchannels_grid++;
        }
    } else {
        fftchannels_block = xmacstridelength;
        fftchannels_grid = 1;
    }

    _gpu_processBaselineBased<<<dim3(numBufferedFFTs, fftchannels_grid), dim3(numPolarisationProducts,
                                                                              fftchannels_block), 0, cuStream>>>
            (
                    gpuM1Freqs,
                    gpuM2Freqs,
                    threadcrosscorrs_gpu,
                    stream1BandIndexes_gpu,
                    stream2BandIndexes_gpu,
                    xmacpasses,
                    numbaselines,
                    xmacstridelength,
                    fftloop,
                    startblock,
                    numblocks,
                    fftchannels,
                    numrecordedbands
            );
}

void
GPUCore::processgpudata(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                        threadscratchspace *scratchspace) {
    ++core_calls;

    auto start = high_resolution_clock::now();

    std::cout << "called GPUCore::processgpudata for the " << core_calls << " time, index: " << index << ", startblock: "
              << startblock << ", numblocks: " << numblocks << std::endl;

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

    numBufferedFFTs = config->getNumBufferedFFTs(procslots[index].configindex);

    //set up the mode objects that will do the station-based processing
    for (int j = 0; j < numdatastreams; j++) {
        //zero the autocorrelations and set delays
        modes[j]->zeroAutocorrelations();
        modes[j]->setValidFlags(&(procslots[index].controlbuffer[j][3]));
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
    numBufferedFFTs = numblocks;
    numfftloops = numblocks / numBufferedFFTs;
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

//    cudaStream_t cuStream;
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
    for (int fftloop = 0; fftloop < numfftloops; fftloop++) {
        start = high_resolution_clock::now();

        numfftsprocessed = 0;   // not strictly needed, but to prevent compiler warning

        // do the station-based processing for this batch of FFT chunks.
        vector<std::thread> streamThreads;

        for (int j = 0; j < numdatastreams; j++) {
            streamThreads.emplace_back([&numfftsprocessed, &modes, j, fftloop, numBufferedFFTs, startblock, numblocks] {
                numfftsprocessed = ((GPUMode *) modes[j])->process_gpu(fftloop, numBufferedFFTs, startblock, numblocks);
            });
        }

        for (auto &t: streamThreads) {
            t.join();
        }

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "total processing: " << duration.count() << endl;
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


        for (int j = 0; j < numdatastreams; j++) {
            ((GPUMode *) modes[j])->fftd_gpu->copyToHost();
            ((GPUMode *) modes[j])->conj_fftd_gpu->copyToHost();
            ((GPUMode *) modes[j])->fftd_gpu->sync();
            ((GPUMode *) modes[j])->conj_fftd_gpu->sync();
        }

        //do the baseline-based processing for this batch of FFT chunks
        auto resultindex = 0;
        for(int f=0;f<config->getFreqTableLength();f++) {
            //All baseline freq indices into the freq table are determined by the *first* datastream
            //in the event of correlating USB with LSB data.  Hence all Nyquist offsets/channels etc
            //are determined by the freq corresponding to the *first* datastream
            auto freqchannels = config->getFNumChannels(f);
            auto xmacpasses = config->getNumXmacStrides(procslots[index].configindex, f);
            for (int x = 0; x < xmacpasses; x++) {
                auto xmacstart = x * xmacstridelength;
                int xmacstrideremain = std::min(freqchannels - xmacstart, xmacstridelength);
                //if(xmacstrideremain <= 0)  // note: since getNumXmacStrides() is a rounded-up value, there may be {xmaclen x '0'} to be concatenated per baseline to keep consistent with Configuration-assigned offsets
                //  break;
                if (xmacstrideremain < 0)
                    continue;

                //do the cross multiplication - gets messy for the pulsar binning
                for (int j = 0; j < numbaselines; j++) {
                    //get the localfreqindex for this frequency
                    localfreqindex = config->getBLocalFreqIndex(procslots[index].configindex, j, f);
                    if (localfreqindex >= 0) {
                        //get the two modes that contribute to this baseline
                        auto ds1index = config->getBOrderedDataStream1Index(procslots[index].configindex, j);
                        auto ds2index = config->getBOrderedDataStream2Index(procslots[index].configindex, j);
                        auto m1 = modes[ds1index];
                        auto m2 = modes[ds2index];

                        //do the baseline-based processing for this batch of FFT chunks
                        for (int fftsubloop = 0; fftsubloop < numBufferedFFTs; fftsubloop++) {
                            auto i = fftloop * numBufferedFFTs + fftsubloop + startblock;
                            if (i >= startblock + numblocks)
                                break; //may not have to fully complete last fftloop

                            //add the desired results into the resultsbuffer, for each polarisation pair [and pulsar bin]
                            //loop through each polarisation for this frequency
                            for (int p = 0;
                                 p < config->getBNumPolProducts(procslots[index].configindex, j, localfreqindex); p++) {
                                //get the appropriate arrays to multiply
                                auto vis1 = &(m1->getGpuFreqsHost(
                                        config->getBDataStream1BandIndex(procslots[index].configindex, j, localfreqindex,
                                                                         p), fftsubloop)[xmacstart]);
                                auto vis2 = &(m2->getGpuConjugatedFreqsHost(
                                        config->getBDataStream2BandIndex(procslots[index].configindex, j, localfreqindex,
                                                                         p), fftsubloop)[xmacstart]);


                                //not pulsar binning, so this is nice and simple - just cross multiply accumulate
                                //baselineweight gets updated later
                                status = vectorAddProduct_cf32(vis1, vis2,
                                                               &(scratchspace->threadcrosscorrs[resultindex +
                                                                                                p * xmacstridelength]),
                                                               xmacstrideremain);
                                if (status != vecNoErr)
                                    csevere << startl << "Error trying to xmac baseline " << j << " frequency "
                                            << localfreqindex << " polarisation product " << p << ", status " << status
                                            << endl;
                            }
                        }//for(fftsubloops)
                        //advance output index to concatenate the xmac slice of the next baseline-polproducts
                        //notes: - when xmacstrideremain is shorter than xmacstridelength we will simply end up with a few padded zeroes in output array, no change in output array spacing,
                        //       - there may be *more* channels (zero pads) than 'numchannels' of the freq; it holds that "freqtable[i].numchannels <= completestridelength[freq]*xmaclen"
                        //       - zero padding is essentially required due Configuration::populateResultLengths() precomputed offsets into xmac output array
                        if (procslots[index].pulsarbin && !procslots[index].scrunchoutput)
                            resultindex += config->getBNumPolProducts(procslots[index].configindex, j, localfreqindex) *
                                           procslots[index].numpulsarbins * xmacstridelength;
                        else
                            resultindex += config->getBNumPolProducts(procslots[index].configindex, j, localfreqindex) *
                                           xmacstridelength;
                    }
                }
            }
        }

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "baseline based processing: " << duration.count() << endl;

        start = high_resolution_clock::now();

        xcblockcount += numfftsprocessed;
        if (xcblockcount == maxxcblocks) {
            //shift/average and then lock results and copy data
            uvshiftAndAverage(index, threadid,
                              (startblock + xcshiftcount * maxxcblocks + ((double) maxxcblocks) / 2.0) * blockns,
                              maxxcblocks * blockns, currentpolyco, scratchspace);
            //reset the xcblockcount, increment xcshiftcount
            xcblockcount = 0;
            xcshiftcount++;
        }
        acblockcount += numfftsprocessed;
        if (acblockcount == maxacblocks) {
            //shift/average and then lock results and copy data
            averageAndSendAutocorrs(index, threadid,
                                    (startblock + acshiftcount * maxacblocks + ((double) maxacblocks) / 2.0) * blockns,
                                    maxacblocks * blockns, modes, scratchspace);
            //reset the acblockcount, increment acshiftcount, zero the autocorrelations
            acblockcount = 0;
            acshiftcount++;
            for (int j = 0; j < numdatastreams; j++)
                modes[j]->zeroAutocorrelations();
        }

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
        cout << "baseline weight: " << duration.count() << endl;
    }

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
//    checkCuda(cudaStreamDestroy(cuStream));

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
}
// vim: shiftwidth=2:softtabstop=2:expandtab
