#ifndef GPUMODE_H
#define GPUMODE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#include "mode.h"
#include "gpumode_kernels.cuh"
#include <mutex>
#include <chrono>

class Configuration;

class GPUMode : public Mode {
public:
    GPUMode(Configuration *conf, int confindex, int dsindex, int recordedbandchan, int chanstoavg, int bpersend,
            int gsamples, int nrecordedfreqs, double recordedbw, double *recordedfreqclkoffs,
            double *recordedfreqclkoffsdelta, double *recordedfreqphaseoffs, double *recordedfreqlooffs,
            int nrecordedbands, int nzoombands, int nbits, Configuration::datasampling sampling,
            Configuration::complextype tcomplex, int unpacksamp, bool fbank, bool linear2circular, int fringerotorder,
            int arraystridelen, bool cacorrs, double bclock);

    ~GPUMode() override;

    int process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                    int numblocks) override;  //frac sample error is in microseconds

    void process_unpack(int index, int subloopindex);
    void set_weights(int subloopindex, int nframes);
    virtual void unpack_all(int) {}
    void runFFT();
    void fringeRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void calculatePre_cpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void fractionalRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks);

    [[nodiscard]] const cuFloatComplex* getGpuFreqs() const override { return fftd_gpu->gpuPtr(); }
    [[nodiscard]] const cuFloatComplex* getGpuConjugatedFreqs() const override { return conj_fftd_gpu->gpuPtr(); }
    [[nodiscard]] const cf32* getGpuFreqsHost(int outputband, int subloopindex) const override {
        return (const cf32*) &fftd_gpu->ptr()[(subloopindex * fftchannels * numrecordedbands) + (outputband * fftchannels)];
    }
    [[nodiscard]] const cf32* getGpuConjugatedFreqsHost(int outputband, int subloopindex) const override {
        return (const cf32*) &conj_fftd_gpu->ptr()[(subloopindex * fftchannels * numrecordedbands) + (outputband * fftchannels)];
    }

    GpuMemHelper<cuFloatComplex> *fftd_gpu;
    GpuMemHelper<cuFloatComplex> *conj_fftd_gpu;

protected:
    int cudaMaxThreadsPerBlock;
    GpuMemHelper<float*> *unpackedarrays_gpu;
    GpuMemHelper<float> *unpackeddata_gpu;
    GpuMemHelper<cuFloatComplex> *complexunpacked_gpu;
    GpuMemHelper<cuFloatComplex> *temp_autocorrelations_gpu;
    GpuMemHelper<char> *packeddata_gpu;
    GpuMemHelper<bool> *valid_frames;

    size_t estimatedbytes_gpu;

    // Remember how long the 'unpackedarrays' are -- norally this would be
    // 'unpacksamples' but e.g. the Mk5Mode implementation overwrites that
    size_t unpackedarrays_elem_count;

    GpuMemHelper<int> *gSampleIndexes;
    GpuMemHelper<bool> *gValidSamples;
    GpuMemHelper<double> *gInterpolator;
    GpuMemHelper<float> *gFracSampleError;
    GpuMemHelper<double> *gLoFreqs;
    GpuMemHelper<unsigned int> *indices;
    GpuMemHelper<double>* grecordedfreqclockoffsets;
    GpuMemHelper<double>* grecordedfreqclockoffsetsdelta;
    GpuMemHelper<double>* grecordedfreqlooffsets;
    // Maps a band index to a frequency index
    // Set and transferred to the GPU during construction
    GpuMemHelper<size_t> *band2freq;

    cudaStream_t cuStream;

    // precalc
    int* nearestSamples;
private:

    cufftHandle fft_plan;
    int cfg_numBufferedFFTs;

    bool is_dataweight_valid(int subloopindex);
    bool is_data_valid(int index, int subloopindex);

    std::chrono::time_point<std::chrono::system_clock, std::chrono::system_clock::duration> constructor_time;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
