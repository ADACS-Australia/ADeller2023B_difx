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

    /**
     * Install the compute stream that all subsequently constructed GPUModes
     * will enqueue their work on. GPUCore calls this with its own stream
     * before any Mode is constructed, so station processing, the XMAC and
     * the transfers form a single in-order queue - the explicit ordering
     * that the procslots pipelining work builds on. If no stream has been
     * installed a GPUMode creates (and owns) a private one.
     */
    static void setSharedComputeStream(cudaStream_t stream) { sharedComputeStream = stream; }

    /**
     * True (default) when per-window weights/validity/sample indexes are
     * computed on the device by the gpu_set_weights kernel, eliminating the
     * valid_frames D2H round-trip and the per-datastream stream drains.
     * DIFX_GPU_WEIGHTS_HOST=1 restores the host set_weights path (which also
     * carries the full-fidelity WDEBUG output).
     */
    static bool useGpuWeights();

    /**
     * Land the device-computed per-window weights on the host and run the
     * (interim, see docs/gpu-deserialization-design.md) host accumulations
     * that consume them: Mode::dataweight[] for the baseline-weight loops
     * and the per-band autocorrelation weights. Must be called after the
     * end-of-subint compute-stream drain (GPUCore does, before
     * host_accumulate). No-op when this subint's weights were computed on
     * the host (fallback path or invalid subint).
     */
    void finishWeights();

    /**
     * Declare whether the Core receive buffers (procslots[].databuffer[]) that
     * setData() hands to process_gpu are page-locked (cudaHostRegister'd).
     * GPUCore calls this once, before any Mode is constructed, after
     * registering (or failing to register) those buffers. When true,
     * process_gpu issues its input H2D copy directly from the delivered
     * buffer; when false it falls back to staging through packeddata_gpu's
     * pinned host half.
     */
    static void setInputBuffersPinned(bool pinned) { inputBuffersPinned = pinned; }

    /**
     * Estimate the total device memory (bytes) a GPUMode for this
     * config/datastream will allocate, using only Configuration lookups -
     * callable before any Mode exists. Mirrors the constructor's device
     * allocations plus the cuFFT plan work area; used by GPUCore's start-up
     * VRAM budget check so an oversized job fails immediately with a clear
     * message instead of partway through buffer allocation.
     */
    static size_t estimateDeviceBytes(Configuration *config, int configindex, int dsindex);

    int process_gpu(int fftloop, int numBufferedFFTs, int startblock,
                    int numblocks) override;  //frac sample error is in microseconds

//    int GPUMode::set_invalid_data(int fftloop, int numBufferedFFTs, int startblock,
//                         int numblocks);

    void process_unpack(int index, int subloopindex);
    void set_weights(int subloopindex, int nframes, int *counts, int numBufferedFFTs);
    virtual void unpack_all(int) {}
    void runFFT();
    void fringeRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void pcalExtraction(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void calculatePre_cpu(int fftloop, int numBufferedFFTs, int startblock, int numblocks);
    void fractionalRotation(int fftloop, int numBufferedFFTs, int startblock, int numblocks, bool calccrosspolautocorrs, int *counts);

    [[nodiscard]] const cuFloatComplex* getGpuFreqs() const override { return fftd_gpu->gpuPtr(); }
    [[nodiscard]] const cuFloatComplex* getGpuConjugatedFreqs() const override { return conj_fftd_gpu->gpuPtr(); }
    [[nodiscard]] const cf32* getGpuFreqsHost(int outputband, int subloopindex) const override {
        return (const cf32*) &fftd_gpu->ptr()[(subloopindex * fftchannels * numrecordedbands) + (outputband * fftchannels)];
    }
    [[nodiscard]] const cf32* getGpuConjugatedFreqsHost(int outputband, int subloopindex) const override {
        return (const cf32*) &conj_fftd_gpu->ptr()[(subloopindex * fftchannels * numrecordedbands) + (outputband * fftchannels)];
    }
    /// Device-side per-FFT validity flags. Invalid FFT windows have stale
    /// (never zeroed) spectra in fftd_gpu/conj_fftd_gpu - the mode kernels
    /// skip them - so consumers (the XMAC kernel) must exclude them; the CPU
    /// path instead zeroes such spectra so they contribute nothing.
    [[nodiscard]] const bool* getGpuValidSamples() const { return gValidSamples->gpuPtr(); }

    GpuMemHelper<cuFloatComplex> *fftd_gpu;
    GpuMemHelper<cuFloatComplex> *conj_fftd_gpu;

protected:
    int cudaMaxThreadsPerBlock;
    int cfg_numBufferedFFTs;
    GpuMemHelper<float*> *unpackedarrays_gpu;
    GpuMemHelper<float> *unpackeddata_gpu;
    GpuMemHelper<cuFloatComplex*> *complex_unpackedarrays_gpu;
    GpuMemHelper<cuFloatComplex> *complex_unpackeddata_gpu;
    GpuMemHelper<cuFloatComplex> *complex_fringe_rotated_gpu;
    GpuMemHelper<cuFloatComplex> *temp_autocorrelations_gpu;
    GpuMemHelper<char> *packeddata_gpu;
    GpuMemHelper<bool> *valid_frames;

    size_t estimatedbytes_gpu;

    // Remember how long the 'unpackedarrays' are -- norally this would be
    // 'unpacksamples' but e.g. the Mk5Mode implementation overwrites that
    size_t unpackedarrays_elem_count;

    GpuMemHelper<int> *gSampleIndexes;
    GpuMemHelper<bool> *gValidSamples;
    /// This subint's validity bit-words (host-born, uploaded per subint for
    /// the gpu_set_weights kernel; FLAGS_PER_INT bits per word).
    GpuMemHelper<unsigned int> *gValidFlags;
    /// Per-window data weights, computed on the device by gpu_set_weights
    /// and copied back asynchronously for the interim host consumers.
    GpuMemHelper<float> *gDataWeights;
    /// Per-freq matching-band count for the indices band map - pure
    /// configuration, built once at construction (see also indices).
    int *countsStatic;
    /// True while the current subint's weights live only on the device
    /// (i.e. gpu_set_weights ran and finishWeights() has not yet run).
    bool weightsOnDevice;
    GpuMemHelper<double> *gInterpolator;
    GpuMemHelper<float> *gFracSampleError;
    GpuMemHelper<double> *gLoFreqs;
    GpuMemHelper<unsigned int> *indices;
    GpuMemHelper<double>* grecordedfreqclockoffsets;
    GpuMemHelper<double>* grecordedfreqclockoffsetsdelta;
    GpuMemHelper<double>* grecordedfreqlooffsets;
    GpuMemHelper<int>* pcal_offsets_hz;
    GpuMemHelper<float> *pcal_output_real = nullptr;  // temporary unassembled output for the pcaloffsethz==0.0f case
    GpuMemHelper<cuFloatComplex> *pcal_output_complex = nullptr;  // temporary unassembled output for the pcaloffsethz!=0.0f case
    GpuMemHelper<int>* N_pcal_bins;
    GpuMemHelper<int>* counts_gpu;
    cudaStream_t cuStream;
    /// The stream installed by setSharedComputeStream (GPUCore's), or nullptr.
    static cudaStream_t sharedComputeStream;
    /// True when GPUCore has page-locked the procslots receive buffers, so
    /// process_gpu can DMA directly from them (see setInputBuffersPinned).
    static bool inputBuffersPinned;
    /// True when cuStream is a private stream this mode must destroy.
    bool ownsStream;

    // precalc
    GpuMemHelper<int> *nearestSamples;

    //GpuMemHelper<int> *counts_gpu;
private:
    cufftHandle fft_plan;
    int pcalResetDataSec = INVALID_SUBINT;
    int pcalResetDataNs = 0;

    bool is_dataweight_valid(int subloopindex);
    bool is_data_valid(int index, int subloopindex);

    std::chrono::time_point<std::chrono::system_clock, std::chrono::system_clock::duration> constructor_time;
    // Per-instance CUDA timing events
    cudaEvent_t ev_start = nullptr, ev_copy1 = nullptr, ev_unpack = nullptr;
    cudaEvent_t ev_copy2 = nullptr, ev_pcal  = nullptr, ev_rotate = nullptr;
    cudaEvent_t ev_fft   = nullptr, ev_frac  = nullptr;

    // Per-instance timing accumulators (microseconds)
    long long t_copyto      = 0;
    long long t_unpack      = 0;
    long long t_pcal        = 0;
    long long t_rotate      = 0;
    long long t_fft         = 0;
    long long t_fracrotate  = 0;
    long long t_postprocess = 0;
    long long t_total       = 0;
    int calls               = 0;


};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
