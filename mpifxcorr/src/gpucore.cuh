#ifndef GPUCORE_H
#define GPUCORE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include "core.h"

class Mode;

/**
 * @class GPUCore
 * @brief GPU-accelerated subclass of Core for cross-multiplication and averaging.
 *
 * In the legacy CPU implementation, data is heavily partitioned into `xmacstridelen` 
 * chunks to keep working data inside the CPU's L1/L2 caches. The CPU calculates 
 * intermediate cross-multiplies, stores them in `scratchspace->threadcrosscorrs`, 
 * and later un-jumbles them via frequency averaging into `procslots[index].results`.
 * * GPUCore completely bypasses this cache-thrashing architecture. It fuses the XMAC 
 * (cross-multiply accumulate) and frequency averaging steps into a single GPU kernel, 
 * writing contiguous, perfectly formatted visibilities directly to a device-side 
 * results buffer, mirroring the final `procslots` layout natively.
 */
class GPUCore : public Core {
public:
    GPUCore(const int id, Configuration *const conf, int *const dids, MPI_Comm rcomm);
    ~GPUCore();

    virtual void loopprocess(int threadid);

    /**
     * Issue all of subint `index`'s GPU work: station processing, the fused XMAC
     * into results_gpu[index], and an asynchronous device->host copy of just the
     * visibility prefix onto d2hStream (recording d2hDone[index]). The host-side
     * trailing sections (baseline weights, autocorrelations, pcal) are added into
     * the pre-zeroed procslots[index].results here too, since they occupy a region
     * disjoint from the visibility copy. Does NOT wait for the visibility copy, so
     * it can overlap the next subint's compute. The slot lock is retained until
     * completegpudata(index) runs.
     */
    void issuegpudata(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                      threadscratchspace *scratchspace);

    /**
     * Complete subint `index`: wait for its visibility device->host copy
     * (d2hDone[index]) to land, after which procslots[index].results is fully valid
     * and the slot may be released to the manager.
     */
    void completegpudata(int index);

protected:
    virtual Mode *getMode(const int configindex, const int datastreamindex) {
        return config->getMode(configindex, datastreamindex, true);
    }

private:
    // -------------------------------------------------------------------------
    // GPU Memory Pointers & Streams
    // -------------------------------------------------------------------------
    /// Compute stream: all station processing (GPUMode) and the XMAC enqueue here,
    /// in order, so the XMAC naturally follows the FFT output it consumes.
    cudaStream_t cuStream;
    /// Device-to-host stream: the visibility transfer back to the host runs here so
    /// it can overlap the NEXT subintegration's compute on cuStream.
    cudaStream_t d2hStream;

    /** * @brief The final, device-side visibility buffers - one per procslot.
     * Replaces the CPU's `scratchspace->threadcrosscorrs`. Each is mapped exactly
     * to the CPU's `procslots[index].results` layout. There is one buffer per
     * procslot (indexed by the procslots ring index) so that the deferred,
     * overlapped device->host copy of subint N reads a stable buffer while subint
     * N+1's XMAC writes its own. Size each: `maxcoreresultlength * sizeof(cuFloatComplex)`.
     */
    std::vector<cuFloatComplex*> results_gpu;

    /// Per-procslot PINNED host staging buffers for the visibility transfer. The
    /// device->host copy must land in page-locked memory to be truly asynchronous
    /// (cudaMemcpyAsync to pageable memory is effectively synchronous and would
    /// defeat the overlap); procslots[].results is pageable, so we stage into
    /// these and completegpudata() copies the landed prefix across on the host.
    std::vector<cuFloatComplex*> results_host;

    /// Per-procslot event marking completion of that slot's visibility device->host
    /// copy on d2hStream. completegpudata() waits on it before the slot is released
    /// for the manager send.
    std::vector<cudaEvent_t> d2hDone;

    /// Per-procslot event marking completion of that slot's input host->device
    /// copies on cuStream (recorded once in issuegpudata, after the fftloop
    /// loop). completegpudata() waits on it before the slot is released,
    /// so the manager cannot start refilling procslots[].databuffer[] while an
    /// async copy from it (pinned-input path) is still in flight. This makes the
    /// input-reuse invariant explicit; on the staging fallback path the wait is
    /// trivially satisfied.
    std::vector<cudaEvent_t> h2dInputDone;

    /// When true (default; disable with DIFX_GPU_PIPELINE=0), subint N's visibility
    /// transfer is left in flight while subint N+1 is issued, and only awaited just
    /// before slot N is released - overlapping the transfer with N+1's compute. When
    /// false, each subint is completed immediately after it is issued (no overlap),
    /// reproducing the pre-pipelining behaviour for A/B comparison.
    bool pipeline;

    /**
     * @brief Device pointers to the FFT output buffers for Datastream 1 (one per baseline).
     * In the CPU case, this is accessed dynamically via `modes[ds1index]->getFreqs()`.
     * Here, we copy an array of device pointers so the GPU kernel can read the
     * raw VRAM buffers directly without querying the host. These pointers are
     * frequency-independent (they depend only on the baseline's datastream) and,
     * because a GPUMode's fftd buffers are allocated once and never move, they are
     * invariant for a given configuration - so they are populated once by
     * buildXmacPlans() rather than every subintegration.
     *
     * NOTE: when the per-slot GPU buffer pool ([DEPTH] double/triple buffering)
     * lands, these cached pointers will need to be rebuilt per slot.
     */
    const cuFloatComplex** d_m1_ptrs;

    /**
     * @brief Device pointers to the conjugated FFT output buffers for Datastream 2 (one per baseline).
     * In the CPU case, accessed via `modes[ds2index]->getConjugatedFreqs()`. See d_m1_ptrs.
     */
    const cuFloatComplex** d_m2_ptrs;

    /**
     * @brief Device pointers to the per-FFT validity flags of each baseline's
     * two datastreams (one entry per baseline, like d_m1_ptrs/d_m2_ptrs).
     * Invalid FFT windows hold stale spectra in the modes' FFT buffers, so the
     * fused XMAC kernel skips any FFT flagged invalid on either datastream
     * (the CPU path zeroes such spectra, making their contribution zero).
     */
    const bool** d_v1_ptrs;
    const bool** d_v2_ptrs;

    /**
     * @brief Per-configuration baseline-weight reduction plan (Increment 2).
     *
     * Replaces the per-window host loop that used to sum dataweight1[w] *
     * dataweight2[w] over the subint's FFT windows for every (freq, baseline,
     * polproduct) in host_accumulate. The accumulators are enumerated in the
     * EXACT order the host finalize fold consumes them (used freq -> baseline
     * with localfreqindex>=0 -> polproduct), so the reduced values D2H'd into
     * h_bweightResults map one-to-one onto the fold's walk. d_bwDw1/d_bwDw2 hold,
     * per accumulator, the two datastreams' device per-window dataweight arrays
     * (GPUMode::getGpuDataWeights). Built once per configuration in
     * buildXmacPlans, freed in freeXmacPlans. Only used on the device-weights
     * path; the DIFX_GPU_WEIGHTS_HOST fallback keeps the host loop.
     *
     * bwDestOffset carries each accumulator's destination: the float index into
     * procslots[].floatresults where its reduced weight is added. Recording it
     * at plan-build time makes the flat fold self-describing - it does not
     * re-walk (and cannot silently diverge from) the config enumeration.
     */
    const float** d_bwDw1 = nullptr;    ///< [bweightNumAccum] ds1 dataweight ptrs
    const float** d_bwDw2 = nullptr;    ///< [bweightNumAccum] ds2 dataweight ptrs
    float* d_bweightResults = nullptr;  ///< [bweightNumAccum] per-subint device sums
    float* h_bweightResults = nullptr;  ///< [bweightNumAccum] pinned host mirror
    std::vector<int> bwDestOffset;      ///< [bweightNumAccum] floatresults index per accumulator
    int bweightNumAccum = 0;

    /**
     * @brief Cached, per-frequency launch metadata for the fused XMAC kernel.
     *
     * All of the DiFX Configuration lookups (band indexes, baseline result
     * offsets, channel counts, pol-product counts) that feed a fused-kernel
     * launch are invariant for a given configindex. Rather than re-walk the
     * config tree and re-upload these small arrays on every subintegration, we
     * compute them once per configuration in buildXmacPlans() and reuse them.
     * The device arrays below are persistent (allocated in buildXmacPlans,
     * freed in freeXmacPlans / on config change).
     */
    struct XmacFreqPlan {
        int numPolarisationProducts;
        int num_averaged_channels;
        int channelstoaverage;
        int* d_stream1BandIndexes;         ///< [numbaselines * numPolarisationProducts]
        int* d_stream2BandIndexes;         ///< [numbaselines * numPolarisationProducts]
        int* d_coreResultBaselineOffsets;  ///< [numbaselines]
        /* Per-baseline, per-stream FFT buffer strides. A GPUMode's fftd buffer
         * is laid out [window][band][fftchannels], where fftchannels =
         * freqchannels * (1 for complex sampling, 2 for real) and the band
         * count are properties of THAT datastream - so on a mixed baseline
         * (e.g. real x complex) the two streams' strides differ and each side
         * must be indexed with its own. */
        int* d_stream1BandStride;          ///< [numbaselines] = that stream's fftchannels
        int* d_stream1WindowStride;        ///< [numbaselines] = fftchannels * numrecordedbands
        int* d_stream2BandStride;          ///< [numbaselines]
        int* d_stream2WindowStride;        ///< [numbaselines]
    };
    std::vector<XmacFreqPlan> xmacPlans;

    /// The configindex the cached xmacPlans were built for (-1 = none yet).
    int xmacPlanConfigIndex = -1;

    /// (Re)build the cached per-frequency XMAC launch metadata for a configuration.
    void buildXmacPlans(int configindex, Mode **modes);
    /// Free the device arrays held by the cached XMAC plans.
    void freeXmacPlans();

    /**
     * Start-up VRAM budget check. Estimates the peak device memory this Core
     * will need (all datastreams' GPUModes for the most demanding config, plus
     * this Core's own results/pointer buffers) via GPUMode::estimateDeviceBytes
     * and compares it against what the device reports free, applying a safety
     * margin. cfatal + MPI_Abort if it will not fit, so an oversized job fails
     * in the first second with a clear required-vs-available message instead of
     * a mid-run allocation failure. Must run before any GPUMode is constructed.
     */
    void checkDeviceMemory();

    int cudaMaxThreadsPerBlock;
    /// SM count of the device, used to size the fused-XMAC launch grid.
    int cudaMultiProcessorCount;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
