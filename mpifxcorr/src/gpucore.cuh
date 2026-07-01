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

    void processgpudata(int index, int threadid, int startblock, int numblocks, Mode **modes, Polyco *currentpolyco,
                        threadscratchspace *scratchspace);

protected:
    virtual Mode *getMode(const int configindex, const int datastreamindex) {
        return config->getMode(configindex, datastreamindex, true);
    }

private:
    // -------------------------------------------------------------------------
    // GPU Memory Pointers & Streams
    // -------------------------------------------------------------------------
    cudaStream_t cuStream;
    
    /** * @brief The final, device-side visibility buffer.
     * Replaces the CPU's `scratchspace->threadcrosscorrs`. This array is mapped 
     * exactly to the CPU's `procslots[index].results` layout, allowing us to do 
     * a single, massive PCIe transfer at the very end of processing.
     * Size: `maxcoreresultlength * sizeof(cuFloatComplex)`.
     */
    cuFloatComplex* results_gpu;

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
        int fftchannels;          ///< freqchannels * sampling_multiplier
        int numrecordedbands;
        int* d_stream1BandIndexes;         ///< [numbaselines * numPolarisationProducts]
        int* d_stream2BandIndexes;         ///< [numbaselines * numPolarisationProducts]
        int* d_coreResultBaselineOffsets;  ///< [numbaselines]
    };
    std::vector<XmacFreqPlan> xmacPlans;

    /// The configindex the cached xmacPlans were built for (-1 = none yet).
    int xmacPlanConfigIndex = -1;

    /// (Re)build the cached per-frequency XMAC launch metadata for a configuration.
    void buildXmacPlans(int configindex, Mode **modes);
    /// Free the device arrays held by the cached XMAC plans.
    void freeXmacPlans();

    int cudaMaxThreadsPerBlock;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
