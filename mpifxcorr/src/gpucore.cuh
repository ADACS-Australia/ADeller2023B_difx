#ifndef GPUCORE_H
#define GPUCORE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core.h"

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
     * @brief Device pointers to the FFT output buffers for Datastream 1.
     * In the CPU case, this is accessed dynamically via `modes[ds1index]->getFreqs()`. 
     * Here, we copy an array of device pointers so the GPU kernel can read the 
     * raw VRAM buffers directly without querying the host.
     */
    const cuFloatComplex** d_m1_ptrs;

    /**
     * @brief Device pointers to the conjugated FFT output buffers for Datastream 2.
     * In the CPU case, accessed via `modes[ds2index]->getConjugatedFreqs()`.
     */
    const cuFloatComplex** d_m2_ptrs;

    /**
     * @brief Flattened array mapping a (baseline, polarisation) to a specific band in Datastream 1.
     * Replaces the CPU's `config->getBDataStream1BandIndex(...)` lookups. 
     * Pre-calculating this on the host prevents the GPU from needing to understand 
     * the complex DiFX Configuration tree.
     */
    int* d_stream1BandIndexes;

    /**
     * @brief Flattened array mapping a (baseline, polarisation) to a specific band in Datastream 2.
     * Replaces the CPU's `config->getBDataStream2BandIndex(...)` lookups.
     */
    int* d_stream2BandIndexes;

    /**
     * @brief Pre-calculated starting offsets for each baseline in the final results array.
     * Replaces `config->getCoreResultBaselineOffset(...)`. This is the secret to 
     * bypassing `threadcrosscorrs`; it allows GPU threads to calculate their 
     * final global memory address directly.
     */
    int* d_coreResultBaselineOffsets;

    int cudaMaxThreadsPerBlock;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
