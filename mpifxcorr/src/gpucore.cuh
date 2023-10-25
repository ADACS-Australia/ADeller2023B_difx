#ifndef GPUCORE_H
#define GPUCORE_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "core.h"

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
  void processBaselineBased(
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
  );

  int cudaMaxThreadsPerBlock;
};

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
