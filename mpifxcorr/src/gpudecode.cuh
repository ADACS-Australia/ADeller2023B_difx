#ifndef GPUDECODE_H
#define GPUDECODE_H

#include <cuComplex.h>


//int mk5_decode_general(struct mark5_stream *ms, int nsamp, float **data);
//__global__ void gpu_unpack(struct mark5_stream *ms, const void *packed, float **unpacked, int nframes, bool *goodframes);
//#endif

int mk5_decode_general(struct mark5_stream ms, int nsamp, float **data);
__global__ void gpu_unpack(struct mark5_stream ms, const void *packed, float **unpacked, int nframes, bool *goodframes);
__global__ void gpu_unpack_complex(struct mark5_stream ms, const void *packed, cuFloatComplex **unpacked, int nframes, bool *goodframes);
#endif