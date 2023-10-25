#ifndef GPUDECODE_H
#define GPUDECODE_H

int mk5_decode_general(struct mark5_stream *ms, int nsamp, float **data);
__global__ void gpu_unpack(struct mark5_stream *ms, const void *packed, float **unpacked, int nframes, int *goodsamples);
#endif