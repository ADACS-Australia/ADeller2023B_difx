#include <mark5access.h>
#include <mark5access/mark5_stream.h>
#include "gpudecode.cuh"
#include <iostream>
#include <bitset>
#include <stdio.h>

#define MARK5_FILL_WORD64 0x1122334411223344ULL

/* the high mag value for 2-bit reconstruction */
static const float HiMag = OPTIMAL_2BIT_HIGH;
static const float FourBit1sigma = 2.95;

// Stack all quantization levels (nbit <= 2) float values in one array and use nbit to offset accordingly
__constant__ float lutall[6] = {-1.0, 1.0, -HiMag, -1.0, 1.0, HiMag};

__device__ __forceinline__ float bitsread_gpu(char byte, int pos, int nbit) {
	// std::cout << "Byte : " << std::bitset<8>(byte) << std::endl;
	// std::cout << "Pos :  " << pos << "\tnbit :  " << nbit << std::endl;
	// std::cout << "Index: " << ((byte >> pos) & ((2 << (nbit - 1)) - 1)) + (nbit << 1) - 2 << std::endl << std::endl;
	return lutall[((byte >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2];		// Should see if this can be optimised
}

__device__ __forceinline__ float multibitsread_gpu(int32_t word, int pos, int nbit) {
	// Larger numbers of bits have equidistant quantization spacing	

	long bitmax = (1L << nbit);

	float quant_factor;
	// TODO: define constants in a header maybe?
	switch (nbit) {
	case 4:
		quant_factor = 1.0 / FourBit1sigma;
		break;
	case 8:
		quant_factor = 1.0 / 3.3;
		break;
	case 16:
		quant_factor = 1.0 / 8.0;
		break;
	case 32:
		quant_factor = 1.0 / 8.0;		// ERROR: 32 bit doesn't work for some reason?
		break;
	default:
		break;
	}

	return (((word >> pos) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) * quant_factor;
}


__device__ int blanker_vdif_gpu(struct mark5_stream *ms)
{
	unsigned long long *data;
	int nword;
	
	if(!ms->payload)
	{
		ms->blankzoneendvalid[0] = 0;

		return 0;
	}

	data = (unsigned long long *)ms->payload;

	nword = ms->databytes/8;

	/* only 1 zone for VDIF data.  a packet is either good or bad. 
	 *
	 * To be good, it cannot have fill pattern at beginning or end 
	 */

	ms->blankzonestartvalid[0] = 0;

	/* Check for fill pattern */
	if(data[0] == MARK5_FILL_WORD64 || data[nword-1] == MARK5_FILL_WORD64)
	{
		ms->blankzoneendvalid[0] = 0;
		return 0;
	}
	else
	{
		//fprintf(m5stderr, "Frame is good\n");
		ms->blankzoneendvalid[0] = 1<<30;
		return 1;
	}
}

__device__ int mark5_stream_next_frame_gpu(struct mark5_stream *ms)
{
	int n;
	int v = 0;

	/* call specialized function to ready next frame */
	n = ms->next(ms);

	/* are we at end of file(s)? */
	if(n < 0)
	{
		ms->payload = 0;
		
		return -1;
	}
	
	if(ms->frame)
	{
		/* validate frame */
		v = ms->validate(ms);
		if(!v)
		{
			++ms->nvalidatefail;
			++ms->consecutivefails;
		}
		else
		{
			++ms->nvalidatepass;
			ms->consecutivefails = 0;
		}
	}

	/* set payload pointer to point to start of actual data */
	if(ms->frame)
	{
		ms->payload = ms->frame + ms->payloadoffset;
	}
	
	/* blank bad data if any */
	if(v)
	{
		ms->blanker(ms);
	}
	else /* blank entire frame if validity check fails */
	{
		//mark5_stream_blank_frame(ms);
		printf("If you are reading this then the line above needs to be uncommented and implemented on the GPU");
	}

	return 0;
}

__device__ int mk5_decode_general_gpu(struct mark5_stream *ms, int nsamp, float **data) {
    const unsigned char *buf;
	int o;
	int nbit, nchan, decimation;
	int nblank = 0;
	long long packed_buffer;

	buf = ms->payload;
	nbit = ms->nbit;
	nchan = ms->nchan;
	decimation = ms->decimation;



	int bit_counter = 0;

	int decomp_factor = 8 / (ms->nbit * ms->nchan);

	// Get the number of skipped channels (if nchan != 2^n)
	int skipped = 0;
	int n = nchan;
	while (n != 0) {
		n = n >> 1;
		skipped++;		
	}
	// 2^skipped is the largest power of 2 greater than nchan
	skipped = ((1 << skipped) - nchan) % nchan;

	bool bitreadflag = (nbit == 1) || (nbit == 2);

	int start = decomp_factor * ms->databytes * ms->framenum;
	for(o = start; o < start + nsamp; o++) {

		if (bit_counter / 8 >= ms->blankzoneendvalid[0])
		{
			// This entire sample is zero. Store in data and skip ahead
			for (int c = 0; c < nchan; c++) {
				data[c][o] = 0.0;
				bit_counter += nbit;
			}
			nblank++;
			
		} else {
			// Iterate over all the channels to read from this sample
			for (int c = 0; c < nchan; c++) {
				if (bitreadflag) {
					data[c][o] = bitsread_gpu(buf[bit_counter / 8], bit_counter % 8, nbit);
				} else {
					data[c][o] = multibitsread_gpu(((u_int32_t*)buf)[bit_counter / 32], bit_counter % 32, nbit);
				}
				bit_counter += nbit;
			}
		}

		// If there are ignored channels, skip over them now
		bit_counter += nbit * skipped;

		// If there is decimation, skip forward correspondingly
		// Decimation not really used so this hasn't been tested much
		bit_counter += (decimation - 1) * nbit * nchan;
	}

	// Return value is no longer needed but exists because mark5 expects one
	return nsamp-nblank;
}

__device__ int validate_gpudata(const struct mark5_stream *ms) {
	return 1;	// The data is perfect and no one can tell it otherwise
}


__device__ int mark5_stream_unpacker_next_gpu(struct mark5_stream *ms) {
	return 1;	// The data is perfect and no one can tell it otherwise
}

__global__ void gpu_unpack(struct mark5_stream *ms, const void *packed, float **unpacked, int nframes, bool *goodframes) {
	// Set up the required function pointers
	ms->decode = *mk5_decode_general_gpu;
    ms->validate = *validate_gpudata;
	ms->next = *mark5_stream_unpacker_next_gpu;
	ms->blanker = *blanker_vdif_gpu;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nframes) {
		return;
	}

	mark5_stream thread_ms = *ms;

	thread_ms.frame = (const unsigned char *)packed + index * ms->framebytes;
	thread_ms.payload = thread_ms.frame + thread_ms.payloadoffset;
	thread_ms.readposition = 0;
	thread_ms.framenum = index;

	// Check whether this frame is valid
	goodframes[index] = thread_ms.blanker(&thread_ms);
	
	// Now actually decode this frame
	thread_ms.decode(&thread_ms, thread_ms.framesamples, unpacked);

}