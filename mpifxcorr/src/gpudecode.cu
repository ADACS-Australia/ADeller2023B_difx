#include <mark5access.h>
#include <mark5access/mark5_stream.h>
#include "gpudecode.cuh"
#include <iostream>
#include <bitset>
#include <stdio.h>

#define MK5B_PAYLOADSIZE 10000
#define MARK5_FILL_WORD64 0x1122334411223344ULL

/* the high mag value for 2-bit reconstruction */
static const float HiMag = OPTIMAL_2BIT_HIGH;
static const float FourBit1sigma = 2.95;

static float lut1bit[256][8];
static float lut2bit[256][4];
static float lut4bit[256][2];
static float lut8bit[256];
static float zeros[8];
// static float complex complex_zeros[8];

// static float complex complex_lut1bit[256][4];
// static float complex complex_lut2bit[256][2];
// static float complex complex_lut4bit[256];

static unsigned char countlut2bit[256][4];

static void initluts()
{
	/* Warning: these are different than for VLBA/Mark4/Mark5B! */
	const float lut2level[2] = {-1.0, 1.0};
	const float lut4level[4] = {-HiMag, -1.0, 1.0, HiMag};
	const float lut16level[16] = {-8/FourBit1sigma,-7/FourBit1sigma,-6/FourBit1sigma,-5/FourBit1sigma,-4/FourBit1sigma,
				      -3/FourBit1sigma,-2/FourBit1sigma,-1/FourBit1sigma,0,1/FourBit1sigma,2/FourBit1sigma,
				      3/FourBit1sigma,4/FourBit1sigma,5/FourBit1sigma,6/FourBit1sigma,7/FourBit1sigma};
	int b, i, l, li;
	
	for(i = 0; i < 8; i++)
	{
		zeros[i] = 0.0;
		//complex_zeros[i] = 0.0;
	}

	for(b = 0; b < 256; b++)
	{
		/* lut1bit */
		for(i = 0; i < 8; i++)
		{
			l = (b>>i) & 0x01;
			lut1bit[b][i] =  lut2level[l];
		}

		/* lut2bit */
		for(i = 0; i < 4; i++)
		{
			l = (b >> (2*i)) & 0x03;
			lut2bit[b][i] = lut4level[l];
			if(fabs(lut2bit[b][i]) < 1.1)
			{
				countlut2bit[b][i] = 0;
			}
			else
			{
				countlut2bit[b][i] = 1;
			}
		}

		/* lut4bit */
		for(i = 0; i < 2; i++)
		{
			l = (b >> (4*i)) & 0x0F;
			lut4bit[b][i] = lut16level[l];
		}

		/* lut8bit */
		lut8bit[b] = (b-128)/3.3;	/* This scaling mimics 2-bit data if 8 bit RMS==~10 */

		/* Complex lookups */

		/* 1 bit real, 1 bit imaginary */
		for(i = 0; i < 4; i++)
		{
		         l =  (b>> (2*i)) & 0x1;
			 li = (b>> (2*i+1)) & 0x1;
			 //complex_lut1bit[b][i] =  lut2level[l] + lut2level[li]*I;
		}

		/* 2 bit real, 2 bit imaginary */
		for(i = 0; i < 2; i++)
		{
		         l =  (b>> (4*i)) & 0x3;
			 li = (b>> (4*i+2)) & 0x3;
			 //complex_lut2bit[b][i] =  lut4level[l] + lut4level[li]*I;
		}

		/* 4 bit real, 4 bit imaginary */
		l =  b & 0xF;
		li = (b>>4) & 0xF;
		//complex_lut4bit[b] =  lut16level[l] + lut16level[li]*I;

	}
}

inline float bitsread(char byte, int pos, int nbit) {
	// Stack all quantization levels (nbit <= 4) float values in one array and use nbit to offset accordingly
	const float lutall[6] = {-1.0, 1.0, -HiMag, -1.0, 1.0, HiMag};

	// std::cout << "Byte : " << std::bitset<8>(byte) << std::endl;
	// std::cout << "Pos :  " << pos << "\tnbit :  " << nbit << std::endl;
	// std::cout << "Index: " << ((byte >> pos) & ((2 << (nbit - 1)) - 1)) + (nbit << 1) - 2 << std::endl << std::endl;
	return lutall[((byte >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2];		// Should see if this can be optimised
}

inline float multibitsread(int32_t word, int pos, int nbit) {
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

	// std::cout << "Bitmax = " << bitmax << std::endl;
	// std::cout << "Position = " << pos << std::endl;
	// std::cout << "nbit = " << nbit << std::endl;
	// std::cout << "Word  = " << std::bitset<32>(word) << std::endl;
	// std::cout << "Word2 = " << std::bitset<32>(word>>pos) << std::endl;
	// std::cout << "Word3 = " << std::bitset<32>(((word >> pos) & ((1L << nbit) - 1))) << std::endl;
	// std::cout << "endian= " << std::bitset<32>(le32toh(word)) << std::endl;
	// std::cout << "Result = " << (((word >> pos) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) << std::endl << std::endl;

	return (((word >> pos) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) * quant_factor;
}

int mk5_decode_general(struct mark5_stream *ms, int nsamp, float **data) {
    const unsigned char *buf;
	const float *fp;
	int o, i;
	int nbit, nchan, decimation;
	int nblank = 0; 

	buf = ms->payload;
	i = ms->readposition;
	nbit = ms->nbit;
	nchan = ms->nchan;
	decimation = ms->decimation;

	int bit_counter = i * 8;

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

	for(o = 0; o < nsamp; o++)
	{
		//std::cout << "Output #: " << o << std::endl;
		if (i >= ms->blankzoneendvalid[0])
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
					data[c][o] = bitsread(buf[bit_counter / 8], bit_counter % 8, nbit);
				} else {
					data[c][o] = multibitsread(((u_int32_t*)buf)[bit_counter / 32], bit_counter % 32, nbit);
				}
				bit_counter += nbit;
			}
		}

		// If there are ignored channels, skip over them now
			bit_counter += nbit * skipped;

		// If there is decimation, skip forward correspondingly
		// Decimation not really used so this hasn't been tested much
		bit_counter += (decimation - 1) * nbit * nchan;

		i = bit_counter / 8;

		if(i >= ms->databytes)
		{
			if(mark5_stream_next_frame(ms) < 0)
			{
				return -1;
			}
			buf = ms->payload;
			i = 0;
			bit_counter = 0;
		}
	}

	ms->readposition = i;

	return nsamp-nblank;
}


__device__ __forceinline__ float bitsread_gpu(char byte, int pos, int nbit) {
	// Stack all quantization levels (nbit <= 4) float values in one array and use nbit to offset accordingly
	const float lutall[6] = {-1.0, 1.0, -HiMag, -1.0, 1.0, HiMag};

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

	// std::cout << "Bitmax = " << bitmax << std::endl;
	// std::cout << "Position = " << pos << std::endl;
	// std::cout << "nbit = " << nbit << std::endl;
	// std::cout << "Word  = " << std::bitset<32>(word) << std::endl;
	// std::cout << "Word2 = " << std::bitset<32>(word>>pos) << std::endl;
	// std::cout << "Word3 = " << std::bitset<32>(((word >> pos) & ((1L << nbit) - 1))) << std::endl;
	// std::cout << "endian= " << std::bitset<32>(le32toh(word)) << std::endl;
	// std::cout << "Result = " << (((word >> pos) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) << std::endl << std::endl;

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
		return nword;
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
	const float *fp;
	int o, i;
	int nbit, nchan, decimation;
	int nblank = 0; 

	buf = ms->payload;
	i = ms->readposition;
	nbit = ms->nbit;
	nchan = ms->nchan;
	decimation = ms->decimation;

	int bit_counter = i * 8;

	int decomp_factor = 8 / (ms->nbit * ms->nchan);

	//printf("Buffer: %i\nRead pos: %i\nnSamples: %i\nnbit: %i\nnchan: %i\n\n\n", (int)buf, i, nsamp, nbit, nchan);

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
		//printf("o = %i\n", o);
		if (i >= ms->blankzoneendvalid[0])
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

		i = bit_counter / 8;

		// TODO: this can probably be removed since we now unpack frame by frame so no need to get the next frame here
		if(i >= ms->databytes)
		{
			if(mark5_stream_next_frame_gpu(ms) < 0)
			{
				return -1;
			}
			buf = ms->payload;
			i = 0;
			bit_counter = 0;
		}
	}

	ms->readposition = i;

	return nsamp-nblank;
}

__device__ int validate_gpudata(const struct mark5_stream *ms) {
	return 1;	// The data is perfect and no one can tell it otherwise
}


__device__ int mark5_stream_unpacker_next_gpu(struct mark5_stream *ms) {
	return 1;	// The data is perfect and no one can tell it otherwise
}

__global__ void gpu_unpack(struct mark5_stream *ms, const void *packed, float **unpacked, int nframes, int *goodsamples) {
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

	thread_ms.blanker(&thread_ms);
	// TODO: need atomic add
	*goodsamples += thread_ms.decode(&thread_ms, thread_ms.framesamples, unpacked);

}