#include <mark5access.h>
#include <mark5access/mark5_stream.h>
#include <vdifio.h>
#include "gpudecode.cuh"
#include <iostream>
#include <bitset>
#include <stdio.h>

#define MARK5_FILL_WORD64 0x1122334411223344ULL

// 2*pi to double precision - matches architecture.h's TWO_PI (IPP_2PI) that the
// non-fused fringe kernels used, so the fused rotator below is numerically
// equivalent. Defined locally because gpudecode.cu does not include the IPP/FFTW
// architecture.h layer.
#ifndef GPUDECODE_TWO_PI
#define GPUDECODE_TWO_PI 6.283185307179586476925286766559
#endif

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

__device__ __forceinline__ cuFloatComplex complex_bitsread_gpu(char byte, int pos, int nbit) {
	return make_cuFloatComplex(lutall[((byte >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2], lutall[((byte >> (pos+nbit)) & ((1 << nbit) - 1)) + (nbit << 1) - 2]);
}


__device__ __forceinline__ float multibitsread_gpu(int32_t word, int pos, int nbit) {
	// Larger numbers of bits have equidistant quantization spacing

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


__device__ __forceinline__ cuFloatComplex complex_multibitsread_gpu(int32_t word, int pos, int nbit) {
	// Larger numbers of bits have equidistant quantization spacing

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
	default:
		break;
	}

	return make_cuFloatComplex(((((word >> pos) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) * quant_factor),((((word >> (pos+nbit)) & ((1L << nbit) - 1)) - (1L << (nbit - 1))) * quant_factor));
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

	/* Reject frames that fail the VDIF header sanity checks.  On the CPU
	 * path these live in mark5_format_vdif_validate() (called per frame
	 * from mark5_stream_next_frame(), which blanks the whole frame on
	 * failure); the GPU unpacker stubs validate out, so they belong here.
	 * As validate runs in the unpacker context (ms->mjd == 0, so its
	 * frame-time consistency test is inert), exactly two checks are live:
	 *  - Data Frame Length of zero ("overly unusual header"): catches
	 *    zero-padding, e.g. a send buffer tail past the end of the
	 *    recording that no real frame was ever written into.
	 *  - The invalid bit: set by vdifmux on frames it fabricates for
	 *    missing input data, and by recorders for known-bad frames.
	 * Without these the GPU decodes and correlates junk that the CPU
	 * correctly blanks.  This reads the vdifio vdif_header bitfields
	 * directly rather than calling getVDIFFrameInvalid()/
	 * getVDIFFrameBytes(), which are host-only static inlines.
	 *
	 * FIXME: hard-coding vdif_header is only safe because the GPU path is
	 * restricted to VDIF-family formats (enforced in Configuration::getMode).
	 * If/when the GPU branch grows CODIF (or other) support, the frame
	 * validity test must be made format-aware - a per-format validate hook
	 * like the CPU's mark5_format_*_validate(), selected at setup time,
	 * with formats that carry no per-frame validity flag skipping the test. */
	if(ms->frame)
	{
		const vdif_header *vh = (const vdif_header *)ms->frame;
		if(vh->invalid || vh->framelength8 == 0)
		{
			ms->blankzoneendvalid[0] = 0;
			return 0;
		}
	}

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

// The number of ignored channels when nchan is not a power of two (the packed
// bitstream carries the next power of two of channels; the unused ones are
// skipped). Depends only on nchan, so the launcher precomputes it once per
// subint on the host and passes it in - out of the per-thread inner loop that
// the old gpu_unpack recomputed it in.
__host__ __device__ __forceinline__ int channel_skip(int nchan) {
	int skipped = 0;
	int n = nchan;
	while (n != 0) { n >>= 1; skipped++; }
	return ((1 << skipped) - nchan) % nchan;
}

// Decode a SINGLE real sample for one band directly from the packed frame
// payload. This is the per-(band, sample) primitive shared by the fused
// decode+fringe kernel; the old gpu_unpack looped this over every channel and
// wrote the result to a global unpacked buffer that the fringe kernel then read
// straight back. framepayload = packed + frame*framebytes + payloadoffset.
__device__ __forceinline__ float decode_one_gpu(
		const unsigned char *framepayload, int nbit, int nchan, int decimation,
		int skipped, int sample_in_frame, int band) {
	const bool bitreadflag = (nbit == 1) || (nbit == 2);
	int bit_counter = sample_in_frame * nbit * (nchan * decimation + skipped) + band * nbit;
	if (bitreadflag)
		return bitsread_gpu(framepayload[bit_counter / 8], bit_counter % 8, nbit);
	else
		return multibitsread_gpu(((const u_int32_t *)framepayload)[bit_counter / 32], bit_counter % 32, nbit);
}

// Complex twin of decode_one_gpu (real+imag packed adjacently per sample, so
// twice the bits per band and per sample; mirrors the old
// mk5_decode_complex_sample_gpu bit layout).
__device__ __forceinline__ cuFloatComplex decode_one_complex_gpu(
		const unsigned char *framepayload, int nbit, int nchan, int decimation,
		int skipped, int sample_in_frame, int band) {
	const bool bitreadflag = (nbit == 1) || (nbit == 2);
	int bit_counter = sample_in_frame * 2 * nbit * (nchan * decimation + skipped) + band * 2 * nbit;
	if (bitreadflag)
		return complex_bitsread_gpu(framepayload[bit_counter / 8], bit_counter % 8, nbit);
	else
		return complex_multibitsread_gpu(((const u_int32_t *)framepayload)[bit_counter / 32], bit_counter % 32, nbit);
}

// Per-frame validity, one thread per frame. Replaces the validity side of the
// old gpu_unpack (which set goodframes[] as a by-product of decoding every
// sample): the fused decode+fringe kernel decodes samples itself, so all that
// is needed up front is the good/bad flag per frame that gpu_set_weights and
// the fused kernel read. blanker_vdif_gpu also sets ms.blankzoneendvalid on the
// thread-local copy, but only its int return (good=1/bad=0) is kept.
__global__ void gpu_blank_frames(struct mark5_stream ms, const void *packed, int nframes, bool *valid_frames) {
	int frame = blockIdx.x * blockDim.x + threadIdx.x;
	if (frame >= nframes) return;

	ms.frame        = (const unsigned char *)packed + (size_t)frame * ms.framebytes;
	ms.payload      = ms.frame + ms.payloadoffset;
	ms.framenum     = frame;

	valid_frames[frame] = blanker_vdif_gpu(&ms);
}

// ===========================================================================
// Fused decode + fringe-rotation kernel.
//
// Replaces the old two-pass gpu_unpack (write unpacked samples to a global
// buffer) + gpu_fringeRotation (read them straight back). Each thread owns one
// (FFT window, band, channel), decodes exactly the one sample it needs from the
// packed frame payload (keeping it in a register), applies the precomputed
// fringe rotator and writes the complex FFT input - no global unpacked buffer,
// no round-trip. Thread/grid mapping and the destIndex/rotator math are
// identical to the old gpu_fringeRotation, so the rotated output is unchanged.
//
// When DOPCAL, the SAME raw decoded sample (pre-rotation) is also folded into
// its phase-cal bin, subsuming the standalone gpu_pcalextraction kernel (whose
// only real work was this binning atomicAdd - the offset/phase assembly is done
// host-side later). The bin index reproduces gpu_pcalextraction exactly:
// (datasamples + sampleIndexes[window] + channel) % N_pcal_bins[band]. The
// no-pcal instantiation compiles all of this away.
//
// Blanking/tail parity with the old path: a sample whose frame is invalid
// (!valid_frames[frame]) or past the delivered data (frame >= framestounpack)
// decodes to 0 - exactly what gpu_unpack's per-frame blankzone zeroing and the
// explicit unpacked-tail memset produced.
template<bool DOPCAL>
__global__ void gpu_fused_fringe(
		cuFloatComplex *const dest,
		const int *const sampleIndexes,
		const bool *const validSamples,
		const double *const bigA,
		const double *const bigBred,
		int fftloop,
		int startblock,
		int numblocks,
		size_t fftchannels,
		struct mark5_stream ms,
		const void *packed,
		const bool *const valid_frames,
		int skipped,
		int framestounpack,
		float *pcal_output,
		const int *N_pcal_bins,
		int datasamples,
		int pcal_bin_stride_length) {
	const size_t subloopindex = blockIdx.x;
	if (!validSamples[subloopindex]) return;

	size_t index = fftloop * gridDim.x + subloopindex + startblock;
	if (index >= startblock + numblocks) return;

	const size_t bandindex = threadIdx.x;
	const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
	const size_t numrecordedbands = blockDim.x;
	if (channelindex >= fftchannels) return;

	// Decode this thread's one raw sample on the fly (was: read src[band][idx]
	// from the global unpacked buffer).
	const int sample_index = sampleIndexes[subloopindex];
	const long global_sample = (long)sample_index + (long)channelindex;
	float srcVal = 0.f;
	if (global_sample >= 0) {
		const int frame = (int)(global_sample / ms.framesamples);
		if (frame < framestounpack && valid_frames[frame]) {
			const int sample_in_frame = (int)(global_sample - (long)frame * ms.framesamples);
			const unsigned char *fp = (const unsigned char *)packed +
					(size_t)frame * ms.framebytes + ms.payloadoffset;
			srcVal = decode_one_gpu(fp, ms.nbit, ms.nchan, ms.decimation, skipped, sample_in_frame, (int)bandindex);
		}
	}

	if (DOPCAL) {
		const int n_bins = N_pcal_bins[bandindex];
		const long so = (long)datasamples + global_sample;
		const int bin = (int)(((so % n_bins) + n_bins) % n_bins);
		atomicAdd(&pcal_output[bandindex * pcal_bin_stride_length + bin], srcVal);
	}

	const size_t destIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

	/* complexrotator[j] = exp( 2 pi i * (A*j + B) ), where A/B (bigAval and
	   bigB_reduced) are precomputed per (window, band) by
	   gpu_precompute_fringe_rotator - so this per-sample kernel only forms the
	   phase and applies the rotator. */
	const double bigAval = bigA[subloopindex * numrecordedbands + bandindex];
	const double bigB_reduced = bigBred[subloopindex * numrecordedbands + bandindex];
	double exponent = (bigAval * (double)channelindex + bigB_reduced);
	exponent -= int(exponent);
	cuFloatComplex cr;
	__sincosf(-GPUDECODE_TWO_PI * exponent, &cr.y, &cr.x);
	cuFloatComplex c = make_cuFloatComplex(srcVal, 0.f);
	dest[destIndex] = cuCmulf(c, cr);
}

// Complex-sampled twin of gpu_fused_fringe.
template<bool DOPCAL>
__global__ void gpu_fused_fringe_complex(
		cuFloatComplex *const dest,
		const int *const sampleIndexes,
		const bool *const validSamples,
		const double *const bigA,
		const double *const bigBred,
		int fftloop,
		int startblock,
		int numblocks,
		size_t fftchannels,
		struct mark5_stream ms,
		const void *packed,
		const bool *const valid_frames,
		int skipped,
		int framestounpack,
		cuFloatComplex *pcal_output,
		const int *N_pcal_bins,
		int datasamples,
		int pcal_bin_stride_length) {
	const size_t subloopindex = blockIdx.x;
	if (!validSamples[subloopindex]) return;

	size_t index = fftloop * gridDim.x + subloopindex + startblock;
	if (index >= startblock + numblocks) return;

	const size_t bandindex = threadIdx.x;
	const size_t channelindex = (blockIdx.y * blockDim.y) + threadIdx.y;
	const size_t numrecordedbands = blockDim.x;
	if (channelindex >= fftchannels) return;

	const int sample_index = sampleIndexes[subloopindex];
	const long global_sample = (long)sample_index + (long)channelindex;
	cuFloatComplex srcVal = make_cuFloatComplex(0.f, 0.f);
	if (global_sample >= 0) {
		const int frame = (int)(global_sample / ms.framesamples);
		if (frame < framestounpack && valid_frames[frame]) {
			const int sample_in_frame = (int)(global_sample - (long)frame * ms.framesamples);
			const unsigned char *fp = (const unsigned char *)packed +
					(size_t)frame * ms.framebytes + ms.payloadoffset;
			srcVal = decode_one_complex_gpu(fp, ms.nbit, ms.nchan, ms.decimation, skipped, sample_in_frame, (int)bandindex);
		}
	}

	if (DOPCAL) {
		const int n_bins = N_pcal_bins[bandindex];
		const long so = (long)datasamples + global_sample;
		const int bin = (int)(((so % n_bins) + n_bins) % n_bins);
		cuFloatComplex *dst = &pcal_output[bandindex * pcal_bin_stride_length + bin];
		atomicAdd(&dst->x, srcVal.x);
		atomicAdd(&dst->y, srcVal.y);
	}

	const size_t destIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

	const double bigAval = bigA[subloopindex * numrecordedbands + bandindex];
	const double bigB_reduced = bigBred[subloopindex * numrecordedbands + bandindex];
	double exponent = (bigAval * (double)channelindex + bigB_reduced);
	exponent -= int(exponent);
	cuFloatComplex cr;
	__sincosf(-GPUDECODE_TWO_PI * exponent, &cr.y, &cr.x);
	dest[destIndex] = cuCmulf(srcVal, cr);
}

// Host launcher for gpu_blank_frames (keeps the <<<>>> launch and the
// mark5_stream template plumbing in this translation unit, so the caller in
// mk5mode_gpu.cu just calls a plain function).
void launch_blank_frames(dim3 grid, dim3 block, cudaStream_t stream,
		struct mark5_stream ms, const void *packed, int nframes, bool *valid_frames) {
	gpu_blank_frames<<<grid, block, 0, stream>>>(ms, packed, nframes, valid_frames);
}

// Host launcher for the fused decode+fringe kernel. Dispatches on the
// real/complex and pcal/no-pcal axes to the four template instantiations, so
// the common (no-pcal) path carries none of the pcal code.
void launch_fused_fringe(dim3 grid, dim3 block, cudaStream_t stream,
		bool usecomplex, bool dopcal,
		cuFloatComplex *dest, const int *sampleIndexes, const bool *validSamples,
		const double *bigA, const double *bigBred,
		int fftloop, int startblock, int numblocks, size_t fftchannels,
		struct mark5_stream ms, const void *packed, const bool *valid_frames,
		int framestounpack,
		void *pcal_output, const int *N_pcal_bins, int datasamples, int pcal_bin_stride_length) {
	const int skipped = channel_skip(ms.nchan);
	if (usecomplex) {
		if (dopcal)
			gpu_fused_fringe_complex<true><<<grid, block, 0, stream>>>(
					dest, sampleIndexes, validSamples, bigA, bigBred,
					fftloop, startblock, numblocks, fftchannels,
					ms, packed, valid_frames, skipped, framestounpack,
					(cuFloatComplex *)pcal_output, N_pcal_bins, datasamples, pcal_bin_stride_length);
		else
			gpu_fused_fringe_complex<false><<<grid, block, 0, stream>>>(
					dest, sampleIndexes, validSamples, bigA, bigBred,
					fftloop, startblock, numblocks, fftchannels,
					ms, packed, valid_frames, skipped, framestounpack,
					nullptr, nullptr, datasamples, pcal_bin_stride_length);
	} else {
		if (dopcal)
			gpu_fused_fringe<true><<<grid, block, 0, stream>>>(
					dest, sampleIndexes, validSamples, bigA, bigBred,
					fftloop, startblock, numblocks, fftchannels,
					ms, packed, valid_frames, skipped, framestounpack,
					(float *)pcal_output, N_pcal_bins, datasamples, pcal_bin_stride_length);
		else
			gpu_fused_fringe<false><<<grid, block, 0, stream>>>(
					dest, sampleIndexes, validSamples, bigA, bigBred,
					fftloop, startblock, numblocks, fftchannels,
					ms, packed, valid_frames, skipped, framestounpack,
					nullptr, nullptr, datasamples, pcal_bin_stride_length);
	}
}
