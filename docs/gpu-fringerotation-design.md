# Design: hoisting the GPU fringe-rotation interpolator work

Status: LANDED 2026-07-21. Companion to gpu-plan.md (work-queue history)
and gpu-changes.md (prose changelog).

## Problem

The fringe-rotation kernels `gpu_fringeRotation` (real samples) and
`gpu_complex_fringeRotation` (complex samples) are launched with one
thread per **(FFT window, band, channel)** — the finest granularity of
the correlator's inner loop. Each thread recomputed, in FP64, the entire
delay-interpolator chain:

- `d0`/`d1`/`d2` = the delay polynomial evaluated at `index`, `index+0.5`,
  `index+1` (depends only on the per-subint `interpolator[0..2]` and the
  per-**window** `index`);
- `a = d2 - d0`, `b = d0 + (d1 - (a*0.5 + d0))/3` (per **window**);
- `bigAval = a*lofreq[band]/fftchannels - sampletime*1e-6*looffset[band]`,
  `bigBval = b*lofreq[band]`, `bigB_reduced = bigBval - int(bigBval)`
  (per **(window, band)**, because of `lofreq[band]`).

Only the final `exponent = bigAval*channel + bigB_reduced` (reduced to
[0,1)) → FP32 `__sincosf` → complex multiply is genuinely per-sample. So
the FP64 chain was repeated `numrecordedbands x fftchannels` times more
than necessary. On GeForce, FP64 runs at 1/32 of FP32, and profiling put
`fringeRotation` at ~66% of GPU time — so this redundancy was the single
largest GPU cost.

## Change

A new kernel `gpu_precompute_fringe_rotator`, launched once per subint
with one thread per (window, band) (`<<<numBufferedFFTs, numrecordedbands>>>`
on the compute stream, ahead of the rotation kernel), computes `bigAval`
and `bigB_reduced` and stores them into device arrays `gBigA`/`gBigBred`
(layout `[window * numrecordedbands + band]`, FP64, device-only, sized
`cfg_numBufferedFFTs x numrecordedbands`).

The two rotation kernels drop the whole `d0…bigB` block and their now-unused
parameters (`interpolator`, `lofreqs`, `recordedfreqlooffsets`,
`sampletime`), and instead read `bigAval`/`bigB_reduced` from the arrays.
All threads sharing a (window, band) read the same two values (a broadcast
from L2), so the per-sample work is now just the phase FMA + `__sincosf` +
multiply.

The precompute uses the identical FP64 expressions in the identical order,
so the result is numerically equivalent to the old inline code. (End-to-end
bit-identity is neither claimed nor observable: the downstream XMAC
accumulates visibilities with `atomicAdd`, whose ordering is
nondeterministic, so the GPU's `.difx` output is not bit-reproducible
run-to-run — two runs of the *same* binary differ at the FP level. This
predates and is orthogonal to this change.)

## Correctness

The change touches only the fringe-rotation stage and is independent of the
weights path, so it was validated by the standard CPU-vs-GPU regression:
8/8 Synthetic scenarios (usb, usb-complex, complex-complex, multi) PASS
diffDiFX on the device-weights path AND the `DIFX_GPU_WEIGHTS_HOST`
fallback, in both `DIFX_GPU_PIPELINE` modes. (The `cmp` bit-identity idea
was abandoned once run-to-run nondeterminism from the XMAC atomics was
confirmed.)

## Result

`tests/FakeData/run-bench.sh` (10-station, T5-T1 best-of-3, RTX 2070):
**32.4 -> 22.9 s (~29%)** — the largest single-change win since
de-serialization Increment 1.

## Follow-up (not done here)

Drop the per-sample precision: `bigB_reduced` is bounded to [0,1) and can
be `float`; only `bigAval * (double)channelindex` needs FP64 (channel can
reach thousands at high spectral resolution and `bigAval` can be large at
high LO frequency). Compute that product in double, reduce to [0,1), then
add the `float` `bigB_reduced` and reduce again. Changes results at FP
level (not bit-identical) — needs its own diffDiFX/SPECDEBUG accuracy
check. Recorded as a work-queue item in gpu-plan.md; dovetails with the
longer-term FP16 unpack+fringe-rotate path.
