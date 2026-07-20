# GPU correlator: the plan

Living plan for the GPU work on `adam-performance-gains`. Companion to
`gpu-changes.md` (what has already happened) and
`tests/FakeData/BENCHMARKS.md` (the numbers). Updated as items land.

## Where we are

Correctness is done for the USB scenarios (usb, usb-complex,
complex-complex, and the 5-station/4-subband `multi` scenario PASS
CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes; LSB deliberately
unimplemented). The current phase is **performance**: the GPU is
gap-dominated because one Core thread alternates host and GPU work.
Landed so far: Lever A (pinned input buffers, direct H2D, 2026-07-18);
de-serialization Increment 1 (set_weights on the device, 2026-07-18);
Increment 2 (baseline weights on the device, 2026-07-20); Increment 2b
(autocorrelation weights on the device, dropping the last interim bulk
`gDataWeights` D2H, 2026-07-21). Increments 1-2 cut the T5-T1 benchmark
66.7 -> 32.4 s; 2b holds it flat (a cleanup, not a perf lever). The
biggest remaining host cost was `host_accumulate`'s baseline-weight loop,
now gone; the next perf work (see the work queue below) is the
fringe-rotation interpolator hoisting, then overlapping the
per-datastream H2D and host work with GPU compute.

## Completed activities

- **Multi-subband, N>2-station Synthetic scenario** (2026-07-18): the
  `multi` scenario — 5 stations x 4 subbands (test-multi.vex/v2d,
  4-channel VDIF from generateVDIF) — PASSes CPU-vs-GPU in both pipeline
  modes and is wired into run-local.sh and run-slurm.sh (rank counts now
  sized per scenario).
- **De-serialize the per-datastream loop — weights arc**: Increment 1
  (set_weights on the device, 2026-07-18) removed the per-datastream
  stream drains; Increment 2 (baseline weights on the device,
  2026-07-20) removed `host_accumulate`'s dominant per-window loop;
  Increment 2b (AC weights on the device, 2026-07-21) dropped the last
  interim bulk `gDataWeights` D2H (now only under the WDEBUG gate).
  Together: T5-T1 66.7 -> 32.4 s. (Prose detail in gpu-changes.md; the
  remaining gap-closing — compute/H2D overlap — is a future item below.)

## Work queue (underway + future)

1. **Fringe-rotation interpolator hoisting** (next perf gain). The
   `gpu_fringeRotation` / `gpu_complex_fringeRotation` kernels launch one
   thread per (FFT window, band, channel) and *every* thread recomputes,
   from scratch, quantities that vary far more coarsely — so this FP64
   work (`fringeRotation` is ~66% of GPU time on GeForce, ~25% on
   data-centre cards) is repeated `numrecordedbands x fftchannels` times
   more than necessary. Compute each quantity only as often as it varies:
   - `d0`/`d1`/`d2` -> the per-window `a`/`b` can be batch-computed once
     per subint (one `a`/`b` pair per FFT window; `d0`/`d1`/`d2` are
     internal per-window intermediates of that single per-subint pass);
   - `a`/`b`: once per FFT window (the CPU does exactly this — once per
     `CPUMode::process` call, cpumode.cpp ~line 390);
   - `bigAval`/`bigBval`/`bigB_reduced`: once per (window, band) — these
     vary with `lofreqs[bandindex]`, so per band within a window, finer
     than `a`/`b`.
   The per-sample kernel then does only the final `exponent` +
   `__sincosf` + complex multiply. Approach: precompute the per-window
   `a`/`b` (and optionally per-(window,band) `bigA`/`bigB`) into device
   arrays once per subint (a small kernel, or extend `calculatePre`) and
   pass them into the rotation kernel. Must stay FP-parity with the CPU
   (WDEBUG/SPECDEBUG, diffDiFX). Supersedes the old "reduce fringeRotation
   FP64" note, and dovetails with the FP16 / kernel-fusion items below.
2. **Compute/H2D overlap across datastreams.** Overlap datastream j+1's
   input H2D and host-side work with datastream j's GPU compute, and
   overlap `host_accumulate`'s residual (autocorr flush, pcal) with the
   next subint. Machinery anticipated by Lever A: move input copies to a
   dedicated H2D stream and record `h2dInputDone` there (see note in
   `issuegpudata`); relax the per-pass `cudaStreamSynchronize`.
3. **perbandweights on GPU** — currently unused on the GPU path but
   should be active, in analogy with CPUMode/Mk5Mode's interlaced-VDIF
   handling (identified in the de-serialization design review). Shape
   the device weights kernel (de-serialization Increment 1) so
   per-(window, band) weights are a natural extension.
4. **LSB on GPU** — unblocks the lsb/lsb-complex/dsb scenarios.
5. **CODIF on GPU** — format-aware frame-validity hook (FIXME in
   `blanker_vdif_gpu`) + widen the `getMode` format gate. MUST add the
   `datalengthbytes <= getMaxDataBytes` clamp in `process_gpu` first:
   non-VDIF datastreams keep the base class's guard-scaled sendbytes,
   which can exceed the packed-data buffer (latent, unreachable today).

## Standing process (adopted 2026-07-18)

- **Benchmark ledger**: every landed change gets a
  `tests/FakeData/run-bench.sh` run (metric T5−T1, best-of-3) recorded
  in `BENCHMARKS.md`. Publishing the ledger to the fork's GitHub wiki is
  intended once the wiki is initialised (the wiki is a git repo:
  `ADACS-Australia/ADeller2023B_difx.wiki.git`).
- **Prose documentation**: `gpu-changes.md` is extended with every
  substantive change, at the same time as the change.
- **Full diff review before every commit.**

## Future infrastructure

- **Continuous integration** on ar313 (and/or the cluster): the natural
  shape is a self-hosted GitHub Actions runner on ar313 executing
  `tests/Synthetic/run-local.sh` (correctness gate) and `run-bench.sh`
  (ledger row) on push/nightly; a cluster leg could submit
  `run-slurm.sh` via a SLURM-side runner for scale testing. Needs: runner
  install (interactive auth), deciding failure notification, and keeping
  GPU jobs serialized on the single 2070.

## Refactor (before merging back into DiFX main)

The fork has diverged significantly from upstream; a structural clean-up
will ease the eventual merge and put GPU code where it belongs:

- `Core` becomes a virtual base with `CPUCore` and `GPUCore` as the
  concrete implementations (today GPUCore inherits a concrete CPU Core).
- `Mode` becomes a virtual base; `CPUMode` concrete; `Mk5Mode` a virtual
  inheritor covering the frame-based formats; `CPUMk5Mode` and
  `GPUMk5Mode` concrete under it. Rationale: the GPU path is restricted
  to frame-based formats by design (no plans for non-frame formats), so
  the GPU implementation belongs under the frame-based branch of the
  hierarchy rather than shadowing all of Mode.
- Same pass: decide keep/remove for the WDEBUG/HDRDEBUG/SPECDEBUG and
  NVTX scaffolding, unify the duplicated pcal reset logic (the agreed
  PCal "ingest folded bins" interface), and clear the latent-issues list
  (heterogeneous-datastream XMAC plans, NOT_SUPPORTED guard for multiple
  phase centres/pulsar binning on GPU, per-call stream create/destroy in
  `processgpudata`).

## Longer-term / opportunistic

- **Subints larger than GPU memory**: today a subint's full set of FFT
  windows must fit on the device, so large station counts or data rates
  force short subints — at the cost of a higher visibility rate from the
  Cores back to the manager. Broaden GPUMode/GPUCore to slice a subint
  into chunks of FFT windows processed a subset at a time (the dormant
  fftloops machinery is the natural seam). Interacts with the VRAM
  budget check, cfg_numBufferedFFTs sizing, and the numfftloops == 1
  assumptions documented in de-serialization Increment 1.
- **Tensor cores for the XMAC**: the cross-multiply-accumulate is a
  complex GEMM at heart, a natural tensor-core target (Turing: FP16
  multiply / FP32 accumulate; data-centre cards add TF32/FP64 paths).
  Needs an accuracy assessment against the correlator's dynamic range.
- **Optional FP16 unpack + fringe rotation** (with 16-bit -> 32-bit
  FFT): an opt-in reduced-precision path that carries the unpacked
  samples and the fringe rotation in FP16, with the FFT taking FP16 in
  and producing FP32 output. Roughly halves the memory traffic for those
  stages (and the compute on cards with fast FP16). Needs an accuracy
  assessment against the correlator's dynamic range; gated so the
  full-precision path stays the default.
- **Fuse the unpack and fringe-rotation kernels**: today `gpu_unpack`
  writes the unpacked samples to global memory and the fringe-rotation
  kernel reads them straight back. Fusing the two keeps samples in
  registers/shared memory and saves that global-memory round-trip
  (memory-bandwidth bound). Interacts with the FP16 path above and with
  the per-window `a`/`b` hoisting (work-queue item 1).
- NUMA/affinity audit (mattered on the cluster, less on the desktop).
- Profile the fftsPerChunk XMAC grid split if atomics show up.
- GPU pcal regression coverage (currently untested by diffDiFX).
