# GPU correlator: the plan

Living plan for the GPU work on `adam-performance-gains`. Companion to
`gpu-changes.md` (what has already happened) and
`tests/FakeData/BENCHMARKS.md` (the numbers). Updated as items land.

## Where we are

Correctness is **done** for the USB scenarios (usb, usb-complex,
complex-complex, and the 5-station/4-subband `multi` scenario PASS
CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes; LSB deliberately
unimplemented). The **performance** phase has landed a series of Core-side
de-serializations — Lever A (pinned input), weights-on-device Increments
1/2/2b, fringe-rotation interpolator hoisting, host-tail overlap, the
`Mk5_GPUMode::unpack_all` drain removal + RING-deep host staging, and the
`gpuprocslot` refactor — taking the T5-T1 benchmark 66.7 → ~22.9 s. Full
history, rationale and the (now-corrected) cuFFT-sync misdiagnosis are in
`gpu-changes.md` (§5-10); numbers in `BENCHMARKS.md`.

**Current understanding (2026-07-22): the GPU path is no longer the
bottleneck — the DataStream interlaced-VDIF corner-turn is.** Whole-pipeline
profiling (Manager + DataStream + Core) showed the GPU is faster than the
DataStream can feed it: the DataStream burns ~93% of its busy CPU
multiplexing 16 interlaced VDIF threads into one (`VDIFMuxer`,
`cornerturn_16thread_2bit`) at ~1.7 Gbps/core — at/below the record rate — so
the GPU then idles ~42% on the procslot mutex. The Core-side work above was
correct de-serialization but aimed at a non-bottleneck. Removing the
corner-turn (single-thread VDIF) cut a 2-station job 8.0 → 4.8 s (~40%). See
gpu-changes.md §10 and the Longer-term item below. Kernel mix (A100): fringe
family ~45%, unpack 24%, fused XMAC 16%, FFT 8%, weights ~7% (FP64 cheap
there, so precision work leans on the 2070).

Completed work has moved to `gpu-changes.md`; the queue below is what remains.

**Next action: item 1 (fuse unpack + fringe rotation).** Item 4 was investigated
and closed with no change (memory-bound, already optimal fused). The remaining
GPU-busy targets are item 1 (unpack+fringe fusion, the largest kernel on the
cluster) and item 8 (occupancy audit, incl. the `<<<1,1>>>` `gpu_sum_weights`).

## Work queue (underway + future)

1. **Fuse the unpack and fringe-rotation kernels.** Today `gpu_unpack`
   writes the unpacked samples to global memory and the fringe-rotation
   kernel reads them straight back; fusing keeps samples in
   registers/shared memory and saves that global-memory round-trip.
   Motivated by the 2026-07-21 profiles: `gpu_unpack` is the single
   largest kernel on the cluster (34.9% of GPU time vs 19.9% on the 2070)
   now that fringe rotation is cheap on fast-FP64 cards, so the memory
   round-trip between them is the thing to cut. Fusing also removes two
   kernel launches per (datastream, subint) - a minor secondary benefit
   (launch overhead is only ~3-4% of host time). Interacts with the FP16
   and precision-drop items below. NOTE: this attacks GPU-busy time, which
   is not the current bottleneck (see "Where we are").
2. **Optional FP16 unpack + fringe rotation** (with 16-bit -> 32-bit
   FFT): an opt-in reduced-precision path that carries the unpacked
   samples and the fringe rotation in FP16, with the FFT taking FP16 in
   and producing FP32 output. Roughly halves the memory traffic for those
   stages (and the compute on cards with fast FP16). Needs an accuracy
   assessment against the correlator's dynamic range; gated so the
   full-precision path stays the default. Combines naturally with the
   kernel fusion (item 1), and subsumes the per-sample precision drop
   (item 3) - if FP16 lands, item 3 is moot.
3. **Fringe-rotation per-sample precision drop** (follow-up to the
   completed hoisting; changes results at FP level, so its own accuracy
   check; a cheaper stepping-stone that the FP16 item above would
   subsume). After the hoisting, the per-sample math is `exponent =
   bigAval * (double)channelindex + bigB_reduced`, reduced to [0,1) before
   the FP32 `__sincosf`. `bigB_reduced` is bounded to [0,1) so it can be a
   `float`. The one quantity that genuinely needs `double` is
   `bigAval * (double)channelindex` — `channelindex` reaches thousands at
   high spectral resolution and `bigAval` can be large when the LO
   frequency is high. So: compute `bigAval * (double)channelindex` in
   double, reduce to [0,1), then add the `float` `bigB_reduced` and reduce
   again — keeping the single necessary FP64 multiply while everything
   else is float (the precompute can then emit `bigB_reduced` as float
   too, halving the gBigBred traffic). Not bit-identical; validate with
   diffDiFX/SPECDEBUG.
4. **DONE (investigated 2026-07-22, no change made): fractional sample
   correction / `gpu_resultsrotatorMultiply`.** It runs ~2× a fringe-rotation
   kernel, but that is because it is **memory-bound**, not because of the
   autocorr index math (that guess was wrong). It is already optimal as a
   **single fused pass** (rotate → conjugate → autocorrelate while the sample
   is hot in registers/L1). Two changes were tried and both rejected on a
   clean build: (a) splitting it into rotate/conj + autocorr kernels is ~2×
   SLOWER (the autocorr kernels re-read fftd/conj COLD from global — fission
   breaks the single-pass locality); (b) a `gFracSlope` precompute is
   NEUTRAL (14.2 vs 14.4 s — the kernel isn't ALU-bound, so hoisting the FP64
   recompute buys nothing). Left fused as-is. See gpu-changes.md.
5. **perbandweights on GPU** — currently unused on the GPU path but
   should be active, in analogy with CPUMode/Mk5Mode's interlaced-VDIF
   handling (identified in the de-serialization design review). Shape
   the device weights kernel (de-serialization Increment 1) so
   per-(window, band) weights are a natural extension.
6. **LSB on GPU** — unblocks the lsb/lsb-complex/dsb scenarios.
7. **CODIF on GPU** — format-aware frame-validity hook (FIXME in
   `blanker_vdif_gpu`) + widen the `getMode` format gate. MUST add the
   `datalengthbytes <= getMaxDataBytes` clamp in `process_gpu` first:
   non-VDIF datastreams keep the base class's guard-scaled sendbytes,
   which can exceed the packed-data buffer (latent, unreachable today).
8. **Kernel launch-configuration / occupancy audit.** Verify every GPU
   kernel launches an appropriate grid/block size - enough threads to
   fill the device, a block shape with good occupancy, and no accidental
   under- or over-subscription. Several kernels were written with ad-hoc
   launch dims (one thread per (window, band, channel) in fringe rotation;
   one thread per accumulator in the weight reductions; single-thread
   reductions like `gpu_sum_weights`); check each with nsys/`ncu` and fix
   any that are badly sized. Cheap, and a natural companion to the
   fusion/occupancy work in items 1-3.

## Standing process (adopted 2026-07-18)

- **Benchmark ledger**: every landed change gets a
  `tests/FakeData/run-bench.sh` run (metric T5−T1, best-of-3) recorded
  in `BENCHMARKS.md`. Since 2026-07-22 the benchmark defaults to
  single-thread VDIF (`SINGLE_THREAD_VDIF=0` to keep the interlaced demux)
  so it measures the GPU, not the DataStream corner-turn. Publishing the
  ledger to the fork's GitHub wiki is intended once the wiki is initialised
  (the wiki is a git repo: `ADACS-Australia/ADeller2023B_difx.wiki.git`).
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

- **Cheaper interlaced-VDIF demux on the DataStream — THE current bottleneck
  for the GPU correlator** (found 2026-07-22; full analysis in gpu-changes.md
  §10). The DataStream burns ~93% of its CPU bit-banging 16 interlaced VDIF
  threads into one (`VDIFMuxer`) at ~1.7 Gbps/core, starving the GPU. **Goal:**
  an unpacker path that handles interlaced VDIF by *reordering* the per-thread
  frames into time order (pure memcpy) + GPU channel de-interleave, instead of
  the CPU multiplex, so real interlaced recordings stop gating the GPU. (For
  fake-data benchmarking, `benchprof-profile.sbatch` already rewrites the
  `.input` `INTERLACEDVDIF/… → VDIF` to bypass the demux.)
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
- **Templating the unpack**: Can we make the unpack on the GPU neater 
  (possibly in conjunction with fusing with fringe rotation).
- **Use cufftdx to fuse the FFT and fractional sample correction**: 
  If stalls at the end of FFTs continues to be a problem causing 
  GPU idle time.
- **Compute/H2D overlap across datastreams (the "Option A / Option B"
  ideas)**: overlap the input H2D with GPU compute - Option A intra-subint
  (datastream j+1's H2D during datastream j's compute), Option B
  inter-subint (prefetch the next subint's packed data during the current
  subint's FFT+frac, holding two procslots). Demoted here from the work
  queue on 2026-07-21: the post-fringe profiles show input H2D
  (`h2d_stage`) is only ~0.6% of wall on the 2070 and ~0.9% on the
  cluster, so overlapping it is worth <1% either way. If it ever becomes
  worthwhile, the machinery is seeded (Lever A's `h2dInputDone`, a
  dedicated H2D stream), and Option A is far simpler than Option B (the
  per-datastream buffers are already separate, so it needs only a
  dedicated H2D stream, no double-buffering).
- NUMA/affinity audit (mattered on the cluster, less on the desktop).
- Profile the fftsPerChunk XMAC grid split if atomics show up.
- GPU pcal regression coverage (currently untested by diffDiFX).
