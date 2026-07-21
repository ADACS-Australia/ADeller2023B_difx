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
`gDataWeights` D2H, 2026-07-21); and the fringe-rotation interpolator
hoisting (2026-07-21). Increments 1-2 cut the T5-T1 benchmark
66.7 -> 32.4 s (2b held it flat); the fringe hoisting then took it to
22.9 s (~29%).

Fresh NVTX profiles (2026-07-21, RTX 2070 benchprof + tooarrana `multi`)
reframed the next step. Scoped to the correlation, the GPU is ~55-58%
busy / ~42-45% idle on both platforms. A per-kernel gap analysis then
**ruled CUDA graphs out** (recheck only if the clean profile below
contradicts it): launch overhead is just 3-4% of host CUDA-API time, and
~97% of the GPU idle is in big (>100us) gaps where the GPU waits on host
work / data delivery - not the small inter-kernel gaps graphs remove. So
the idle is **host / data-delivery bound** (the per-subint host tail plus
the procslot data-delivery pipeline), amplified on the desktop by
12-ranks-on-8-cores oversubscription and on the cluster `multi` job by
VDIF disk I/O - a separate, larger investigation, deferred until a clean
GPU-bound cluster profile (fake data, one rank per core, ~400 subints)
isolates the real gap. Input H2D is <1% of wall on both, so the
transfer-overlap idea (old work-queue item 1) is demoted to longer-term.
Meanwhile the live GPU-side levers cut GPU-busy time: fuse unpack+fringe
(work-queue item 1; unpack is the largest kernel on the cluster at 34.9%)
and reduced precision (FP16 / precision-drop, items 2-3; fringe FP64 is
24% of GPU time on the 2070 but only 12% on the fast-FP64 cluster card,
so precision work leans desktop).

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
- **Fringe-rotation interpolator hoisting** (2026-07-21): the per-window
  `a`/`b` and per-(window, band) `bigAval`/`bigB_reduced` are precomputed
  once per subint (`gpu_precompute_fringe_rotator`) instead of being
  recomputed in every (window, band, channel) thread of the rotation
  kernels — the per-sample kernel now only forms the phase and applies the
  rotator. A pure refactor of the same FP64 arithmetic (numerically
  equivalent; the GPU final output is not bit-reproducible run-to-run
  anyway, due to XMAC atomics). T5-T1 32.4 -> 22.9 s (~29%) — the
  per-sample FP64 recompute was the dominant GeForce cost (FP64 at 1/32
  rate). 8/8 device + fallback PASS, both pipeline modes. Design:
  gpu-fringerotation-design.md. Follow-up (per-sample precision drop) is a
  work-queue item below.

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
   (launch overhead is only ~3-4% of host time; see "Where we are").
   Interacts with the FP16 and precision-drop items below.
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
4. **perbandweights on GPU** — currently unused on the GPU path but
   should be active, in analogy with CPUMode/Mk5Mode's interlaced-VDIF
   handling (identified in the de-serialization design review). Shape
   the device weights kernel (de-serialization Increment 1) so
   per-(window, band) weights are a natural extension.
5. **LSB on GPU** — unblocks the lsb/lsb-complex/dsb scenarios.
6. **CODIF on GPU** — format-aware frame-validity hook (FIXME in
   `blanker_vdif_gpu`) + widen the `getMode` format gate. MUST add the
   `datalengthbytes <= getMaxDataBytes` clamp in `process_gpu` first:
   non-VDIF datastreams keep the base class's guard-scaled sendbytes,
   which can exceed the packed-data buffer (latent, unreachable today).
7. **Kernel launch-configuration / occupancy audit.** Verify every GPU
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
