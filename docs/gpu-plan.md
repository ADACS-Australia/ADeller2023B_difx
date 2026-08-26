# GPU correlator: the plan

Living plan for the GPU work on `adam-performance-gains`. Companion to
`gpu-changes.md` (what has already happened) and
`tests/FakeData/BENCHMARKS.md` (the numbers). Updated as items land.

## Where we are

**Correctness: done** for the USB scenarios (usb, usb-complex,
complex-complex, and the 5-station/4-subband `multi` PASS CPU-vs-GPU in both
`DIFX_GPU_PIPELINE` modes; LSB deliberately unimplemented). We are in the
**performance** phase: T5−T1 is at **11.8 s**, down from 66.7 s. What landed
to get there, and why, is in `gpu-changes.md` (§5-14, one section per change);
the numbers are in `BENCHMARKS.md`. The queue below is only what remains.

**Current A100 kernel mix** (5 s profile, 4500 ms kernel busy):
`gpu_resultsrotatorMultiply` **42%**, `gpu_fuse_xmac_and_average` 23%,
`gpu_fused_fringe` 21%, `vector_fft` 11%, everything else <2%. GPU busy-union
is ~90% of span; residual idle is mostly sub-20 µs inter-kernel gaps. FP64 is
cheap on the A100, so precision work leans on the 2070.

**Two benchmarking rules** (each cost a wrong conclusion once — §10, §14):
- Benchmark with **single-thread VDIF**. The DataStream interlaced-VDIF
  corner-turn is CPU-bound and starves the GPU, capping what any GPU benchmark
  can show. The Longer-term item below tracks fixing it for production.
- Submit cluster runs **`sbatch --exclusive`** for any number that reaches the
  ledger. A co-tenant job stalls the input copies and costs ~10% of wall,
  invisibly. Each run now prints its own `node ownership:` verdict.

**Next GPU-busy target:** the **FP16 / precision-drop items (1, 2)**, aimed at
`gpu_resultsrotatorMultiply` — 42% of GPU busy and memory-bound, so halving its
traffic helps directly. Then the rest of the occupancy audit (item 6). (The
DataStream corner-turn remains the real *production* bottleneck — see
Longer-term.)

## Work queue (underway + future)

1. **Optional FP16 unpack + fringe rotation** (with 16-bit -> 32-bit
   FFT): an opt-in reduced-precision path that carries the unpacked
   samples and the fringe rotation in FP16, with the FFT taking FP16 in
   and producing FP32 output. Roughly halves the memory traffic for those
   stages (and the compute on cards with fast FP16). Needs an accuracy
   assessment against the correlator's dynamic range; gated so the
   full-precision path stays the default. Combines naturally with the
   completed unpack+fringe fusion (gpu-changes.md §12 — the fused decode is
   its natural seam), and subsumes the per-sample precision drop (item 2) -
   if FP16 lands, item 2 is moot.
2. **Fringe-rotation per-sample precision drop** (follow-up to the
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
6. **Kernel launch-configuration / occupancy audit.** Verify every GPU
   kernel launches an appropriate grid/block size - enough threads to
   fill the device, a block shape with good occupancy, and no accidental
   under- or over-subscription. Several kernels were written with ad-hoc
   launch dims (one thread per (window, band, channel) in the fused
   decode+fringe kernel; one thread per accumulator in the weight
   reductions); check each with nsys/`ncu` and fix any that are
   badly sized. Cheap, and a natural companion to the fused-decode and
   precision work (items 1-2). (The old unpack-layout question is moot -
   `gpu_unpack` was deleted in the fusion.) **DONE so far (2026-07-23):**
   the `<<<1,1>>>` `gpu_sum_weights` single-thread reduction (7.7% of A100
   GPU busy) was eliminated by folding its sum into `gpu_set_weights`
   (per-window atomicAdd) - see gpu-changes.md §13.
7. **GPU pcal regression coverage** — the GPU phase-cal path is untested by
   diffDiFX (phaseCalInt=0 in every synthetic/FakeData scenario), and the
   pcal-fused `DOPCAL` path added in the unpack+fringe fusion is validated by
   construction/review only. Add a phaseCalInt>0 synthetic scenario to
   run-local.sh so CPU-vs-GPU covers pcal extraction.
8. **NUMA/affinity audit — CLOSED 2026-08-26** (gpu-changes.md §14). The 2×
   run-to-run swing in input-transfer speed turned out to be **node
   contention, not placement**: every measured placement, GPU-local through
   cross-socket, reached the same 26 GB/s per-copy peak. Fixed by running
   ledger benchmarks `--exclusive`; `GPUCore` now logs its CPU/NUMA placement
   in every difxlog so transfer numbers can be attributed after the fact.
   Reopen only if a profile shows a placement effect that survives an
   exclusive node.

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
  fake-data benchmarking, the `benchprof-profile-*.sbatch` scripts already
  rewrite the `.input` `INTERLACEDVDIF/… → VDIF` to bypass the demux.)
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
- Profile the fftsPerChunk XMAC grid split if atomics show up.
