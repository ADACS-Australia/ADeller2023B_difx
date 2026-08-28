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

**Current A100 kernel mix** (5 s profile at `edad94b34`, 3924 ms kernel busy,
landed 2026-08-28 - job 15956983, capture in `tests/benchprof-28082026/`):
`gpu_resultsrotatorMultiply` **34.9%** (277.9 µs/call), `gpu_fuse_xmac_and_average`
24.7%, `gpu_fused_fringe` 24.1% (191.9 µs/call), `vector_fft` 12.3%, everything
else <2%. GPU busy-union is **91.4%** of span; the residual 8.6% idle has **no
gap over 500 µs at all** and 97% of it in sub-20 µs inter-kernel gaps, so the
host/data-delivery stall that dominated the 2026-07 captures is gone and what is
left on the host side is launch overhead (CUDA graphs). FP64 is cheap on the
A100, so precision work leans on the 2070. Note this capture is the **untiled**
build, so it is the baseline for the tiling A/B, not a measurement of it.

**What actually limits `gpu_resultsrotatorMultiply`** — measured, not assumed
(`tests/frac-probes/RESULTS.md`, four env-gated timing probes on the A100):
atomics **24%**, cross-pol traffic **25%**, band traffic + compute ~50%. Two
results worth carrying forward: the answer does **not** transfer between cards
(atomics are 24% on the A100 and 0% on the bandwidth-saturated 2070), and
giving every (window, band, channel) its own thread is **2.3-2.4× SLOWER** on
both cards — do not move bands out of the in-thread loop. `docs/gpu-autocorr-
design.md` holds the resulting three-step plan; §15 was step one.

**Two benchmarking rules** (each cost a wrong conclusion once — §10, §14):
- Benchmark with **single-thread VDIF**. The DataStream interlaced-VDIF
  corner-turn is CPU-bound and starves the GPU, capping what any GPU benchmark
  can show. The Longer-term item below tracks fixing it for production.
- Give cluster runs **most of the node** for any number that reaches the
  ledger: `sbatch --cpus-per-task=5 …` (60 of gina's 64 cores). tooarrana
  rejects `--exclusive`, and widening `--cpus-per-task` does *not* start extra
  mpifxcorr ranks — that is `--ntasks`. Keep `--nodes=1`, or SLURM splits the
  ranks over two nodes and the data path changes under you. A co-tenant job
  stalls the input copies and costs ~10% of wall, invisibly. Each run prints a
  `node ownership:` verdict.

**PENDING (narrowed 2026-08-28):** the queued A100 profile of `edad94b34`
landed and is written up in `BENCHMARKS.md` - it confirms §15 is worth −30% on
`gpu_resultsrotatorMultiply` there. What it does *not* give is a wall time: it is
a 5 s nsys profile, so the `gina4 (A100) 24.5 s` figure behind `gpu-payback.pdf`
and its artifact is still pre-§15 and still conservative. Refreshing that needs
one `sbatch --cpus-per-task=5 benchprof-profile-nonsys-20s.sbatch` at the current
commit, and the tiling A/B (`benchprof-fringetile-ab.sbatch`) is still to run.

**Agreed order from here** (2026-08-27; design in `gpu-autocorr-design.md`):

1. **Autocorrelations into Core/XMAC.** Removes the atomics entirely rather than
   reducing them, takes the cross-pol traffic with them, and deletes a whole
   host-tail data path (device→host copy, `vectorCopy_cf32` mirror,
   `averageFrequency`, the `vectorAdd_cf32_I` loop and `autocorrcopylock`).
   ~17% of A100 kernel busy plus the tail. Invasive: `Mode`, `Core`,
   `Configuration`, the results-buffer layout, and it must keep the CPU path's
   Mode-based autocorrelations working alongside — the CPU/GPU divergence class
   that has caused the most bugs here.
2. **FP16 spectra**, in three stages, because stage (c) cannot be skipped:
   (a) the standalone `fftbench` accuracy/speed probe — no correlator changes;
   (b) the implementation, opt-in and gated;
   (c) **the science-level accuracy gate — a PREREQUISITE, not a follow-up.**
   FP16 changes results well beyond FP rounding, so `diffDiFX` cannot validate
   it at any threshold. Adam asked for fringe-fitting S/N and image S/N tests
   (2026-08-26); until they exist FP16 can be built and measured but not
   landed. Worth starting (c) in parallel with (a).

**Superseded:** the window-group reduction was built, measured as a net loss on
the 2070, and its branch **deleted** (2026-08-27) once item 1 was agreed, since
item 1 removes the atomics entirely rather than reducing them. The approach and
every measurement are written up in `gpu-autocorr-design.md` — enough to rebuild
it from the note if an A100 result ever justifies revisiting. Do not re-plan it
from scratch.

**Still the real production bottleneck: the DataStream corner-turn.** The
2026-08-27 costing exercise made this concrete rather than theoretical: at the
measured ~1.7 Gbps per core, a VGOS-rate 4 × 16 Gbps observation needs ~38 cores
doing nothing but demultiplexing interlaced VDIF — more silicon than the
correlation itself. Every GPU benchmark in this project deliberately bypasses it
with single-thread VDIF, which is correct for measuring the correlator and
misleading for specifying hardware. The frame-reorder + GPU de-interleave design
is in Longer-term; on current evidence it gates real-world usefulness more than
any remaining GPU-side optimisation.

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
   `gpu_unpack` was deleted in the fusion.) **DONE 2026-08-27:** `ncu` (now
   usable on ar313) found `gpu_fused_fringe` - 46.5% of 2070 kernel time, and
   never examined because it is only 21% on the A100 - latency-bound on an
   uncoalesced band-major write, with a 1024-thread block that left one
   resident block per SM. Fixed with a shared-memory transpose and a
   256-thread block: 1.35x on the kernel, 7.3% on T5-T1, 1.30-1.54x across 13
   (bands, channels) shapes, gated by `DIFX_GPU_FRINGE_TILE`. See
   gpu-changes.md §16 and gpu-fringetile-design.md; the A100 A/B is the open
   half. **DONE so far (2026-07-23):**
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
   cross-socket, reached the same 26 GB/s per-copy peak. Fixed by giving
   ledger benchmarks most of the node (`--cpus-per-task=5`; tooarrana rejects
   `--exclusive`); `GPUCore` now logs its CPU/NUMA placement in every difxlog
   so transfer numbers can be attributed after the fact. Reopen only if a
   profile shows a placement effect that survives an isolated node.

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
