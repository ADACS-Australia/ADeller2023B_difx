# GPU correlator: the plan

Living plan for the GPU work on `adam-performance-gains`. Companion to
`gpu-changes.md` (what has already happened) and
`tests/FakeData/BENCHMARKS.md` (the numbers). Updated as items land.

## Where we are

**Correctness: done** for the USB scenarios (usb, usb-complex, complex-complex
and the 5-station/4-subband `multi` PASS CPU-vs-GPU in both `DIFX_GPU_PIPELINE`
modes; LSB deliberately unimplemented). We are in the **performance** phase:
T5−T1 is at **10.1 s**, down from 66.7 s. Every change and why is in
`gpu-changes.md` (settled sections in `gpu-changes-archive.md`); the numbers are
in `tests/FakeData/BENCHMARKS.md`. The queue below is only what remains.

**Current kernel mix.** The two cards disagree, and that has repeatedly decided
what is worth doing — always check which one a claim came from:

| kernel | RTX 2070 (tiled build) | A100 (untiled, `edad94b34`) |
|---|---|---|
| `gpu_fused_fringe(_tiled)` | **39.7%** (677 µs) | 24.1% (192 µs) |
| `vector_fft` (cuFFT) | 25.1% | 12.3% |
| `gpu_resultsrotatorMultiply` | 23.3% (397 µs) | **34.9%** (278 µs) |
| `gpu_fuse_xmac_and_average` | 8.0% | 24.7% |
| kernel busy, 5 s window | 1360 ms | 3924 ms |

The A100 capture predates the tiling, which is worth 1.48x on `gpu_fused_fringe`
there, so its share is now ~17-18% and the rotator is the largest kernel on both
cards. **GPU busy-union on the A100 is 91.4%** with no idle gap over 500 µs and
97% of the remaining idle in sub-20 µs gaps: the host/data-delivery stall that
dominated the 2026-07 captures is gone, and what is left on the host side is
launch overhead (CUDA graphs). FP64 is cheap on the A100, so precision work leans
on the 2070.

**What actually limits `gpu_resultsrotatorMultiply`** — measured, not assumed
(`tests/frac-probes/RESULTS.md`, four env-gated probes on the A100): atomics
**24%**, cross-pol traffic **25%**, band traffic + compute ~50%. Two results to
carry forward: the answer does **not** transfer between cards (atomics are 24% on
the A100 and 0% on the 2070, where the kernel is FP64-pipe bound), and giving
every (window, band, channel) its own thread is **2.3-2.4× slower** on both cards
— do not move bands out of the in-thread loop.

**Two benchmarking rules**, each of which cost a wrong conclusion once (§10, §14):
- Benchmark with **single-thread VDIF**; the interlaced corner-turn is CPU-bound
  and starves the GPU (Longer-term tracks fixing it for production).
- Give cluster runs **most of the node** for any *absolute* number:
  `sbatch --cpus-per-task=5` (60 of gina's 64 cores; tooarrana rejects
  `--exclusive`), and keep `--nodes=1`. Each run prints a `node ownership:`
  verdict.

And one added 2026-08-28: **if the effect is a few percent of one kernel, measure
the kernel, not a correlation.** A shared-node wall-clock A/B of the tiling
returned the wrong sign; the 2-minute kernel sweep settled it. See
`gpu-profiling.md`, "Measurement pitfall".

**PENDING:** a 20 s wall run at the current commit
(`sbatch --cpus-per-task=5 benchprof-profile-nonsys-20s.sbatch`, queued
2026-08-28) — the `gina4 (A100) 24.5 s` figure behind `gpu-payback.pdf` and its
artifact is still pre-§15 and conservative. That is the only open measurement.

**Agreed order from here** (2026-08-27; design in `gpu-autocorr-design.md`):

1. **Autocorrelations into Core/XMAC.** Implementation plan and increments are in
   `gpu-autocorr-design.md`; **increment 0 (a dual-pol synthetic scenario, so the
   cross-pol autocorrelations are tested at all) landed 2026-08-28**, and
   increment 1 (per-(baseline, pol) output offsets) is next.
   Removes the atomics entirely rather than
   reducing them, takes the cross-pol traffic with them, and deletes a whole
   host-tail data path (device→host copy, `vectorCopy_cf32` mirror,
   `averageFrequency`, the `vectorAdd_cf32_I` loop and `autocorrcopylock`).
   Invasive: `Mode`, `Core`, `Configuration`, the results-buffer layout, and it
   must keep the CPU path's Mode-based autocorrelations working alongside — the
   CPU/GPU divergence class that has caused the most bugs here.
2. **FP16 spectra**, in three stages, because stage (c) cannot be skipped:
   (a) the standalone `fftbench` accuracy/speed probe — no correlator changes;
   (b) the implementation, opt-in and gated;
   (c) **the science-level accuracy gate — a PREREQUISITE, not a follow-up.**
   FP16 changes results well beyond FP rounding, so `diffDiFX` cannot validate it
   at any threshold. Adam asked for fringe-fitting S/N and image S/N tests
   (2026-08-26); until they exist FP16 can be built and measured but not landed.
   Worth starting (c) alongside (a).

**Superseded:** the window-group reduction was built, measured as a net loss on
the 2070, and its branch deleted (2026-08-27), since item 1 removes the atomics
outright. Approach and measurements are preserved in `gpu-autocorr-design.md` —
do not re-plan it from scratch.

**Still the real production bottleneck: the DataStream corner-turn.** At the
measured ~1.7 Gbps per core, a VGOS-rate 4 × 16 Gbps observation needs ~38 cores
doing nothing but demultiplexing interlaced VDIF — more silicon than the
correlation itself. Every benchmark here bypasses it with single-thread VDIF,
which is right for measuring the correlator and misleading for specifying
hardware. Design in Longer-term; on current evidence it gates real-world
usefulness more than any remaining GPU-side optimisation.

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
6. **Kernel launch-configuration / occupancy audit — DONE** (2026-07-23 and
   2026-08-27). The `<<<1,1>>>` `gpu_sum_weights` reduction was folded away
   (gpu-changes.md §13), and `ncu` then found `gpu_fused_fringe` latency-bound on
   an uncoalesced write with one resident block per SM — fixed by the tiled
   kernel: 1.35x on the 2070, 1.48x-2.2x on the A100, 7.3% of T5−T1
   (§16, `gpu-fringetile-design.md`). Remaining kernels look sanely shaped; ncu
   is now the cheap way to check any new one (`tests/FakeData/ncu-frac.sh`).

7. **STA (Short Term Accumulate) dumps on the GPU path** — the fast-transient /
   monitoring multicast reads `Mode::autocorrelations`, which the
   autocorrelation-into-XMAC work removes on the GPU path. Decision (2026-08-28):
   refuse the dump explicitly rather than send stale spectra;
   `DIFX_GPU_XMAC_AUTOCORR=0` restores both the Mode path and STA. Reimplementing
   it from the device results region is possible and not planned. See
   `gpu-autocorr-design.md`.

8. **GPU pcal regression coverage** — still no CPU-vs-GPU test: `phaseCalInt=0`
   in every synthetic/FakeData scenario, so `diffDiFX` never sees the phase-cal
   path. Partially mitigated 2026-08-28: `fringetile-sweep.sh SWEEP_PCAL=1`
   exercises the `DOPCAL` kernels and compares their bins across the tiled and
   untiled paths (agreement 3e-07), which catches indexing errors but says
   nothing about agreement with the CPU. The real fix is still a `phaseCalInt>0`
   synthetic scenario in `run-local.sh`.

9. **NUMA/affinity audit — CLOSED** 2026-08-26: the 2× swing in input-transfer
   speed was node contention, not placement (every placement reached the same
   26 GB/s). Fixed by giving ledger runs most of the node; `GPUCore` now logs its
   CPU/NUMA placement. Full analysis: gpu-changes.md §14. Reopen only if a
   profile shows a placement effect on an isolated node.

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
