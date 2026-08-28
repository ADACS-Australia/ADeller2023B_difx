# GPU correlator benchmark ledger

## Protocol

Metric: **T5 − T1** — wall time of a 5 s FakeData correlation minus wall
time of a 1 s correlation (same config, `tInt = 1 s`, 20 ms subints,
10 stations). Differencing cancels the large constant start-up/shutdown
cost and measures the steady-state correlation rate: the metric is the
cost of 4 s of data.

Run it with `./run-bench.sh [repeats]` (default 3; report best-of-N to
suppress scheduling noise on the oversubscribed desktop). `USEGPU=0`
benchmarks the CPU path; `EXTRA_ENV="DIFX_GPU_PIN_INPUT=0"` etc. for A/B
legs. The script prints a ready-to-paste ledger row.

**Since 2026-07-22 the benchmark defaults to single-thread VDIF** (the fake
data is rewritten `INTERLACEDVDIF→VDIF` after vex2difx) so it measures the GPU
correlator, not the DataStream interlaced-VDIF corner-turn (the demux that
dominated the pipeline - see the A100 section §10). Set `SINGLE_THREAD_VDIF=0`
to keep the interlaced demux. Rows dated on/before 2026-07-21 are all
interlaced and are NOT directly comparable to single-thread rows.

Rules:
- Machine otherwise idle (no builds, no nsys, no browser). **On the cluster
  that means `sbatch --cpus-per-task=5 <script>`** — a co-tenant job stalls the
  input copies and costs ~10% of wall with no other symptom (gpu-changes.md
  §14). tooarrana rejects `--exclusive`, so isolation comes from reserving most
  of the node: 5 × 12 tasks = 60 of gina's 64 cores. That does **not** start
  more mpifxcorr ranks (the rank count is `--ntasks`). **Keep `--nodes=1`** —
  without it SLURM satisfies a wide `--cpus-per-task` by splitting the ranks
  across two nodes, which reserves a GPU per node and pushes some
  DataStream→Core traffic onto the fabric instead of intra-node CMA (seen
  2026-08-26). Each run prints a `node ownership: N/M cores` verdict and warns
  if the allocation spanned more than one node; anything under 90%, or
  multi-node, does not belong in this file.
- Record every change that lands: benchmark at the commit that follows it.
- Note the VDIF threading (single-thread default vs `SINGLE_THREAD_VDIF=0`).
- FakeData output is NOT bit-reproducible run to run (see README) - this
  ledger tracks time only, never correctness.

## Ledger (RTX 2070 desktop, ar313)

| date | commit | mode | T1 (s) | T5 (s) | T5−T1 (s) | notes |
|---|---|---|---|---|---|---|
| 2026-07-18 | 785bcaec0 | gpu | 24.0 | 90.7 | 66.7 | first ledger row; Lever A (pinned input) in |
| 2026-07-18 | f39cd5bc3 | gpu | 22.5 | 84.9 | 62.4 | set_weights on device (de-serialization Incr 1) |
| 2026-07-20 | 57de0ae03 | gpu | 16.4 | 48.8 | 32.4 | baseline weights on device (de-serialization Incr 2) |
| 2026-07-21 | 6f9e0dcef | gpu | 16.3 | 48.7 | 32.4 | AC weights on device, interim D2H dropped (Incr 2b; flat, a cleanup) |
| 2026-07-21 | 80f6e291a | gpu | 12.7 | 35.6 | 22.9 | fringe-rotation interpolator hoisted out of the per-sample loop (~29%) |
| 2026-07-21 | 7b8e31104 | gpu | 12.8 | 35.6 | 22.8 | host-tail overlap (intra-subint half-split); flat on the 2070 |
| 2026-07-21 | 7b8e31104 | gpu DIFX_GPU_PIPELINE=0 | 12.7 | 35.3 | 22.6 | synchronous, no overlap; ≈ pipeline=1 ⇒ no idle to hide on the 2070 (GPU-compute-bound). The overlap targets the ~48% idle measured on the A100, so the win is expected on the cluster re-profile, not here. |
| 2026-07-22 | (unpack-drain fix) | gpu DIFX_GPU_PIPELINE=1 | 11.8 | 35.5 | 23.7 | RING-deep host staging + drop per-station `unpack_all` drain; `cudaStreamSynchronize` 4021→31 (13.1 s→0.002 s). Flat on the 2070 (compute-bound + oversubscribed ⇒ no idle to convert). A100 wall win TBC on the cluster. |
| 2026-07-22 | (unpack-drain fix) | gpu DIFX_GPU_PIPELINE=0 | 12.6 | 35.5 | 22.9 | same build, no overlap; ≈ pipeline=1 on the 2070. |
| 2026-07-22 | f3046d081 | gpu, single-thread VDIF | 9.1 | 23.1 | **14.0** | benchmark now single-thread by default (no DataStream corner-turn); ~40% faster than the interlaced row below - the corner-turn cost the oversubscribed desktop too, not just the A100 |
| 2026-07-22 | f3046d081 | gpu, INTERLACEDVDIF (SINGLE_THREAD_VDIF=0) | 12.2 | 35.4 | 23.2 | interlaced comparison at the same commit |
| 2026-07-23 | (fused unpack+fringe) | gpu, single-thread VDIF | 8.4 | 20.7 | **12.3** | fused unpack into fringe rotation (gpu-plan.md item 1): decode straight from packed data, no unpacked-buffer round-trip; removes the largest kernel. 14.0→12.3 (~12%) on the 2070; larger A100 gain expected (unpack was a bigger fraction there). Both T1 and T5 down. |
| 2026-07-23 | (fused sum_weights) | gpu, single-thread VDIF | 8.5 | 20.3 | **11.8** | fold the total-weight reduction into gpu_set_weights (per-window atomicAdd), delete the `<<<1,1>>>` gpu_sum_weights. 12.3→11.8 (~4%) on the 2070; on the A100 it should be ~its 7.7% GPU-busy share. |
| 2026-08-26 | (conjugate array removed) | gpu, single-thread VDIF | 7.0 | 17.8 | **10.8** | delete `conj_fftd_gpu` and conjugate in the multiply instead (gpu-changes.md §15). 11.8→10.8 (~8.5%); `gpu_resultsrotatorMultiply` 536.3→396.9 us/call (−26%) on the 2070, and 82 MB/mode of VRAM freed. Larger A100 gain expected — that kernel is 42-44% of its kernel busy. |
| 2026-08-27 | (tiled fused fringe) | gpu, single-thread VDIF | 6.5 | 16.6 | **10.1** | shared-memory transpose in the fused decode+fringe kernel so its band-major write coalesces (gpu-changes.md §16, design in gpu-fringetile-design.md). Same-session A/B against the same binary with `DIFX_GPU_FRINGE_TILE=0`: **10.9 -> 10.1 s (7.3%)**; the kernel itself 948 -> 703 us/call (1.35x) and it is 46.5% of 2070 kernel time. Shape sweep (`fringetile-sweep.sh`, 13 shapes from 1x4096 to 128x128 to 16x4096): **1.30-1.54x, no shape slower**, output bit-identical to the untiled path everywhere. A100 A/B still to run. |
| 2026-08-27 | (tiled fused fringe) | gpu DIFX_GPU_FRINGE_TILE=0 | 6.8 | 17.7 | 10.9 | the untiled path in the same binary and the same session - the A/B baseline for the row above, and it reproduces the 2026-08-26 10.8 s row, so there is no session drift in the comparison. |

## A100 kernel profile at `edad94b34` (landed 2026-08-28, job 15956983)

The profile that had been queued since 2026-08-26, finally through the queue.
`benchprof-profile-nsys-5s.sbatch`, gina6, `--cpus-per-task=5` (60/64 cores,
ledger-quality), 5 s window = 492 subints, single-thread VDIF. Capture and
analysis kept in `tests/benchprof-28082026/`. **This is the UNTILED build** -
the kernel is `gpu_fused_fringe`, not `_tiled` - so it is the A100 baseline for
the tiling work, not an A/B of it.

| quantity | value |
|---|---|
| kernel span | 5440 ms |
| kernel busy | 3924 ms (72.1% of span) |
| GPU busy (union with memcpy) | **4975 ms = 91.4%** |
| GPU idle | 466 ms = 8.6% |
| idle in gaps > 500 us | **0 ms (zero such gaps)** |
| idle in gaps < 20 us | 452 ms = 97% of idle |

| kernel | us/call | share | us/call before §15 |
|---|---|---|---|
| `gpu_resultsrotatorMultiply` | 277.9 | 34.9% | 396.5 (**−30%**) |
| `gpu_fuse_xmac_and_average` | 246.1 | 24.7% | 258 |
| `gpu_fused_fringe` | 191.9 | 24.1% | 192 (untouched) |
| `vector_fft` | 97.8 | 12.3% | 98 |

Two things follow.

**The conjugate-array deletion (§15) pays on the A100 as it did on the 2070:**
−30% on `gpu_resultsrotatorMultiply` there against −26% here, and every other
kernel is within a percent of its pre-§15 value, which is what makes the
attribution safe.

**The between-subints idle problem is finished.** The 2026-07-21 capture had
35.3% idle with 85.6% of it in gaps > 500 us - a host/data-delivery stall. This
one has 8.6% idle, **no gap over 500 us at all**, and 97% of what remains in
sub-20 us inter-kernel gaps. By the heuristic in `gpu-profiling.md` that is
launch-overhead bound, i.e. what is left for the host side is CUDA graphs, not
more overlap work.

**Still outstanding for the cost/benefit artifact:** this is a 5 s *profile*, not
a 20 s wall time, so the `gina4 (A100) 24.5 s` row below is still pre-§15 and
still conservative. Refreshing it needs one
`sbatch --cpus-per-task=5 benchprof-profile-nonsys-20s.sbatch` at the current
commit.

## A100 tiling A/B, 2026-08-28 (job 16004634) - INCONCLUSIVE, and why

`benchprof-fringetile-ab.sbatch` on gina3, **shared node (12/64 cores = 18%)**,
2 pairs, ~20 s window per leg. Do not quote the ratio: the run is recorded
because its failure mode is instructive.

| leg | script wall | correlation phase (difxlog) | outside the correlation |
|---|---|---|---|
| tiled rep1 (first in pair) | 25.5 s | 22 s | **3.5 s** |
| untiled rep1 | 23.3 s | 23 s | 0.3 s |
| tiled rep2 (first in pair) | 24.0 s | 22 s | **2.0 s** |
| untiled rep2 | 23.4 s | 22 s | 1.4 s |

The script reported **0.971x - "tiled is slower"** - and that verdict is an
artifact of two things, neither of which is the kernel:

1. **The A/B ordering was biased.** The script ran the legs 1,0,1,0, so the
   tiled leg was *always first in its pair* and always paid the first-of-pair
   cost (srun launch, GPU context init, Lustre metadata). That cost measured
   2-3.5 s of a ~24 s leg on a shared node - several times the ~5% effect being
   measured. Fixed: ABBA ordering plus a discarded warm-up leg.
2. **The correlation phase says the opposite.** Steady-state visibility dumps
   (20 per leg, from the difxlog) show tiled completing 19 intervals in 19 s in
   both reps, untiled taking 20 s in both - i.e. tiled ahead by exactly one
   second of drift, reproducibly. But difxlog timestamps are 1-second resolution,
   so that is ~5% +/- 5%.

So the two measurements in this run disagree within their uncertainties, and
neither has the resolution for a few-percent effect on an 18%-owned node.

**What to run instead**: the kernel-level sweep, which times the kernel directly,
is immune to node sharing and launch overhead, and takes ~2 minutes:
`srun --gres=gpu:1 --time=15 --mem=8000 ./fringetile-sweep.sh 2500 30`. A wall
A/B at this effect size needs the fixed script *and* `--cpus-per-task=5`.

`DIFX_GPU_FRINGE_TILE` stays **on** by default on the strength of the 2070
evidence (1.2-1.5x at every shape, T5-T1 7.3%); the A100 remains unresolved and
`=0` reverts it per-site with no rebuild.

## CPU baselines (added 2026-08-27, for the cost/benefit costing)

These exist so the GPU numbers can be compared against something, and because
the ar313 CPU leg is easy to misread. **On ar313 the CPU path is core-starved,
not compute-limited**: the correlator competes with ten DataStream ranks for four
physical cores, and halving the cores available cost only 1.41x rather than 2x.
So the desktop CPU row says what that machine does and must NOT be scaled up to
describe a CPU server - which is what the cluster rows are for.

| machine | config | metric | result |
|---|---|---|---|
| ar313 (i7-6700, 4 cores) | CPU only, 2 physical cores | T5-T1 | 134.5 s |
| ar313 | CPU only, 4 physical cores | T5-T1 | **95.5 s** |
| ar313 | + RTX 2070 SUPER | T5-T1 | **10.8 s** (8.8x) |
| dave13 (2 x EPYC 7543, 64 cores) | CPU only, 10 threads | wall, 20 s window | 96.6 s |
| dave13 | CPU only, 20 threads | wall, 20 s window | 51.7 s (1.87x, 93% eff) |
| dave37 | CPU only, 40 threads | wall, 20 s window | **32.4 s** (1.60x, 80% eff) |
| gina4 (A100) | 12 cores + one A100 | wall, 20 s window | **24.5 s** |

Run with `benchprof-cpu-20s.sbatch` (`--export=ALL,NTHREAD=n`) and, for the GPU
row, `benchprof-profile-nonsys-20s.sbatch`. Note the two different metrics: the
desktop rows are T5-T1 (start-up differenced away), the cluster rows are raw wall
for a 20 s window (start-up included, so their steady-state rates are better than
shown). To compare across the two, ar313's fitted line gives 481 s (CPU) and
58.3 s (GPU) for a 20 s window.

**The 20->40 thread efficiency drop from 93% to 80% is a configuration artefact,
not a scaling limit:** each processor has 32 cores, so one Core process with 40
threads spans both sockets. Two processes of 20 threads each, bound one per
socket, would do better - so the 40-thread row is a worst case for a big CPU box.

**Derived equivalences** (interpolating the cluster CPU curve): the used
RTX 2070 SUPER is worth ~18 EPYC 7543 cores; one A100 is worth 50-60, i.e. most
of a dual-socket node, while occupying 12 cores rather than 40. The full
cost/benefit write-up, including August 2026 hardware prices and a VGOS-rate
sizing exercise, is the artifact recorded in `docs/gpu-payback.pdf`.

## A100 cluster profiling (benchprof-profile-nsys-5s.sbatch, 10 stations)

Note on windows: subints are 10 ms, so the profiling window scales linearly
(4 s = 400 subints/station, 5 s = 500, 20 s = 2000). Rows here span a mix of
4 s and (2026-07-23) 5 s windows, so absolute ms are NOT directly comparable
across rows - the **per-subint** figures in the fused-row note below, and idle
%, are the window-independent quantities. All default to single-thread VDIF. A
~20 s capture is NOT usable under nsys 2022.2.1 (its injection library
segfaults ~13 s in - see gpu-profiling.md); the 20 s soak run is nsys-free
(benchprof-profile-nonsys-20s.sbatch).

Clean GPU-bound profile: A100-SXM4-80GB, one rank/core (isolation via
`--cpus-per-task`; see the Rules above),
fake data, Core rank wrapped with nsys (`.nsys-rep` + sqlite kept under
the run dir). "GPU span" = first-kernel to last-kernel; "busy (kernels)"
= sum of kernel durations (single compute stream, so serial); "busy
(+copy)" additionally counts the H2D/D2H copies that share the compute
stream. This is a diagnostic ledger (idle %), not the T5-T1 metric.

| date | commit | GPU span (ms) | busy kernels (ms) | idle kernels | busy +copy | idle +copy | notes |
|---|---|---|---|---|---|---|---|
| 2026-07-21 | pre-overlap (80f6e291a) | 10497 | 5471 | 47.9% | — | — | reference (tooarrana clean profile, per gpu-plan notes) |
| 2026-07-21 | 7b8e31104 | 8798 | 5088 | 42.2% | 5695 | 35.3% | host-tail overlap; idle did NOT collapse |
| 2026-07-22 | unpack-drain fix, INTERLACED | 9858 | 5077 | 48.5% | 5731 | 41.9% | sync fix held (no `cudaStreamSynchronize` storm) but GPU still starved by the DataStream corner-turn — see §10 |
| 2026-07-22 | + single-thread VDIF (genheaders), 4 s | 6050 | 5058 | 16.4% | 5614 | **7.2%** | corner-turn removed; kernel busy unchanged (5058≈5077) so the win is pure idle removal; wall (CUDA-API span) 12169 → 7886 ms (**~35%**); GPU now compute-bound |
| 2026-07-23 | fused unpack+fringe (619c273be), 5 s | 7032 | 4863 | 30.9% | 6337 | 9.9% | fused kernel replaces unpack+fringe; kernel-only idle rises (faster kernels ⇒ fixed tail is a bigger fraction) but the H2D copies now fill it (idle+copy 9.9%). See the per-subint comparison below. |
| 2026-07-23 | fused sum_weights (d503b7708), 5 s | 5800 | 4500 | 22.4% | 5208 | 10.2% | `gpu_sum_weights` gone from the mix; kernel busy −7.4%, exactly its old share. Same 5 s window as the row above ⇒ directly comparable. Wall (CUDA-API span) 9064 → 7710 ms, but see the H2D caveat below — only the kernel-busy part is attributable. |
| 2026-08-26 | placement report (56eb3b4c3), 5 s | 5852 | 4545 | 22.3% | 5159 | 11.9% | SHARED node (1 core/task) — kept because it is the run that **disproved** the NUMA hypothesis: definitively mis-placed (GPU node 1, rank node 2) yet the fastest H2D of any capture, 13.01 GB / 588.6 ms = **22.1 GB/s**. Kernel busy matches the row above to <1%, as expected for a logging-only change. See gpu-changes.md §14. |

**Weight-kernel elimination, measured against the row above** (both captures
are 5 s / ~495 subints per station, so the totals compare directly). Capture +
analyser output: `tests/23072026-weightkerneliminated/benchprof/nsys/`.

| kernel | fused unpack+fringe | + fused sum_weights | Δ |
|---|---|---|---|
| `gpu_sum_weights` `<<<1,1>>>` | 373.2 ms (7.7%) | **absent** | −373.2 ms |
| `gpu_set_weights` | 22.5 ms (0.5%) | 40.3 ms (0.9%) | +17.8 ms (the atomicAdd) |
| **net kernel busy** | 4862.5 ms | 4500.5 ms | **−362 ms (−7.4%)** |

Every other kernel is stable to <0.5% across the two captures
(`gpu_resultsrotatorMultiply` 1894.9 → 1894.7 ms, `gpu_fused_fringe` 948.6 →
946.8, `vector_fft` 484.0 → 482.0), which is the cross-check that the two runs
are comparable and that nothing but the weight path moved.

**⚠ H2D caveat — do not read the whole wall gain as ours.** The same volume of
input moved far faster in the second capture: **12.90 GB / 1444.7 ms (8.9 GB/s)
→ 12.88 GB / 683.0 ms (18.9 GB/s)**, at an unchanged copy count (~29.7k). No
part of this change touches the transfer path. The first of those runs was
simply **contended** — it shared its node with another job, which stalls the
input copies (gpu-changes.md §14 has the per-copy distributions that show it).
Of the ~15% CUDA-API-span improvement, only the ~7.4% kernel-busy part is
attributable to the code. **This is why ledger runs must reserve most of the
node** (`sbatch --cpus-per-task=5 …`; tooarrana rejects `--exclusive`), and why
the two 2026-07-23 rows are comparable on kernel busy but not on wall.

New A100 kernel mix (% of kernel busy, 4500.5 ms): `gpu_resultsrotatorMultiply`
**42.1%** (the #1 target — memory-bound, the FP16 candidate),
`gpu_fuse_xmac_and_average` 22.7%, `gpu_fused_fringe` 21.0%, `vector_fft`
10.7%, `gpu_baseline_weights` 1.4%, `gpu_set_weights` 0.9%, precompute 0.7%,
`gpu_blank_frames` 0.6%. Busy-union (kernels+copies) is **89.8%** of the span,
and 74% of the residual idle is now in sub-20 µs inter-kernel gaps — launch
overhead, not host starvation. H2D at 683 ms is 11.8% of the span (the
remaining wall floor under the FP16 work).

**Fused unpack+fringe, per-subint (window-independent) vs the pre-fusion 4 s
profile above** (normalising each kernel's total by its launch count, since the
two captures cover 400 vs 500 subints/station):

- unpack + fringe (the fused portion): **0.509 → 0.197 ms/subint (−61%)**
- total GPU kernel busy: **1.28 → 0.98 ms/subint (−23%)**
- wall (CUDA-API span ÷ launches): **2.00 → 1.83 ms/subint (−8%)**

The GPU-busy win is large; wall gains less because the run is no longer
GPU-busy-bound. New A100 kernel mix (% of kernel busy): `gpu_resultsrotatorMultiply`
**39%** (now the #1 target - memory-bound, an FP16 candidate), `gpu_fuse_xmac_and_average`
21%, `gpu_fused_fringe` 19.5%, `vector_fft` 10%, **`gpu_sum_weights` `<<<1,1>>>` 7.7%**
(a single-thread reduction), weights/precompute/blank <2% each. H2D is now
~12.9 GB / 1444 ms on the compute stream (~20% of span) - a new wall floor the
fusion doesn't touch. **Follow-up: `gpu_sum_weights` deleted** - its
sum folded into `gpu_set_weights` (per-window atomicAdd), so that 7.7% should be
gone from the next A100 profile (2070 T5-T1 12.3→11.8).

The overlap shortened the GPU span ~16% (10.5 -> 8.8 s) and trimmed idle
~48% -> ~42% (kernels-only), but did **not** collapse the between-subints
gap. Root cause of the residual idle (2657 ms, 85.6% of it in >500 us
gaps): the process thread issues ~10 `cudaStreamSynchronize` per subint
(5.58 s total) - one per station, matching cuFFT's per-`cufftExecC2C`
driver footprint (`cuStreamIsCapturing`/`cuLaunchKernel`/`cudaStream-
Synchronize` all ~3948). cuFFT is synchronising the compute stream on
every FFT exec, serialising host and GPU at station granularity. The
host tail the overlap moved (~900 us/subint) was never the bottleneck.
See gpu-plan.md work queue.

**CORRECTION (2026-07-22): the sync is NOT cuFFT.** The
`cudaStreamSynchronize` count matched `cufftExecC2C` only because unpack
and the FFT sit in the same per-station tofft iteration. cuFFT is async
(the fftbench probe returns ~0 ms after a 50 ms backlog on this A100);
stubbing the FFT left the sync count unchanged; an nsys sync-backtrace
resolved the caller to `Mk5_GPUMode::unpack_all` -> `valid_frames->sync()`.
Removed on the device path (gated to the host-weights fallback) with
RING-deep host staging to keep the tail-overlap safe - see gpu-changes.md
§9 and gpu-plan.md work-queue item 0.

**Resolved (2026-07-22, §10 / last two table rows):** the drain removal
alone left the A100 idle at ~42% - the true limiter was the DataStream
INTERLACEDVDIF corner-turn, not any Core-side sync. Rebuilding fake data
as single-thread VDIF (mark5access `genheaders`, `191528ff9`) removed the
corner-turn and the idle collapsed **41.9% → 7.2%**, wall 12.2 → 7.9 s
(~35%). Kernel busy is unchanged (5058 ≈ 5077 ms), so the speed-up is
purely the GPU no longer starving. The residual 7.2% idle is now mostly
small (<20 us) inter-kernel gaps; the GPU is compute-bound, with
`gpu_resultsrotatorMultiply` (28.9%), `gpu_unpack` (24.5%) and the
`<<<1,1>>>` `gpu_sum_weights` (5.8%) the next kernel targets.

## Pre-protocol reference points (15 s of data, tInt 2 s, whole-run wall)

Measured 2026-07-18 during Lever A work; not comparable to the ledger
rows above, kept for the record:

| build | wall (15 s data) |
|---|---|
| pre-Lever-A (1bf496b16+build fixes) | 268.0 s |
| Lever A pinned (785bcaec0) | 258.9 s |
| Lever A, staging fallback (`DIFX_GPU_PIN_INPUT=0`) | 269.3 s |
