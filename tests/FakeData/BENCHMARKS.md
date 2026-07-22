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
- Machine otherwise idle (no builds, no nsys, no browser).
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

## A100 cluster profiling (benchprof-profile.sbatch, 10 stations)

Note: the rows below are the original ~4 s / 400-subint window; since
2026-07-22 the sbatch profiles a ~20 s / ~2000-subint window (spans scale
accordingly, idle % is the comparable quantity) and defaults to single-thread
VDIF.

Clean GPU-bound profile: A100-SXM4-80GB, one rank/core `--exclusive`,
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
| 2026-07-22 | + single-thread VDIF (genheaders) | 6050 | 5058 | 16.4% | 5614 | **7.2%** | corner-turn removed; kernel busy unchanged (5058≈5077) so the win is pure idle removal; wall (CUDA-API span) 12169 → 7886 ms (**~35%**); GPU now compute-bound |

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
