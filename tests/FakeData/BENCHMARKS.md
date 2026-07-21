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

Rules:
- Machine otherwise idle (no builds, no nsys, no browser).
- Record every change that lands: benchmark at the commit that follows it.
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

## A100 cluster profiling (benchprof-profile.sbatch, 400 subints, 10 stations)

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

## Pre-protocol reference points (15 s of data, tInt 2 s, whole-run wall)

Measured 2026-07-18 during Lever A work; not comparable to the ledger
rows above, kept for the record:

| build | wall (15 s data) |
|---|---|
| pre-Lever-A (1bf496b16+build fixes) | 268.0 s |
| Lever A pinned (785bcaec0) | 258.9 s |
| Lever A, staging fallback (`DIFX_GPU_PIN_INPUT=0`) | 269.3 s |
