# GPU correlator: prose record of changes

This is the running, human-readable account of the CUDA/GPU work on this
fork (`ADACS-Australia/ADeller2023B_difx`, branch `adam-performance-gains`),
kept alongside the code so the story survives the commit log. Newest work
is at the bottom; extend this file with every substantive change.

The cast: `GPUCore` (`gpucore.cu`, a `Core` that owns one GPU) drives
per-datastream `GPUMode` objects (`gpumode.cu`, unpacking in
`gpudecode.cu`, format glue in `mk5mode_gpu.cu`) which must produce
output matching the CPU path (`core.cpp` / `cpumode.cpp`). Correctness is
policed by `tests/Synthetic` (CPU-vs-GPU diff of every scenario);
performance by `tests/FakeData` (see `BENCHMARKS.md` there).

## 1. Foundation

The GPU path was hand-merged from earlier prototype work and extended
with a GPU unpack testing framework, constructor-time memory allocation,
parallelised unpacking, and fixes from testing against real data. Data
weights were moved to CPU calculation with the GPU decoder tidied around
them. Complex-sampled stations and interlaced VDIF became supported
inputs.

## 2. Pulse cal (PCal) on the GPU

PCal extraction was implemented on the GPU in stages: real USB first,
then LSB, then complex sampling, with the accumulation logic optimised
and a long tail of correctness fixes — the milestone commit
(`f3c0dbd9f`) fixed a decode bug that made CPU and GPU results diverge
and got PCal extracting correctly. GPU pcal currently has no regression
coverage (diffDiFX compares only visibilities); a CPU+GPU pcal refactor
(PCal gaining a first-class "ingest folded bins" interface) is agreed
but not started.

## 3. The correctness campaign

A sustained CPU-vs-GPU parity effort, driven by the synthetic test
scenarios (deliberately including an end-of-recording boundary mid-job).
The recurring bug class was frame validity and windows that straddle
missing/invalid data. Landed fixes include:

- treating undelivered frames as invalid and zeroing the stale unpacked
  tail (`a1c6d2192`);
- rejecting frames with the VDIF invalid bit, and frames with zero Data
  Frame Length (`b3ebfd0f1`, `fbaf9c4e0`);
- resetting weights/validity when a subint arrives with no data
  (`e883f6bcc`), and restoring the autocorrelation flush into the
  results buffer (`46c48622d`);
- fused-XMAC correctness and occupancy fixes (`1713a5aa5`), and
  per-stream FFT buffer strides in the fused XMAC kernel (`d30ce1e9c`) —
  the identical-datastream half of that problem is fixed, heterogeneous
  datastreams remain unsupported by the XMAC plans;
- CPU-side bugs found along the way and hand-ported upstream (branch
  `bugfix-complex` off dev): the complex samplegranularity erasure
  (`ade81edb0`), the mark5access extra-blanker bug (`8c673cac7`), a
  generateVDIF error check; plus the fork-specific configuration.cpp
  double-halving fix (`695f8b08f`).

Outcome (2026-07-09, cluster, CUDA 12.8): scenarios usb,
complex-complex and usb-complex PASS bit-identical CPU-vs-GPU, in both
`DIFX_GPU_PIPELINE` modes. (On the desktop's newer toolchain the
acceptance bar is FP-level agreement at ~1e-7, judged by diffDiFX
thresholds — bit-identity is toolchain-dependent, not a design goal.)
LSB and DSB scenarios are excluded by design: `GPUMode` rejects lower
sideband until LSB is implemented.

## 4. Debug and test infrastructure

Built in parallel with the campaign and still in the tree (env-gated):

- `DIFX_WEIGHT_DEBUG` — identical per-window weight lines from both
  paths, plus HDRDEBUG frame-class transition dumps (`ee26c7820`,
  `fbaf9c4e0`);
- `DIFX_SPEC_DEBUG` — format-identical spectral tracing every 128th
  window to localise divergence to a pipeline stage (`f874f1ef1`);
- `diffDiFX.py --diagnose` — per-record best-fit complex gain/residual
  (amplitude error ⇒ weight bug, phase ⇒ timing bug), with separate
  tolerances for header weight and u/v/w (`b40eb7722`, `9dc608040`,
  `5d723be68`);
- the SLURM CPU-vs-GPU regression harness (`154fe8b37`) and its local
  no-SLURM equivalent `tests/Synthetic/run-local.sh` (`2a07491dc`);
- NVTX ranges around the host-side phases for nsys profiling
  (`1bf496b16`), and the `tests/FakeData` 10-station fake-data benchmark
  with mpirun/nsys recipes (`dfe8f0d66`).

## 5. Performance phase

Diagnosis first: on a 10-station benchmark the GPU was gap-dominated —
only ~15 s of GPU-busy inside ~70 s of correlation, because the single
Core thread alternates host work and GPU work. Groundwork and levers:

- fused XMAC grid enlarged, unnecessary atomics dropped, per-config
  launch metadata cached (`d4877acf1`, `65672fa36`, `7a5979afd`);
- one persistent compute stream shared by all modes, plus a start-up
  VRAM budget check that refuses jobs that cannot fit (`8ed77b08a`);
- visibility D2H transfer overlapped with the next subint
  (`DIFX_GPU_PIPELINE`, `789eba445`);
- first desktop NVTX profile (2026-07-17, RTX 2070, 10-station/10 ms
  job): ~42% GPU-busy; `host_accumulate` is the dominant host cost
  (71.6 ms per subint, 37% of wall); `fringeRotation` is 66% of GPU time
  on GeForce because of its double-precision phase math (FP64 is 1/32 of
  FP32 on Turing — a desktop-specific skew, but an optimisation target);
- **Lever A** (`785bcaec0`): the Core receive buffers are page-locked
  once at GPUCore construction and each mode's input H2D now DMAs
  directly from the delivered buffer, eliminating the per-subint host
  staging memcpy (formerly ~2250 us per datastream-subint on the cluster
  against ~280 us of actual PCIe). Gated by `DIFX_GPU_PIN_INPUT` with a
  warn-and-fall-back staging path; a per-slot `h2dInputDone` event makes
  the buffer-reuse invariant explicit for the coming de-serialization.
  Desktop benchmark: 269 s → 259 s (~4%); the cluster fraction was
  larger. Verified 6/6 Synthetic PASS in both pipeline modes.

- **De-serialization Increment 1 — set_weights on the device**
  (2026-07-18): the per-window weight/validity/sample-index calculation
  moved from a host loop (fed by a device-to-host copy of the frame
  validity, and followed by re-uploads) into a small per-window kernel
  that computes everything in place. All three per-datastream stream
  drains inside process_gpu disappear on the default path
  (`DIFX_GPU_WEIGHTS_HOST=1` restores the host path, which also carries
  full-fidelity WDEBUG). The config-static band map is now built once at
  construction. Verified: fractional boundary-window weights bit-identical
  to the CPU (WDEBUG money grep), 8/8 scenarios PASS, benchmark
  T5-T1 66.7 -> 62.4 s; nsys shows station_processing collapsed from
  83.6 to 63.6 ms/subint (= its GPU kernel content). host_accumulate
  (72 ms/subint, 48% of wall) is Increment 2's target. Design and audit
  trail: docs/gpu-deserialization-design.md.

- **De-serialization Increment 2 — baseline weights on the device**
  (2026-07-20): the per-window baseline-weight loop that dominated
  `host_accumulate` — summing `dataweight1[w]*dataweight2[w]` over the
  subint's FFT windows for every (freq, baseline, polproduct),
  O(windows x freqs x baselines x pols) on the host — moved to a device
  reduction kernel (`gpu_baseline_weights`, one thread per accumulator,
  sequential window sum matching the CPU order). A per-config plan built
  in `buildXmacPlans` gathers each baseline's two device `gDataWeights`
  arrays and records each accumulator's destination float offset into the
  results buffer; the reduced weights are D2H'd (a few hundred floats)
  and folded in by a flat, self-describing one-pass loop that cannot
  diverge from the plan enumeration. The host per-window loop and nested
  fold survive only on the `DIFX_GPU_WEIGHTS_HOST` fallback path. A code
  review caught a latent correctness bug fixed here: the invalid-subint
  early return zeroed the host `dataweight[]` but not the device
  `gDataWeights` the reduction reads, so an out-of-data datastream at a
  recording boundary would have contributed stale weights — the device
  buffer is now zeroed there too. Verified: 8/8 Synthetic scenarios PASS
  on the device path AND on the `DIFX_GPU_WEIGHTS_HOST` fallback, both
  pipeline modes; boundary-window weights byte-identical CPU-vs-GPU
  (WDEBUG); benchmark T5-T1 62.4 -> 32.4 s (~48%, from gutting the
  dominant host loop). Design/audit trail:
  docs/gpu-deserialization-design.md.

- **De-serialization Increment 2b — autocorrelation weights on the
  device, interim D2H dropped** (2026-07-21): the last routine bulk
  device-to-host copy on the weights path (the per-subint full
  `gDataWeights` array, kept through Increment 2 to feed the host AC
  per-band weight accumulation) is gone. The AC-weight loop's band map
  (`indices`/`countsStatic`) is window-independent, so the accumulation
  equals `totalW = sum_w dataweight[w]` times a config-static per-band
  multiplicity — only `totalW` is per-subint. A tiny reduction kernel
  (`gpu_sum_weights`, single-thread sequential sum matching the host
  window order) computes it on device, and only that one float is D2H'd
  each subint; the host loop collapses from O(windows x freqs) to
  O(freqs). The full per-window array is D2H'd only under the WDEBUG
  gate. Bit-identical AC weights in the common (single-occurrence-band)
  case — the device sum reproduces the host summation order — and
  FP-level for the rare multi-occurrence band. `gTotalWeight` is only
  read when `weightsOnDevice` is set (device branch only), so the
  invalid-subint/fallback paths never read a stale scalar (no explicit
  zeroing needed, unlike Increment 2's `gDataWeights`). Verified: 8/8
  Synthetic PASS device + `DIFX_GPU_WEIGHTS_HOST` fallback, both pipeline
  modes; boundary weights byte-identical (WDEBUG); benchmark T5-T1 flat
  at 32.4 s (a cleanup/D2H-removal, not a perf lever — the win was
  Increment 2). Design/audit trail: docs/gpu-deserialization-design.md.

- **Fringe-rotation interpolator hoisting** (2026-07-21): the
  `gpu_fringeRotation` / `gpu_complex_fringeRotation` kernels launched one
  thread per (FFT window, band, channel) and recomputed the FP64
  interpolator math — `d0`/`d1`/`d2` -> `a`/`b` (per window) and
  `bigAval`/`bigB_reduced` (per (window, band)) — in *every* thread, i.e.
  `numrecordedbands x fftchannels` times more than needed. A new
  `gpu_precompute_fringe_rotator` kernel (one thread per (window, band))
  computes `bigAval`/`bigB_reduced` once per subint into device arrays
  (`gBigA`/`gBigBred`); the per-sample kernels now only read those two
  values and form `exponent = bigAval*channel + bigB_reduced` -> FP32
  `__sincosf` -> complex multiply. Same FP64 expressions, just hoisted out
  of the inner loop, so it is numerically equivalent (the GPU final output
  is not bit-reproducible run-to-run regardless, because the XMAC
  accumulates with `atomicAdd`). This matters because `fringeRotation` was
  ~66% of GPU time on GeForce, where FP64 runs at 1/32 of FP32.
  Verified: 8/8 Synthetic PASS device + `DIFX_GPU_WEIGHTS_HOST` fallback,
  both pipeline modes; benchmark **T5-T1 32.4 -> 22.9 s (~29%)** — the
  largest single-change win since Increment 1. A per-sample precision drop
  (keep only `bigAval*(double)channel` in FP64, `bigB_reduced` as float)
  is a queued follow-up. Design: docs/gpu-fringerotation-design.md.

## 6. Multi-station, multi-subband correctness coverage

Until 2026-07-18 every correctness scenario was 2 stations with a single
subband, so the multi-datastream / multi-band machinery (the XMAC plans,
the per-datastream loops that the de-serialization work rewrites) was
never exercised by the safety net. The `multi` scenario adds 5 stations
(the two original test sites plus Parkes/Hobart/Ceduna-like southern
sites) each recording 4 x 4 MHz USB subbands as 4-channel VDIF
(`generateVDIF -C 4`, distinct seeds/tones per station). First run: PASS
CPU-vs-GPU in both pipeline modes, 1440 records compared at ~4e-7
relative difference. The local and SLURM harnesses now size their MPI
rank counts per scenario instead of assuming 4.

## 7. Desktop bring-up (2026-07-17/18)

Development moved from Mac + SLURM cluster to a local RTX 2070 desktop
(Fedora 43, CUDA 13.2, OpenMPI 5): the mpifxcorr `.cu.o` build rule and
configure.ac were made portable (configure-derived flags, env-respected
`NVCC`/`NVCCFLAGS`, CUDA include/lib paths derived from nvcc's location),
GCC 15 / C23 breakage in difx2fits and the vex parser was fixed
(`5c1a6f8d2`), and the local test/benchmark workflows above were
established.
