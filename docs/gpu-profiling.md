# GPU correlator profiling — method & reference numbers

How to profile the GPU idle, and the reference numbers from the
2026-07-21 A100 investigation that found the between-subints idle is a
per-station cuFFT synchronisation (not the host tail). Companion to
`gpu-plan.md` (what to do next) and `BENCHMARKS.md` (the ledger).

## How to capture

`tests/FakeData/benchprof-profile.sbatch` runs a clean GPU-bound
correlation (A100, one rank/core `--exclusive`, `source=fake` so no disk
I/O, 400 subints, 10 stations) with `tests/FakeData/nsys-wrapper.sh`
wrapping **only the Core rank** (the wrapper picks the Core rank via the
OMPI/SLURM/PMIX rank env var; rank 0 deletes stale `.difx`). One capture =
one correlator process. The `.nsys-rep` lands in `nsys/` under the run
dir. The cluster login node's nsys is older than the capture and cannot
`nsys stats`/`export` it — **scp the `.nsys-rep` to the desktop**
(nsys 2025.6.3) to analyse.

The desktop `benchprof` (`run-bench.sh` config) is `source=fake` too and
can be profiled the same way; the synthetic `multi` scenario reads VDIF
so its profile is I/O-polluted (useful only as a cross-check).

## How to analyse — one command

```bash
tests/FakeData/nsys-analyze.py <capture>.nsys-rep   # or a pre-exported .sqlite
```

Exports sqlite (if needed) and prints, in one pass, every summary used in
the investigation: device/spans, GPU busy/idle union + gap histogram,
big-gap NVTX attribution, NVTX phase breakdown, kernel mix, memcpy
volume, per-thread OS-runtime blocking, and the cuFFT per-exec-sync
fingerprint. Single compute stream, so GPU "busy" = union of
kernel+memcpy intervals and "idle" = the gaps.

### The gap-analysis heuristic

- **Many small gaps (<20 us)** dominate idle → launch-overhead bound
  (CUDA graphs would help). *Not the case here — 8.4% of idle.*
- **Few big gaps (>500 us)** dominate idle → host / data-delivery bound
  (graphs won't help). *This case — 85.6% of idle.*
- Attribute each big gap to the innermost NVTX range at its midpoint. A
  gap covered by **no** NVTX range is host code between the instrumented
  phases (loop overhead, lock waits, or a library-internal sync).
- Cross-check host blocking: `OSRT_API` per thread (pthread_mutex_lock,
  `process_vm_readv` = MPI shared-mem recv) and
  `CUPTI_ACTIVITY_KIND_RUNTIME` per thread (cudaStreamSynchronize /
  cudaEventSynchronize). The NVTX-emitting thread is the loopprocess
  (kernel-issuing) thread; the process-named thread doing `process_vm_readv`
  is the Core main thread (`receivedata`).

## Reference numbers — A100-SXM4-80GB, commit 7b8e31104 (host-tail overlap)

400 subints, 10 stations. Pre-overlap reference (80f6e291a) in brackets.

| quantity | value |
|---|---|
| kernel span (first→last kernel) | 8798 ms  (pre: 10497) |
| GPU busy, kernels only | 5088 ms = 57.8% (pre: 5471 = 52%) |
| GPU busy, +compute-stream copies (union) | 5695 ms = 64.7% |
| **GPU idle (union)** | **3104 ms = 35.3%** (pre: ~48%) |
| idle in >500 us gaps | 2657 ms = 85.6% of idle |
| CUDA-API span (wall proxy) | 10877 ms |
| H2D on compute stream | 11.2 GB / 580 ms;  D2H 0.42 GB / 44 ms |

**Kernel mix:** gpu_resultsrotatorMultiply 29.6%, gpu_unpack 23.7%,
gpu_fuse_xmac_and_average 16.0%, gpu_fringeRotation 15.1%, vector_fft
8.1%, gpu_sum_weights 5.6%, rest <1% each. Fringe family (fractional
rotation + fringe rotation + precompute) ~45%. FP64 is cheap on the A100,
so precision work (2070-only) is not a cluster lever.

**NVTX phases (host-side, per subint):** complete_d2h_wait ~456 us,
station_processing ~446 us, host_finalize ~337 us; set_weights /
calculatePre_cpu / h2d_stage fire ~10×/subint at tens of us each. The
whole host tail is ~900 us/subint — small.

### The finding: per-station cuFFT stream sync

The process (kernel-issuing) thread spends **5.58 s in
`cudaStreamSynchronize`** — 3947 calls ≈ **10 per subint**, one per
station. The count matches cuFFT's per-`cufftExecC2C` driver footprint
exactly: `cuStreamIsCapturing` (3948), driver `cuLaunchKernel` (3948),
`cudaStreamSynchronize` (3947), all ≈ 10 stations × 400 subints. So
**cuFFT synchronises the compute stream on every FFT exec**, serialising
host and GPU at station granularity. Per subint: the first sync (in the
tofft loop) backs up behind the whole XMAC phase (~7 ms); the other nine
each wait one station's kernels (~720 us).

Corroboration: our device-path source has **no** per-subint sync (every
`GpuMemHelper::sync()` / explicit `cudaStreamSynchronize` in gpumode.cu /
gpucore.cu is one-time setup or gated on `!useGpuWeights()` / SPECDEBUG,
and pipelining was ENABLED). The Core **main** thread's 6.2 s of
`pthread_mutex_lock` (waiting to reuse a procslot) is a *symptom* — it
waits on the FFT-sync-throttled process thread — not the cause. cuFFT
plans are built with default auto work-area allocation (gpumode.cu ~451,
`cufftPlanMany` + `cufftSetStream`, no `cufftSetAutoAllocation` /
`cufftSetWorkArea`), a known trigger for per-exec syncs.

Consequence for the plan: the host-tail overlap (7b8e31104) could not
collapse this idle because the tail was never the bottleneck. Killing the
FFT sync is work-queue item 0 in `gpu-plan.md`; kernel fusion (item 1)
only attacks the ~14 ms/subint busy half and is deferred behind it.
