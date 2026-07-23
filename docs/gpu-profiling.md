# GPU correlator profiling — method & reference numbers

How to profile the GPU idle, and the reference numbers from the
2026-07-21/22 A100 investigation that found the between-subints idle is a
per-station whole-stream drain in `Mk5_GPUMode::unpack_all` (first
misattributed to cuFFT — see the corrected finding below; fixed 2026-07-22).
Companion to `gpu-plan.md` (what to do next) and `BENCHMARKS.md` (ledger).

## How to capture

`tests/FakeData/benchprof-profile-nsys-5s.sbatch` runs a clean GPU-bound
correlation (A100, one rank/core, `source=fake` so no disk I/O, ~5 s =
~250 x 20ms subints, 10 stations; defaults to single-thread VDIF via
`SINGLE_THREAD_VDIF=1` so it profiles the GPU rather than the DataStream
interlaced-VDIF corner-turn) with `tests/FakeData/nsys-wrapper.sh`
wrapping **only the Core rank** (the wrapper picks the Core rank via the
OMPI/SLURM/PMIX rank env var; rank 0 deletes stale `.difx`). One capture =
one correlator process. The `.nsys-rep` lands in `nsys/` under the run
dir. The cluster login node's nsys is older than the capture and cannot
`nsys stats`/`export` it — **scp the `.nsys-rep` to the desktop**
(nsys 2025.6.3) to analyse.

**Keep captures short.** nsys 2022.2.1's injection library
(`libToolsInjection64.so`) segfaults on a long/heavy trace — a ~20 s
`cuda,nvtx,osrt` capture over ~1000 subints crashed its worker thread
~13 s in (the backtrace was entirely inside `libToolsInjection64.so`; the
correlator was fine). Hence the ~5 s window; if it still falls over, drop
`osrt` (`NSYS_TRACE=cuda,nvtx`) or use a newer nsys. For a full-length
**timing/soak** run (no profile), use the nsys-free companion
`benchprof-profile-nonsys-20s.sbatch`.

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

### The finding: per-station whole-stream drain in `unpack_all` (NOT cuFFT)

The process (kernel-issuing) thread spends **5.58 s in
`cudaStreamSynchronize`** (A100; ~13 s on the 2070) — ~3947 calls ≈ **10
per subint**, one per station. Per subint: the first sync (in the tofft
loop) backs up behind the whole XMAC phase (~7 ms); the other nine each
wait one station's kernels (~720 us).

**First misattributed to cuFFT** because the count matched
`cufftExecC2C`'s footprint (`cuStreamIsCapturing` 3948, driver
`cuLaunchKernel` 3948, `cudaStreamSynchronize` 3947). That match was a
coincidence — unpack and the FFT are in the same per-station tofft
iteration. **The real source is `Mk5_GPUMode::unpack_all` ->
`valid_frames->sync()`**, established three ways (2026-07-22):

1. `utilities/fft-profiling/fftbench.cu` — a faithful standalone cuFFT
   microbench — shows `cufftExecC2C` returns ~0 ms after a 50 ms stream
   backlog on both the 2070 (cuFFT 12.2) and the A100 (CUDA 12.8): async,
   no per-exec sync.
2. Stubbing out the `cufftExecC2C` call left the sync count unchanged
   (4021 on the 2070).
3. `nsys profile --sample=cpu --backtrace=dwarf --cudabacktrace=sync:1000`
   then joining `CUDA_CALLCHAINS` (stackDepth) to the sync rows resolves the
   caller to `Mk5_GPUMode::unpack_all`.

To find a sync's caller: run with `--cudabacktrace=sync` **and**
`--sample=cpu` (the unwinder needs sampling active; `--sample=none` leaves
`callchainId=0`), export sqlite, then
`SELECT sd.value, mo.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN
CUDA_CALLCHAINS c ON c.id=r.callchainId JOIN StringIds sd ON c.symbol=sd.id
JOIN StringIds mo ON c.module=mo.id WHERE ... ORDER BY c.stackDepth`. Also
useful: `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.syncType` (via
`ENUM_CUPTI_SYNC_TYPE`) tells stream-sync vs event-sync vs wait-event.

The drain existed only so the `DIFX_GPU_WEIGHTS_HOST` fallback could read
`valid_frames` on the host; on the device path `gpu_set_weights` reads
`valid_frames->gpuPtr()` directly. **Fixed 2026-07-22** (gpu-changes.md
§9): gated to the fallback, with RING-deep host staging so the tail-overlap
pipeline stays correct without the drain's implicit barrier.
`cudaStreamSynchronize` dropped 4021 -> 31. The Core **main** thread's
`pthread_mutex_lock` time was a *symptom* (waiting on the drain-throttled
process thread), not a separate cause.
