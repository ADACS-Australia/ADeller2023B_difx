# GPU profiling: archived reference numbers

Superseded profile captures, split out of `gpu-profiling.md` on 2026-08-28.
Kept because the current numbers are quoted against them - the 2026-07-21
capture in particular is the "before" in every idle-fraction comparison.

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
