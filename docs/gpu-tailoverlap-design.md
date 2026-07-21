# GPU host-tail overlap — design / audit trail

## Problem

The clean A100 cluster profile (2026-07-21; one rank/core, `--exclusive`,
fake data, 400 subints) measured the GPU ~48% idle. The idle is real (not
oversubscription or disk I/O) and host-bound: 90% of it is in >500 µs
gaps, 70% a between-subints host tail, 20% intra-subint host waits;
sub-20 µs launch gaps are only 8.4% (so CUDA graphs stay ruled out).

Root cause is structural. The GPU path uses one compute stream and one
Core process thread (`gpucore.cu` forbids >1). `GPUCore::issuegpudata`
enqueued a subint's GPU work, **drained the stream**
(`cudaStreamSynchronize`), then ran the whole per-subint host tail —
`finishWeights`, `averageAndSendAutocorrs`, the baseline-weight fold,
`copyPCalTones`, and (in `completegpudata`) the visibility memcpy — with
the GPU parked, before `process_gpu` for the next subint launched a
kernel. The existing `DIFX_GPU_PIPELINE` split only overlapped the
visibility D2H.

## Approach: split each subint's GPU work at the FFT

Rather than double-buffer the per-datastream Mode device buffers (VRAM
cost, and it duplicates the big fftd/unpacked buffers), split the work so
the *input half* of the next subint fills the GPU while the current
subint's outputs drain and its host tail runs. The input half touches
none of the buffers the tail consumes, so no buffer needs duplicating.

`GPUMode::process_gpu` →
- `process_gpu_tofft`: input H2D, `calculatePre_cpu`, unpack,
  `gpu_set_weights` (sample indices / validity / per-window dataweights),
  fringe rotation, FFT.
- `process_gpu_afterfft`: `gpu_sum_weights` + `gTotalWeight` D2H, pcal
  extraction + copy, `fractionalRotation` (autocorrelations).

The FFT is the natural boundary, but two tail-consumed outputs —
`gTotalWeight` and pcal — were computed *before* the FFT in the old code.
Both were moved into the after-FFT half (neither feeds the FFT). After
this move the first half writes no host-tail-consumed buffer. The split
is ~48% / ~52% of kernel time on the A100 (unpack+fringe+FFT vs
fractional-rotation+XMAC+weights).

`GPUCore`:
- `issuegpudata` → `issue_tofft` (host prep + first half) and
  `issue_afterfft_xmac_drain` (second half + fused XMAC + baseline-weight
  reduction + output drain).
- the whole host tail moves to `completegpudata`.

## Loop (`GPUCore::loopprocess`)

```
setupAndTofft(slot0)                       # prologue: subint 0 first half
while keepprocessing[cur]:
    issue_afterfft_xmac_drain(cur)         # second half + XMAC + drain
    lock(next)                             # keep the manager one slot ahead
    if pipeline && next real && same cfg:
        setupAndTofft(next)                # pre-issue next first half
    completegpudata(cur)                   # await outputs + host tail
    unlock(cur); numprocessed++
    if new cur real:
        if config changed: updateconfig(...)
        if next not pre-issued: setupAndTofft(new cur)
unlock(terminator slot)
```

Invariant at each loop top: the current slot's first half has been issued
and its lock held. At most two slot locks are held (current + next),
matching the pre-pipelining protocol. The pipeline is broken (next first
half deferred until after complete) across a config change, at end of
data, and when `DIFX_GPU_PIPELINE=0`.

## Streams / events

- `cuStream` (compute): first half, second half, XMAC — in order.
- `d2hStream`: visibility D2H, after `cudaStreamWaitEvent(evComputeDone)`.
- **new** `evComputeDone[slot]` on cuStream after the second half + XMAC +
  baseline-weight D2H — replaces the mid-loop `cudaStreamSynchronize` so
  the next first half can be enqueued and run during the drain + tail.
- `d2hDone[slot]` (on d2hStream, after the vis D2H) — `completegpudata`
  waits it; because d2hStream first waited `evComputeDone`, this
  transitively covers the cuStream output D2Hs (autocorr / gTotalWeight /
  pcal / baseline-weights).
- `h2dInputDone[slot]` — unchanged slot-release gate.

## Why no buffer duplication is needed

- fftd / conj_fftd / gDataWeights: written by the next subint's first half,
  but consumed by the current subint's second half earlier on the *same*
  compute stream — stream order alone protects them.
- autocorr / gTotalWeight / pcal host pinned mirrors, and the single
  `h_bweightResults`: written by the current subint's second half, read by
  its `completegpudata` in the *same* loop iteration, and only overwritten
  by the *next* subint's second half (next iteration). The only thing
  between them — the next subint's first half — touches none of these.

## Per-Mode small-state aliasing (the subtle part)

The overlap requires the next subint's `issue_tofft` (including its host
prep) to run before the current subint's tail, and there is one Mode set.
So the prep mutates small per-Mode state the tail reads. Resolutions:

- **`weightsOnDevice` (invalid-subint gate).** `finishWeights` must skip
  an invalid subint, but the next subint's `set_weights` has already
  rewritten `weightsOnDevice`. Fixed by capturing each datastream's
  validity per procslot at issue time (`validsubint[slot][ds]` =
  `isSubintValid()` = `datalengthbytes>1 && offsetseconds!=INVALID_SUBINT`)
  and passing it to `finishWeights`; the mutable member is no longer read
  in the tail.
- **`zeroAutocorrelations` / `resetpcal`.** These clear host mirrors the
  tail writes/reads; they moved from the pre-GPU prep into the tail (device
  path), before `finishWeights` / `copyPCalTones`.
- **`DIFX_GPU_WEIGHTS_HOST` fallback.** This path fills `weights[][]` in
  `set_weights` (first half) and copies `autocorrelations[][]` in the
  second half — both before the tail — so the tail zeroing would wipe them.
  On this path `zeroAutocorrelations` runs at `issue_tofft` start instead,
  and the pipeline is **forced synchronous** (constructor sets
  `pipeline=false` when `!useGpuWeights()`): its host-weight-in-first-half
  + autocorr-in-second-half lifecycle cannot tolerate the next subint's
  first half being inserted between afterfft and complete. It is a
  debug/comparison path and does not need the overlap.
- **`DIFX_WEIGHT_DEBUG`.** Its per-window output prints Mode scalars that
  reflect the next subint under overlap, so it aborts at startup unless
  `DIFX_GPU_PIPELINE=0`.

## Verification

CPU-vs-GPU via `tests/Synthetic/run-local.sh` on an RTX 2070: usb,
usb-complex, complex-complex, multi all PASS in both `DIFX_GPU_PIPELINE`
modes; usb/complex-complex/multi PASS on the `DIFX_GPU_WEIGHTS_HOST`
fallback. Performance (desktop T5-T1; A100 idle-collapse re-profile via
`benchprof-profile.sbatch`) is pending.

## Rejected alternative

Double-buffering the Mode output buffers (two full Mode sets, or
parity-indexed small mirrors). Correct, but either ~2× per-Mode VRAM
(busts the 8 GB 2070 at benchprof scale; needs a VRAM gate + per-parity
XMAC plans) or parity-indexed accessor surgery across GPUMode. The
half-split gets the same overlap with zero extra device memory.
