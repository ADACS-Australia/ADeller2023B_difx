# Shrinking `gpu_resultsrotatorMultiply`: three changes, in order of value

Design note for the two changes that shrink `gpu_resultsrotatorMultiply`, the
largest A100 kernel (42-44% of kernel busy). Companion to `gpu-plan.md`
(items 6, and a new Longer-term entry) and `tests/frac-probes/RESULTS.md`
(the measurements this rests on).

## What the measurements say

`gpu_resultsrotatorMultiply` does four things in one kernel: the fractional
sample rotation, the conjugation, the **per-band autocorrelation**, and the
**cross-pol autocorrelation**. Four env-gated timing probes
(branch `probe-frac-limits`, `DIFX_GPU_FRAC_PROBE=0..3`), run in one job on
gina13, decompose its 396 us/launch on the A100:

| probe | A100 us/call | delta | RTX 2070 delta |
|---|---|---|---|
| 0 baseline | 396.5 | - | - |
| 1 `atomicAdd` -> plain store | 299.9 | **-24.4%** | +0.0% |
| 2 cross-pol phase off | 248.6 | **-37.3%** | -27% |
| 3 bands on `gridDim.z` | 909.7 | **+129%** | +139% |

So, on the A100: **atomics ~97 us (24%)**, **cross-pol traffic ~100 us (25%)**,
band-phase traffic + compute ~200 us (50%). Every other kernel is stable to
<1% across the four captures, so the attribution is clean.

Two findings shape everything below.

**The atomics matter on the A100 and not at all on the 2070.** The 2070 is
bandwidth-saturated (~85% of its 448 GB/s), so atomic latency hides behind
memory stalls; the A100 runs the same traffic at 26% of peak, so the
serialisation is exposed. A "what is this bound by" answer does not transfer
between these cards - measure on the one you care about.

**Do not move bands out of the in-thread loop.** Probe 3 gave every
(window, band, channel) its own thread - 16x the parallelism, identical total
traffic - and it was 2.3x SLOWER on the A100 and 2.4x slower on the 2070.
Today a block streams one window's bands as B contiguous ~1 KB reads; with
bands on the grid, concurrently-scheduled blocks are consecutive *windows* of
one band, `fftchannels * numrecordedbands * 8` bytes apart (32 KB here). The
effect is consistent across both cards, so it is locality, not a card quirk.
**This rules out the (band, channel, window-chunk) mapping** that the earlier
plan proposed.

## Part 1 (landing now): delete the materialised conjugate array

The pipeline used to keep a second full array of spectra, `conj_fftd_gpu`,
holding nothing but the conjugate of `fftd_gpu`, so that the XMAC and the
cross-pol autocorrelations could read a pre-conjugated operand. The reasoning
was reuse: each station's conjugate is read by up to Nstations-1 baselines, so
conjugating once looked cheaper than conjugating per use.

It is not, because **conjugating inside the multiply is free**:

```
(a+bi)(c+di) = (ac - bd) + (ad + bc)i      4 mul, 2 add
(a+bi)(c-di) = (ac + bd) + (bc - ad)i      4 mul, 2 add
```

Identical instruction counts - only the signs differ, and they fold into the FMA
sign bits. So the "saving" bought nothing, while the array cost:

- **an 8-byte global write per spectral point** in the rotator: 82 MB/launch of
  its ~205 MB. At the traffic sensitivity probe 2 measured (~100 us per 82 MB)
  that is **~100 us/launch, ~500 ms, ~11% of A100 kernel busy**;
- **82 MB of VRAM per Mode** - 820 MB across benchprof's 10 datastreams,
  which matters for the 8 GB 2070 and for the subint-must-fit-in-VRAM limit;
- **double the XMAC's working set** (164 MB of spectra rather than 82 MB) on a
  kernel measured at an effective 3.6 TB/s, i.e. served from L2, where
  footprint is exactly what matters;
- a per-subint memset of the array on the invalid-subint path.

`cuCmulConjf(a, b)` (in `gpumode_kernels.cuh`, shared by both translation units
rather than duplicated) replaces it. Consumers:

- the XMAC's `m2` pointer array now points at `fftd_gpu`, and the product
  becomes `cuCmulConjf(v1, v2)` - same orientation, `v1 * conj(v2)`;
- the **band** autocorrelation is now taken from the register the rotator
  already holds, as `v.x*v.x + v.y*v.y`, and accumulated into the real
  component only. `v * conj(v)` is real by construction, so the second atomic
  per element was adding a provably-zero imaginary part - this is the real-only
  optimisation identified but left unapplied during the July item-4
  investigation;
- the **cross-pol** autocorrelations read both operands from `fftd_gpu` and
  conjugate in the multiply. Each value they read was written by the same
  thread earlier in its own band loop (same window, same channel, different
  band), so there is no new race. They also get cheaper: previously two cold
  reads per product from two different arrays, now two reads from one array
  that the band loop has just touched.

**Numerics.** The expressions are the same, so differences are at most 1 ulp
from FMA grouping - well inside the acceptance bar, and the same class as the
FMA-contraction differences already documented for this project. Verified
bit-identical in a non-contracted host build for generic, sign-flipped and
subnormal operands. The one real change is the band autocorrelation's imaginary
component, which was accumulating FMA rounding residue (~1e-8 of the real part)
around a physically-zero value and is now exactly zero.

**Why this one goes first:** it is the smallest of the three changes, the only
one whose win should be visible on the 2070 (bandwidth-bound there, and this
removes write traffic - so `run-bench.sh` can measure it locally instead of
waiting for cluster time), and it frees memory rather than spending it.

## Part 2 (next): window-group reduction

Cut the atomics without touching the thread count, the per-thread work, or the
block->window mapping - the three things probe 3 shows are fragile.

Today one block is one FFT window and its threads are channels, so every
`(band, channel)` autocorrelation address in that block is touched exactly once
and there is nothing to reduce locally; the 5000-deep contention is *across*
the 2500 blocks. The fix is to put **G windows in the same block**, so each
address is touched G times within it and can be reduced before the atomic:

```
G     = window group (tunable, see below)
grid  = dim3(ceil(numBufferedFFTs / G), fftchannels_grid)
block = dim3(fftchannels_block, G)           // e.g. (128, 8) = 1024 threads
        subloopindex = blockIdx.x * G + threadIdx.y
        channelindex = blockIdx.y * blockDim.x + threadIdx.x
```

Every thread still owns exactly one window and still loops all bands inside, so
the per-thread access pattern is bit-for-bit what it is today. Thread count is
unchanged (`numBufferedFFTs * recordedbandchannels`). Only the *grouping* of
threads into blocks changes, and with it the number of global atomics:
`W*C*(B+2F)` becomes `(W/G)*C*(B+2F)`, a **G-fold reduction**.

Reduction mechanics, band by band inside the existing loop:

```
extern __shared__ cuFloatComplex sh[];      // blockDim.x entries, ~1 KB
for (band = 0; band < numrecordedbands; band++) {
    if (threadIdx.y == 0) sh[threadIdx.x] = 0;
    __syncthreads();
    if (active) { rotate; write fftd; write conj;
                  shared-atomicAdd(&sh[threadIdx.x], v * conj(v)); }
    __syncthreads();
    if (threadIdx.y == 0) global-atomicAdd(&autocorr[band*C + tx], sh[tx]);
}
```

Deliberately **band-by-band with a one-channel-row buffer**, not one buffer for
all bands: shared memory then stays ~1 KB regardless of band count, so a
64-band config does not lose occupancy. Shared-memory atomics are hardware ops
and G-way contention is trivial. Cross-pol gets the same treatment, reduced per
`(freq, channel)`.

### The correctness hazard to get right

The current kernel `return`s early in three places (invalid subint, past
`numblocks`, channel past `recordedbandchannels`). With G windows per block,
**different `threadIdx.y` values have different `subloopindex`, so those returns
are no longer block-uniform** - a thread that returns never reaches
`__syncthreads()`, which is undefined behaviour. Every early return must become
an `active` predicate that inactive threads carry through the loop (contributing
nothing but still synchronising). This is the single most likely way to get this
change subtly wrong.

### Choosing G, and behaviour across the parameter range

```
blockDim.x = min(recordedbandchannels, 256)
G          = clamp(1024 / blockDim.x, 1, min(numBufferedFFTs, 32))
```

| regime | blockDim.x | G | effect |
|---|---|---|---|
| benchprof (128 chan) | 128 | 8 | 8x fewer atomics, 1024-thread blocks |
| few channels (16) | 16 | 32 | 32x fewer - small-channel configs, which have the fewest threads, get the most grouping |
| many channels (2048) | 256 | 4 | 4x; already plenty of threads |
| few windows (W=4) | - | <=4 | clamped to W, still correct |
| G == 1 | - | 1 | degenerates to exactly today's kernel |

Band count does not enter the mapping at all, which is what makes this robust
where the previous design was not. Override with
`DIFX_GPU_FRAC_WINDOW_GROUP` for A/B, per the existing `DIFX_GPU_*` convention.

**Expected gain:** up to ~97 us/launch on the A100 = ~484 ms = **10.7% of
kernel busy**; G=8 recovers most but not all of it. **Flat on the 2070** by
construction (atomics are free there), so `run-bench.sh` will not show it - the
number has to come from a cluster re-profile. Behaviour change: the
autocorrelation summation order moves, so FP-level not bit-identical - the same
class as the §13 weight-sum fusion, within the acceptance bar.

## Part 3 (future, and it subsumes Part 2): autocorrelations belong in Core

Adam's observation, 2026-08-26: there is nothing special about an
autocorrelation that requires it to happen in `Mode`. It is the baseline where
both stations are the same antenna, so it belongs with the cross-multiply in
`Core` - the split is an old design quirk.

Tracing the current path makes the case stronger than a kernel argument. Today:

1. `gpu_resultsrotatorMultiply` accumulates autocorrs at **full** spectral
   resolution into `temp_autocorrelations_gpu` (`autocorrwidth * nbands *
   recordedbandchannels`), via the atomics measured above.
2. that buffer is D2H'd every subint;
3. `finishWeights` (host tail) `vectorCopy_cf32`s it into
   `Mode::autocorrelations[i][j]` - for benchprof, ~655 KB of host memcpy per
   subint across the 10 datastreams;
4. `Core::processdata` calls `Mode::averageFrequency()`, which averages
   128 -> 64 channels **on the host**;
5. `Core::processdata` then `vectorAdd_cf32_I`s the result into
   `procslots[index].results` at `getCoreResultAutocorrOffset(...)`, under
   `autocorrcopylock`.

Meanwhile `gpu_fuse_xmac_and_average` already accumulates over windows, already
averages down to `num_averaged_channels`, and already writes straight into
`results_gpu` at `coreResultBaselineOffsets`. **Autocorrelations are the same
operation into a different offset** - `getCoreResultAutocorrOffset` already
exists for exactly that slot. Adding self-baselines to the XMAC would delete
all five steps above:

- the atomics (~97 us/launch) and the cross-pol phase (~100 us/launch) leave
  `gpu_resultsrotatorMultiply` entirely, reducing it to a pure elementwise
  rotate-and-conjugate of ~200 us;
- `temp_autocorrelations_gpu` and its per-subint D2H disappear;
- steps 3-5 - all **host tail** work, the thing the tail-overlap work (§8) has
  been fighting, and visible as `host_finalize` in the A100 profile - disappear
  with them, along with `autocorrcopylock`.

Cost: the XMAC grid grows by the self-baselines (for benchprof, 45 -> 55
baselines, ~22%), but it re-reads spectra already resident in its L2 working
set, so the marginal cost should be sub-proportional - call it +40 us against
-197 us, i.e. **net ~17% of A100 kernel busy**, plus the host-tail saving.

**It also makes Part 2 redundant**: with no autocorrelation accumulation left in
the kernel there are no atomics to reduce. Part 2 is worth doing before it only
because it is contained (one kernel + its launcher, a day's work) whereas this
is not - it moves a data path across `Mode`, `Core`, `Configuration` and the
results-buffer layout, and it has to keep the CPU path's Mode-based
autocorrelations working in parallel, which is precisely the CPU/GPU divergence
class that has produced the most bugs on this branch. Sequencing it second is a
deliberate choice to take a cheap 10% now rather than block on a 17% refactor;
if that trade looks wrong, do Part 3 directly and skip Part 2.

## Verification (Part 1 - the conjugate removal)

1. `tests/Synthetic/run-local.sh usb usb-complex complex-complex multi` - PASS
   CPU-vs-GPU in **both** `DIFX_GPU_PIPELINE` modes **and** under
   `DIFX_GPU_WEIGHTS_HOST=1` (the autocorrelations feed the AC weights, and the
   band autocorrelation's accumulation changed).
2. `run-bench.sh` T5-T1 best-of-3 on the 2070 - this one **should** show a win,
   unlike Parts 2 and 3; the 2070 is bandwidth-bound and this removes ~40% of
   the rotator's traffic. Expect the rotator kernel down ~20-27%.
3. Confirm the device-memory estimate in the difxlog drops by ~82 MB per mode.
4. Cluster re-profile for the A100 number.

## Verification (Part 2 - the window-group reduction)

1. `tests/Synthetic/run-local.sh usb usb-complex complex-complex multi` - PASS
   CPU-vs-GPU in **both** `DIFX_GPU_PIPELINE` modes **and** under
   `DIFX_GPU_WEIGHTS_HOST=1` (the autocorrelations feed the AC weights).
2. A small-parameter scenario (few bands/channels) so the `G` clamp is
   exercised by a real run, plus `DIFX_GPU_FRAC_WINDOW_GROUP=1` to confirm the
   G==1 path reproduces today's kernel.
3. `compute-sanitizer --tool=racecheck` on one synthetic run - the new
   `__syncthreads`/shared-reduction structure is exactly what it is for.
4. `run-bench.sh` T5-T1 on the 2070 for the ledger (expected flat).
5. Cluster re-profile for the real number: `sbatch --nodes=1 --cpus-per-task=5
   benchprof-profile-nsys-5s.sbatch`, checking the `node ownership` line.
6. Code review before validating; full diff before commit.
