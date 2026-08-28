# Shrinking `gpu_resultsrotatorMultiply`: three changes, in order of value

Design note for the two changes that shrink `gpu_resultsrotatorMultiply`, the
largest A100 kernel (42-44% of kernel busy). Companion to `gpu-plan.md`
(items 6, and a new Longer-term entry) and `tests/frac-probes/RESULTS.md`
(the measurements this rests on).

## What the measurements say

`gpu_resultsrotatorMultiply` does four things in one kernel: the fractional
sample rotation, the conjugation, the **per-band autocorrelation**, and the
**cross-pol autocorrelation**. Four env-gated timing probes
(`DIFX_GPU_FRAC_PROBE=0..3`, on the since-deleted `probe-frac-limits` branch),
run in one job on
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

## Part 2: window-group reduction - BUILT, MEASURED, NOT LANDED (2026-08-26)

Implemented and validated correct, then measured as a net loss on the 2070 with
only a speculative gain on the A100. The branch was **deleted on 2026-08-27**
once Part 3 was agreed, because Part 3 removes these atomics outright rather
than reducing them. The design below is therefore the only surviving record, and
is deliberately detailed enough to rebuild from: the thread mapping, the barrier
structure, the correctness hazard, and the measurements that killed it.

**RTX 2070, `gpu_resultsrotatorMultiply` us/call, vs 396.9 for the kernel
without any of this machinery:**

| block (chan x windows) | threads | us/call | vs baseline |
|---|---|---|---|
| (64, 2) | 128 | 411.4 | +3.7% |
| (32, 4) | 128 | 416.7 | +5.0% |
| (128, 1) | 128 | 424.5 | +7.0% |
| (16, 8) | 128 | 467.0 | +17.7% |
| (128, 8) | 1024 | 576.7 | +45% |

Two lessons. **Widening the block to fit a window group is the expensive part** -
1024-thread blocks on a card with 1024 threads/SM leave one block per SM where
there were eight; narrowing the channel dimension instead keeps the block at 128
threads and makes the grouping nearly free. But **the machinery has a floor**:
G=1 reduces nothing and still costs 7%, so the shared-memory round-trip and the
two barriers per band are themselves the cost. On a bandwidth-saturated card the
atomics being removed were free (probe 1: +0.0%), so nothing pays for that floor.

The A100 is the opposite case - atomics are 24% of the kernel - so this could
plausibly net ~13% of the kernel, ~5.5% of kernel busy. Unmeasured. But **Part 3
subsumes it entirely and is worth ~3x more**, so the next increment should go
there rather than chasing this. Correctness was never the problem: racecheck
reports 0 hazards over a complete correlation, and every validation leg passed.

## Part 2 design (as built, for reference)

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

## Part 3: implementation plan (2026-08-28)

Code reading turned the shape of this work around. The design note above framed
it as invasive - "it moves a data path across `Mode`, `Core`, `Configuration` and
the results-buffer layout". Most of that is already built:
`gpu_fuse_xmac_and_average` is launched **per frequency**, and every per-baseline
thing it needs is already a host-built array - the two spectra pointers, the two
validity arrays, the per-stream band indexes, the per-stream window/band strides,
and the output offset. **An autocorrelation is a baseline whose two streams are
the same datastream**, so most of this is plan construction on the host, not new
device code.

### The one kernel change

Output indexing today is `coreResultBaselineOffsets[baseline] + pol *
num_averaged_channels + avg_chan`, i.e. a baseline's pol products are contiguous.
The autocorrelation region is not laid out that way: per datastream it is
`[band][channels]` over the *used* bands (`Core::processdata`, the
`getCoreResultAutocorrOffset` loop), and the cross-pol autocorrelations sit in a
separate block after it. So generalise to a per-(baseline, pol) offset array:

    coreResultOffsets[baseline * numPolarisationProducts + pol]

Cross-baselines fill it with exactly `base + pol * num_averaged_channels`, so
their behaviour is bit-identical. Everything else the kernel already does -
skipping invalid windows, the `channelstoaverage` mean, the store-vs-atomic
choice - is what the autocorrelations need too.

Two facts checked rather than assumed, because both would have sunk the plan:

- **`Mode::averageFrequency` is a mean** (`vectorMean_cf32` over
  `channelstoaverage`), which is exactly the kernel's `1/channelstoaverage`
  scaling. No scaling mismatch.
- **The AC weights do not come from the autocorrelation values.** They are
  `modes[j]->getWeight(false, k)`, the data weights (`core.cpp`, the
  `getCoreResultACWeightOffset` loop). So removing the device autocorrelation
  buffer cannot disturb them - which the Part 1/2 verification notes worried
  about.

### Increments

**Increment 0 - a dual-polarisation synthetic scenario. DONE 2026-08-28**
(`tests/Synthetic/test-dualpol.{vex,v2d}`, commit `5f6c3961d`): the usb scenario
with a second channel at the same sky frequency in the opposite polarisation and
`doPolar = True`. PASSes CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes and under
`DIFX_GPU_WEIGHTS_HOST=1`, and the output records were parsed to confirm it
exercises the path rather than merely running: ant1-ant1 and ant2-ant2 carry RL
and LR records at rms |V| 3.37 against 5.71 for the parallel hands. The original
reasoning follows.

**Increment 0 - a dual-polarisation synthetic scenario. Do this first.**
Every scenario in `tests/Synthetic` is single-pol (all Rcp), so
`writecrossautocorrs && maxproducts > 2` is never true and **the cross-pol
autocorrelation path has no test coverage at all** - not on the GPU, not on the
CPU. That is the part of this work with the highest divergence risk, and moving
it while untested would be moving code we cannot validate. A 2-pol scenario also
retro-covers the existing kernel's `calccrosspolautocorrs` block, which is
review-validated only today.

**Increment 1 - per-(baseline, pol) output offsets.** The kernel change above,
with the cross-baseline plan filling the array as today. No behavioural change;
lands on its own so the invasive step starts from a green base. Verify: Synthetic
PASS both pipeline modes, bench flat.

**Increment 2 - self-baselines in the per-frequency plan**, gated by
`DIFX_GPU_XMAC_AUTOCORR` (default off while it is being built). For each
frequency `f` and each datastream `j` using it, append a baseline with both
stream pointers, validity arrays and strides set to `j`'s, and pol slots:
`(band, band)` for each of `j`'s bands at `f` (the parallel autocorrelations),
plus `(bandA, bandB)` and `(bandB, bandA)` when the cross-pol autocorrelations
are wanted. Unused slots get `-1`, which the kernel already skips.
`numPolarisationProducts` becomes a max over cross and self baselines rather than
a reference baseline's value. The offsets come from
`getCoreResultAutocorrOffset` and must replicate `Core::processdata`'s
`isFrequencyUsed`/`isEquivalentFrequencyUsed` band filtering exactly, or every
subsequent band's data lands in the wrong slot.

**Increment 3 - delete the old path.** Only once increment 2 is PASSing: the
per-band `atomicAdd` and the whole cross-pol block leave
`gpu_resultsrotatorMultiply` (with `calccrosspolautocorrs`, `counts_gpu` and
`indices`); `temp_autocorrelations_gpu`, its per-subint memset and D2H, and the
two host mirror loops in `gpumode.cu` go; and `Core`'s autocorrelation
accumulation is skipped on the GPU path (it must keep running for the CPU path -
the divergence class that has caused the most bugs here).

### Risks, in the order they are likely to bite

| risk | handling |
|---|---|
| **Zoom bands.** The autocorrelation region covers `getDNumTotalBands`, which includes zoom bands; the GPU path handles recorded bands only. | Check at plan-construction time and `NOT_SUPPORTED` if zoom bands are present, rather than silently writing short. |
| **Band filtering must match `Core::processdata` exactly** (`isFrequencyUsed` / `isEquivalentFrequencyUsed`). | Build the offsets with the same predicate, and assert the total consumed length equals the region size. |
| **Cross-pol autocorrelation slots are untested today.** | Increment 0. |
| Store-vs-atomic: a self-baseline slot is written once per subint, so the kernel's plain-store branch is correct - but only if nothing else writes there. | Assert disjointness of the autocorr offsets against the baseline offsets when building the plan. |
| Pulsar binning / multiple phase centres change the results layout. | Already `NOT_SUPPORTED` on the GPU path; confirm the guard covers this path too. |

### Expected gain

`gpu_resultsrotatorMultiply` loses its atomics (24% on the A100) and its cross-pol
phase (25%), leaving a pure elementwise rotate: **~-49% of a kernel that is 34.9%
of A100 kernel busy**. The XMAC grows by the self-baselines (45 -> 55 for
benchprof, and it re-reads spectra already in its L2 working set, so
sub-proportional). Net **~12-15% of A100 kernel busy**, plus the host tail:
`host_finalize` is 150 ms over 500 subints in the current A100 profile, and the
per-subint autocorrelation D2H goes with it. On the 2070 the atomics are free and
the rotator is only 23.3%, so expect ~6% there - this is a cluster optimisation,
like the autocorrelation work always was.

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
