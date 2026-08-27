# Tiled `gpu_fused_fringe`: a shared-memory transpose for the fringe write

Design for making the fused decode+fringe-rotation kernel's global write
coalesced without un-coalescing its decode read. Companion to
`gpu-fringerotation-design.md` (which covers the FP64 precompute already landed)
and `gpu-profiling.md` (the ncu measurements that motivate this).

Status: **implemented and verified on the 2070, 2026-08-27** (`DIFX_GPU_FRINGE_TILE`,
default on). Results at the end; the A100 A/B is still to run.

## The measurement

`ncu` on the RTX 2070 SUPER, 2-station `benchprof2` (16 bands x 128 channels,
`fftchannels` 256), current build:

| | value |
|---|---|
| duration | 948 us/launch — **46.5% of all 2070 kernel time** |
| Compute (SM) throughput | 53.6% |
| DRAM throughput | 21.1% |
| warp stalls | **58.4%** of 38.6 cycles/instruction on **L1TEX scoreboard** |
| uncoalesced global access | **29% excessive sectors** (2.56M of 8.96M) |
| `Block Limit Warps` | **1** (block size 1024 = 32 warps = the SM's whole warp budget) |

ncu's own estimates: 46% from the stall, 22% from the coalescing, 16% from the
occupancy gap. They overlap; the design below addresses all three.

## Why the write is scattered, and why the mapping is nonetheless right

The kernel maps `threadIdx.x = bandindex` (16), `threadIdx.y = channel` (64), so
a warp's lane index runs `band + numrecordedbands * channel`. Both sides of the
kernel see that mapping, and they want opposite things:

- **Read** (`decode_one_gpu`): `bit_counter = sample*nbit*(nchan*decimation +
  skipped) + band*nbit`. Bands are **adjacent bits of one sample word**, so
  band-on-lane makes 32 lanes read ~`32*nbit` bits — a single sector, and the
  reason L1 hit rate is 90.7% and DRAM only 21%. Sample (= channel) stride is
  `nbit*nchan` bits, so channel-on-lane spreads a warp over
  `32*nchan*nbit/8` bytes: 128 B at 16 bands (fine) but 1 KB at 128 bands with
  4 of every 32 B used (not fine).
- **Write**: `destIndex = window*fftchannels*nbands + band*fftchannels +
  channel` is **band-major**, so consecutive lanes (consecutive *bands*) are
  `fftchannels` complex apart — 2 KB at `fftchannels`=256. Each lane's 8 bytes
  lands in its own sector: 32 sectors per warp where 8 would do.

The destination layout is not free to change: `dest` is the cuFFT batched input,
which requires each FFT's `fftchannels` samples contiguous per (window, band).

So neither mapping is right for both sides, and which one loses depends on the
band count — exactly the shape-dependence to design out rather than tune for.

## The design: transpose in shared memory

One block owns a tile of `BT` bands x `CT` channels for one FFT window, and runs
in two phases over a flat `tid = threadIdx.x` block of `BT*CT` threads:

1. **Decode + rotate**, indexed `band_local = tid & (BT-1)`,
   `ch_local = tid >> log2(BT)` — band on the lane, so the read keeps its
   single-sector footprint. Result goes to `tile[band_local][ch_local]` in
   shared memory. The pcal `atomicAdd` (when `DOPCAL`) stays here, unchanged.
2. `__syncthreads()`, then **store**, re-indexed `ch_local = tid & (CT-1)`,
   `band_local = tid >> log2(CT)` — channel on the lane, so each warp writes a
   run of 32 consecutive channels = 256 contiguous bytes = 8 fully-used sectors.

`BT` and `CT` are template parameters, so both index decompositions are shifts
and masks, not integer division.

### Tile policy

Block size is fixed at **256 threads** with `CT = 256/BT`, and `BT` is chosen
from the band count:

| `numrecordedbands` | `BT` | `CT` | shared | warp write run |
|---|---|---|---|---|
| 1, or any odd count | 1 | 256 | 2.0 KB | 256 B |
| 2, 6, 10, ... (2 but not 4) | 2 | 128 | 2.1 KB | 256 B |
| 4, 12, 20, ... (4 but not 8) | 4 | 64 | 2.1 KB | 256 B |
| any multiple of 8 (8, 16, 32, 64, 128) | 8 | 32 | 2.1 KB | 256 B |

`BT` = the largest power of two <= 8 that **divides** `numrecordedbands`
(implemented; a plain `pow2_floor` rule was measured first and cost 0-3% at 3
and 6 bands, where it left a quarter of the band slots idle). Four instantiations
(x real/complex x pcal/no-pcal = 16 kernels). Every combination writes 256 B
per warp, so the write is optimally coalesced at **every** shape — that is the
point of the design. Grid becomes
`(numBufferedFFTs, ceil(fftchannels/CT), ceil(numrecordedbands/BT))`;
`gridDim.x` stays `numBufferedFFTs` because the kernel's
`index = fftloop*gridDim.x + subloopindex + startblock` depends on it.

Consequences of 256 threads instead of 1024: `Block Limit Warps` goes from 1 to
**4** resident blocks per SM (4 x 8 warps = the SM's 32-warp budget, i.e. the
ceiling), with shared memory (~2 KB) and registers leaving that limit intact - so
the latency this kernel is stalled on finally has other warps to hide behind.
Measured: achieved occupancy 84.2% -> 95.4%.

### Shape robustness (the first acceptance condition)

- **Channels 64 -> 16k**: only the grid's y extent changes. `CT` never exceeds
  256, so the channel tile is a fixed small window regardless. Where
  `fftchannels < CT` (possible only for `BT`=1 or 2, i.e. one or two bands),
  the surplus threads are predicated off; that is a tiny job in absolute terms.
- **Bands 1 -> 128**: only the grid's z extent changes. At `BT`=1 the transpose
  is an identity (band-on-lane and channel-on-lane coincide) and the cost is one
  shared round-trip — the case that is *already* coalesced today, so this is the
  one shape where the design must be shown not to regress, and it is measured
  below.
- **Partial tiles**: bands that are not a multiple of `BT` (3, 6, 12, ...) and
  channels not a multiple of `CT` leave part of the last tile idle. Handled by
  predicating the decode and the store, **not** by returning early — an early
  return before `__syncthreads()` would be undefined behaviour. The block-uniform
  early returns at the top of the kernel (invalid window, past `numblocks`) are
  safe because they depend only on `blockIdx.x`.
- Bank conflicts: rows are padded by `PAD = 16/BT` elements. `CT+1` - the
  reflex choice, and what was implemented first - was **wrong**: an 8-byte
  shared access is serviced in half-warps of 16 lanes, and with
  `lane = band + BT*ch` the banks tile only if the row stride is congruent to
  `16/BT` mod 16. ncu measured 640k conflicts across 320k store requests with
  `CT+1`, and 0 with `CT + 16/BT`. Verified, not assumed - and it made no
  difference to the time (see the results), because the kernel is L1TEX-latency
  bound rather than shared-throughput bound.

### Architecture robustness (the second acceptance condition)

The win is removing excessive sectors and raising blocks-per-SM. Both are
architecture-independent properties of NVIDIA memory hierarchies (32-byte
sectors, 128-byte cache lines, per-SM warp budgets) rather than Turing quirks,
and neither depends on FP64 rate, L2 size, or bandwidth. The cost is per element
one shared store, one shared load and one `__syncthreads()` per tile — visible
only if a card is instruction-issue-bound on this kernel, which nothing measured
so far is.

That argument is not a measurement, so:

- the tiled path is gated by **`DIFX_GPU_FRINGE_TILE`** (default on, `=0`
  selects today's kernel unchanged), so any site can A/B it and revert without a
  rebuild, and
- it is not claimed as a general win until the A100 leg below runs.

## Test plan

**Correctness** (must PASS before anything else): `tests/Synthetic/run-local.sh`,
every GPU-eligible scenario, both `DIFX_GPU_PIPELINE` modes, tiled and legacy
paths. This covers the real and complex twins and 1-4 band shapes. Correctness
of the transpose is a pure indexing property, so a mismatch shows up as garbage,
not as rounding.

**Shape sweep** (the first acceptance condition). A DiFX job per shape would
mean vex surgery per shape, so this is done by `tests/FakeData/fringetile-sweep.{cu,sh}`
instead: it `#include`s `gpudecode.cu` and drives the **real** launcher over a
parameter grid, so there is no second copy of the kernel to drift. Because both
paths do identical arithmetic in identical order, a per-shape output hash is an
exact correctness check as well - which is how the partial-tile and odd-band
cases are covered. Shapes:

| bands x channels | what it covers |
|---|---|
| 1 x 4096 | single band; the shape today already coalesces — must not regress |
| 2 x 2048 | `BT`=2 |
| 6 x 512 | **non-power-of-2 bands**: partial band tile |
| 16 x 256 | the profiled benchmark shape |
| 64 x 128 | many bands: where the legacy read mapping pays off most |

Metrics per run: duration, excessive sectors, L1TEX-scoreboard stall share,
achieved occupancy, shared bank conflicts.

**Ledger**: `tests/FakeData/run-bench.sh` T5-T1, best of 3, into
`BENCHMARKS.md` as usual.

**A100 leg** (the second acceptance condition): the same tiled/legacy A/B as a
cluster sbatch on tooarrana, `--cpus-per-task=5 --nodes=1` per the standing
rules. Until it reports, the change is 2070-verified only.

## Risks

| risk | handling |
|---|---|
| early return before `__syncthreads()` | per-thread work is predicated, never returned from; only block-uniform returns precede the barrier |
| `numrecordedbands` was read from `blockDim.x` | now an explicit kernel argument — the tiled block no longer has band as `blockDim.x` |
| shared-memory cost at `BT`=1 | measured explicitly as the 1 x 4096 sweep row |
| bank conflicts from the transpose | padding + ncu verification |
| divergence from the CPU path | none: no arithmetic changes, only indexing and staging |
| pcal path untested (`phaseCalInt=0` everywhere) | unchanged code, kept in phase 1; still review-validated only (`gpu-plan.md` queue item 7) |

## Results (RTX 2070 SUPER, 2026-08-27)

| | untiled | tiled |
|---|---|---|
| kernel duration (16 bands x 256 ch) | 948 us | **703 us** (1.35x) |
| Compute (SM) throughput | 53.6% | 72.4% |
| DRAM throughput | 21.1% | 30.9% |
| achieved occupancy | 84.2% | 95.4% |
| `Block Limit Warps` | 1 | 4 |
| excessive sectors | 29% | none reported |
| shared bank conflicts | n/a | 0 (640k with the `CT+1` pad) |
| T5-T1, same binary and session | 10.9 s | **10.1 s** (7.3%) |

Shape sweep (`tests/FakeData/fringetile-sweep.sh 2500 30`, 15 shapes, both
sampling modes): **~1.2x - 1.5x real, ~1.15x - 1.7x complex
(`SWEEP_COMPLEX=1`), no shape slower in either**, `dest` **bit-identical at
every shape**, and the phase-cal bins (`SWEEP_PCAL=1`, compared with a tolerance
because `atomicAdd` order changes) agreeing to 3.1e-07. Best is 1.5-1.7x at
1x4096 - the shape that already coalesced, which gains from the occupancy half
alone. The narrowest shape, 1 band x 64 channels, is 1.00x: `CT` is clamped to
the channel count and `BT`=1 skips the transpose, so the tiled path reduces to
the same geometry there.

Two policy details exist because the first version of each was measured and
found wanting:

- **`CT` is clamped** to `min(256/BT, pow2_floor(fftchannels))`, floor 32. With
  `CT` fixed at `256/BT`, a job narrower than the tile left most of each block
  masked off - 1 band x 64 channels measured **0.46x**, because the untiled
  geometry sizes its block to the channel count and is optimal there.
- **`BT`=1 stores straight to global**, skipping the shared staging and the
  barrier, because at one band per tile the two lane decompositions coincide and
  the transpose is an identity. Worth ~2% at the narrowest shapes. Correctness: every
GPU-eligible Synthetic scenario PASSes CPU-vs-GPU in both `DIFX_GPU_PIPELINE`
modes, on both paths.

One measurement caution for whoever repeats this: at `--launch-count`/rep counts
below ~30 the 16x4096 row swings by up to 18% on the untiled side (warm-up), and
at `numBufferedFFTs` = 10 rather than the real job's 2500 every shape looks like
~1.0x because the launch is latency-dominated. Use the sweep's defaults.
