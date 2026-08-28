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

## Index

Sections 1-9 and 11 are settled history and live in
[`gpu-changes-archive.md`](gpu-changes-archive.md); the numbering is continuous
across both files, so a reference to `§8` still means §8. The sections below
are the ones that still bear on current work.

- **§1. Foundation** *(archive)* — the port's starting point: GPUCore/GPUMode, what was inherited and what was rewritten
- **§2. Pulse cal (PCal) on the GPU** *(archive)* — phase cal on the GPU, and the PCal interface question it left open
- **§3. The correctness campaign** *(archive)* — the correctness campaign - every CPU-vs-GPU divergence found and fixed, with the debugging lenses that found them
- **§4. Debug and test infrastructure** *(archive)* — the debug and test scaffolding (DIFX_GPU_PIPELINE, WEIGHT/SPEC debug, the Synthetic harness)
- **§5. Performance phase** *(archive)* — the first performance push: what was tried, what landed, what was rejected
- **§6. Multi-station, multi-subband correctness coverage** *(archive)* — multi-station, multi-subband correctness coverage (the `multi` scenario)
- **§7. Desktop bring-up (2026-07-17/18)** *(archive)* — desktop bring-up on ar313 (2026-07-17/18)
- **§8. Host-tail overlap (intra-subint half-split, 2026-07-21)** *(archive)* — host-tail overlap, the intra-subint half-split
- **§9. Per-station unpack drain removed; RING-deep host staging (2026-07-22)** *(archive)* — per-station unpack drain removed; RING-deep host staging
- **§10. The real bottleneck: DataStream interlaced-VDIF corner-turn (2026-07-22)** — the DataStream interlaced-VDIF corner-turn - **still the real production bottleneck**
- **§11. Build: nvcc header-dependency tracking; item-4 investigation (2026-07-22)** *(archive)* — nvcc header-dependency tracking, and the stale-object class of bug it closed
- **§12. Fused unpack + fringe rotation (2026-07-23)** — fused unpack + fringe rotation (the largest single win)
- **§13. Total-weight reduction fused into set_weights; gpu_sum_weights deleted (2026-07-23)** — total-weight reduction folded into gpu_set_weights
- **§14. Input-transfer speed: contention, not NUMA placement (2026-08-26)** — input-transfer speed: contention, not NUMA placement - **source of the standing benchmark rules**
- **§15. The materialised conjugate array is gone (2026-08-26)** — the materialised conjugate array deleted
- **§16. Tiled fused fringe: a shared-memory transpose for the write (2026-08-27)** — tiled fused fringe: a shared-memory transpose for the write

## 10. The real bottleneck: DataStream interlaced-VDIF corner-turn (2026-07-22)

Whole-pipeline profiling (Manager + DataStream + Core, not just the Core)
overturned the framing of sections 8-9: the Core-side de-serialization was
all correct, but it was **not** the bottleneck. The A100 after-fix reprofile
confirmed the unpack-drain fix held (no `cudaStreamSynchronize` storm) yet the
GPU was still ~42% idle, with the Core's compute thread blocked ~8 s on the
procslot mutex *waiting for input data*.

Profiling the other ranks found why: the **Manager is pure wait** (epoll/poll),
and the **DataStream spends ~93% of its busy CPU in `cornerturn_16thread_2bit`
+ memcpy** — `VDIFMuxer::multiplex` demultiplexing 16 interlaced VDIF threads
into one — at only **~1.7 Gbps/core**, at/below the 2 Gbps/station record rate.
So the GPU is simply faster than the DataStream can feed it (a 2-station job:
GPU 8 s vs CPU 35 s). MPI transport is not the limit: `process_vm_readv` is
0.45 s at 2 stations (the 13.6 s seen at 10 stations was desktop
oversubscription), and forcing CMA/vader/TCP transports changed nothing.

Proven by switching the fake data to **single-thread VDIF** (no interlacing →
no corner-turn): a 2-station uncontended GPU job dropped **8.0 s → 4.8 s
(~40%)** with identical output. That required a mark5access fix — VDIF had no
`genheaders`, so `Mk5DataStream::fakeToMemory` never framed single-thread fake
VDIF (all frames invalid, 0-byte output). `mark5_format_vdif_genheaders`
(`191528ff9`, `libraries/mark5access/mark5access/format_vdif.c`) writes valid
headers (invalid bit clear, monotonic time). For fake-data benchmarking,
rewrite the `.input` after vex2difx with
`sed -i -E 's|(DATA FORMAT:[[:space:]]+)INTERLACEDVDIF/[0-9:]+|\1VDIF|'`
(keep DATA FRAME SIZE) — now the default in both `benchprof-profile.sbatch`
(which also widened its profile window ~4 s → ~20 s so startup/teardown is a
small fraction of the headline wall) and `run-bench.sh` (`SINGLE_THREAD_VDIF=0`
keeps the interlaced demux for comparison).

**Confirmed on both machines (2026-07-22):** A100 GPU idle 41.9% → 7.2%, wall
12.2 → 7.9 s (~35%), kernel busy unchanged (see BENCHMARKS.md A100 §10 rows).
Desktop T5-T1 (best-of-3) 23.2 s interlaced → 14.0 s single-thread (~40%) — the
corner-turn cost the oversubscribed desktop too, not just the A100. The GPU is
now compute-bound; next kernel targets: `gpu_resultsrotatorMultiply` (28.9%),
`gpu_unpack` (24.5%), the `<<<1,1>>>` `gpu_sum_weights` (5.8%).

The production fix (real interlaced recordings can't just be relabelled) is a
longer-term goal: an unpacker that reorders the per-thread frames into time
order (pure memcpy) + GPU channel de-interleave, instead of the CPU bit-banging
multiplex. See gpu-plan.md (Longer-term).

## 12. Fused unpack + fringe rotation (2026-07-23)

**What changed (gpu-plan.md item 1).** The GPU station path used to run two
kernels back to back: `gpu_unpack` decoded every sample into a global
per-band unpacked buffer (`unpackeddata_gpu` / `complex_unpackeddata_gpu`),
and `gpu_fringeRotation` immediately read those samples straight back to apply
the fringe rotator. That is a full global-memory round-trip of the largest
data array in the pipeline, and `gpu_unpack` was the single largest kernel on
the cluster (24.5–34.9% of GPU time; ~20% on the 2070). The two kernels are
now **fused** into one (`gpu_fused_fringe` / `gpu_fused_fringe_complex`, in
`gpudecode.cu`): each thread owns one (FFT window, band, channel), decodes
exactly the one sample it needs directly from the packed VDIF frame payload
into a register, applies the precomputed rotator and writes the complex FFT
input. **No unpacked buffer, no round-trip** — the unpacked-sample buffers and
their pointer arrays are gone entirely (also freeing that VRAM).

**Phase cal fused in too.** The standalone `gpu_pcalextraction` kernel read
the same raw unpacked samples; its only real work was folding each sample into
a phase-cal bin (the offset/phase assembly is host-side). That folding is now
done inside the fused kernel, gated by a `template<bool DOPCAL>` so the common
(pcal-off) instantiation compiles all of it away. `gpu_unpack`, the unpacked
buffers, the tail-zeroing memset, and the standalone pcal kernels were all
deleted, so there is no reason left to keep or optimise the old unpack layout
(gpu-plan.md item 1's "templating the unpack" and the layout half of item 8
are moot).

**Design / structure.**
- `decode_one_gpu` / `decode_one_complex_gpu` (device primitives) are the
  per-(band, sample) decode extracted from the old `mk5_decode_sample_gpu`;
  same bit-offset arithmetic, so the decoded values are identical.
- `gpu_blank_frames` (one thread per frame, runs the existing
  `blanker_vdif_gpu`) produces `valid_frames`, which used to be a by-product
  of `gpu_unpack`. `gpu_set_weights` and the fused kernel both read it.
- Blanking/tail parity: a sample whose frame is invalid (`!valid_frames`) or
  past the delivered data (`frame >= framestounpack`) decodes to 0 — exactly
  what the old per-frame blankzone zeroing and the explicit unpacked-tail
  memset produced.
- The fused launch needs the `mark5_stream`/packed data, so it lives in
  `Mk5_GPUMode` behind two new virtuals (`blankFrames`, `launchFusedRotate`);
  the format-agnostic per-(window,band) coefficient precompute
  (`gpu_precompute_fringe_rotator`) stays in `GPUMode::fringeRotation`. This
  matches the direction of the planned Mk5Mode virtual-hierarchy refactor.
- Kernels + the four template instantiations live in `gpudecode.cu` (with the
  mark5access includes and decode LUTs); `mk5mode_gpu.cu` calls them through
  plain `launch_*` wrappers declared in `gpudecode.cuh`.

**Correctness.** run-local.sh usb / usb-complex / complex-complex / multi PASS
CPU-vs-GPU in BOTH pipeline modes, and PASS again under the
`DIFX_GPU_WEIGHTS_HOST=1` fallback (which `blankFrames` still lands
`valid_frames` on the host for). An independent line-by-line parity review
against the old kernels found no correctness divergence (bit layout, per-frame
validity, tail zeroing, rotator constant `TWO_PI`, and the pcal bin residue
all match). The pcal-fused path itself is exercised by no test (phaseCalInt=0
everywhere, as before) and is validated by construction/review only — a
phaseCalInt>0 regression test is the agreed follow-up.

**Benchmark (2070, T5−T1, single-thread VDIF).** 14.0 s → **12.3 s (~12%)** —
the busy-half win expected from removing the round-trip. The A100 gain should
be larger since `gpu_unpack` was a bigger fraction there. See BENCHMARKS.md.

**Not addressed (deliberately):** the fused decode still reads the packed byte
per thread (good locality: consecutive channels are consecutive time samples
in one band's bitstream) but the FP16 path (item 2), per-sample precision drop
(item 3), and the occupancy audit (item 8, e.g. `gpu_sum_weights` `<<<1,1>>>`)
remain open.

## 13. Total-weight reduction fused into set_weights; gpu_sum_weights deleted (2026-07-23)

**Motivation.** The 2026-07-23 A100 profile of the fused-unpack binary showed
`gpu_sum_weights` — the `<<<1,1>>>` single-thread reduction of the per-window
`gDataWeights[w]` to the scalar `gTotalWeight` — at a startling **7.7% of GPU
busy** (373 ms / 4952 launches). Once unpack+fringe shrank (§12), this trivial
serial kernel became a top-5 cost purely from single-thread execution + launch
overhead.

**What changed.** The total weight is now a **free by-product of
`gpu_set_weights`**, which already runs one thread per FFT window and computes
each `dataweight[w]`: each window thread `atomicAdd`s its weight into
`gTotalWeight` (zeroed with a `cudaMemsetAsync` on the stream just before the
launch). `gpu_sum_weights` is deleted — no separate launch, no serial loop.
Better than parallelising it into a standalone reduction kernel, which would
still be an extra launch. The **compute** moved to tofft; the `gTotalWeight`
**D2H stays in afterfft** (the tail-overlap constraint — the pinned host mirror
must be written at drain time), and single-stream ordering keeps the device
scalar stable between the tofft accumulate and the afterfft D2H, exactly as
before.

**Behaviour.** The atomic accumulation reorders the sum vs the old window-order
loop, so the AC per-band weight total is now FP-level rather than bit-identical
for multi-occurrence bands — within the acceptance bar (the final visibilities
are not bit-reproducible run-to-run anyway). Device-weights and
`DIFX_GPU_WEIGHTS_HOST` fallback are otherwise untouched (the fallback never
entered this branch; invalid subints never read `gTotalWeight`).

**Correctness.** run-local.sh usb / usb-complex / complex-complex / multi PASS
CPU-vs-GPU in BOTH pipeline modes, and PASS under the host-weights fallback.

**Benchmark (2070, T5−T1).** 12.3 → **11.8 s (~4%)**. On the A100 the win
should be ~its 7.7% GPU-busy share (2070 has fewer/cheaper serial-kernel
stalls). See BENCHMARKS.md. Part of gpu-plan.md item 6 (the occupancy audit);
the rest of that audit — ad-hoc launch dims elsewhere — remains open.

## 14. Input-transfer speed: contention, not NUMA placement (2026-08-26)

**The anomaly.** The A100 profile of the `gpu_sum_weights` removal (§13) showed
the expected −7.4% of kernel busy but a ~15% better wall. The excess was in the
input transfers: the same ~12.9 GB of pinned H2D took **1444.7 ms in one
capture and 683.0 ms in the next** (8.9 vs 18.9 GB/s) at an unchanged copy
count, for a binary whose transfer path had not been touched. Kernel times
agreed to <0.5% across the two, so only the copy engine differed. A 2× swing in
a quantity worth ~10% of wall, invisible to everything in the correlator, makes
every wall number in the ledger suspect.

**First hypothesis (WRONG): NUMA placement.** Each capture's `SCHED_EVENTS`
records the one CPU the Core rank ran on (the sbatch pins one core per rank),
and the slow run sat on a different CPU from the fast ones. The mechanism was
plausible and specific: `Core`'s receive buffers are allocated — and therefore
first-touched, fixing their NUMA node — by the base constructor, and `GPUCore`
then `cudaHostRegister`s them. Page-locking does not move pages, so a rank
placed remotely from its GPU would feed every transfer across the interconnect.

**What killed it.** Adding the placement report below and re-running produced a
run that was *definitively* mis-placed (GPU on NUMA node 1, rank on node 2) and
posted the **fastest** H2D of any capture, 22.1 GB/s. With the node topology in
hand (NPS4: 8 NUMA nodes × 8 CPUs, distance 12 within a socket, 32 across;
`nvidia-smi topo -m` puts GPU0 on node 1, CPUs 11-14) the earlier captures
re-map to contradict the theory outright — two runs on the same GPU, both
*cross-socket*, differ 2×:

| capture | GPU node | rank node | distance | pinned H2D |
|---|---|---|---|---|
| gina6 | 1 | 7 | 32 (cross-socket) | 19.3 GB/s |
| gina8 | 1 | 6 | 32 (cross-socket) | **8.9 GB/s** |
| gina4 | 1 | 2 | 12 (same socket) | **22.1 GB/s** |

**The actual cause: contention.** Per-copy distributions of the 2.59 MB input
transfers show every node reaching an identical **0.099 ms floor = 26.1 GB/s**,
the PCIe gen4 x16 line rate. There is no per-node bandwidth cap and no distance
penalty. What differs is the tail:

| | min | p50 | p90 | max |
|---|---|---|---|---|
| gina8 (8.9 GB/s aggregate) | 26.1 GB/s | 18.3 | **6.8** | 0.6 |
| gina15 (18.9) | 26.1 GB/s | 20.8 | 18.4 | 1.6 |
| gina4 (22.1) | 26.1 GB/s | 25.0 | 22.7 | 8.4 |

Stalled copies, not slow ones — interference from a co-tenant job. No benchmark
run has ever been isolated: `a55b6c12c` ("can't use exclusive") dropped the
`--exclusive` request back in July because tooarrana rejects the flag outright,
so every profile in the ledger has been sharing its node. This was not a
regression that crept in — the isolation was never there.

**What changed.**
- Both cluster sbatch scripts document **two modes**: `sbatch
  --cpus-per-task=5 <script>` for anything that goes in BENCHMARKS.md, plain
  `sbatch <script>` for a quick look. Each run prints its own verdict
  (`node ownership: N/M cores`) plus `numactl -H` and `nvidia-smi topo -m`, so
  a number can be trusted or discarded after the fact rather than from memory
  of how it was submitted.
  **Correction (2026-08-26, same day):** this first said `--exclusive`, which
  tooarrana rejects outright ("Exclusive option not permitted, please request
  the number of cores you need per task with `--cpus-per-task=x` instead").
  Reserving most of the node is the supported equivalent: `--cpus-per-task=5`
  x 12 tasks = 60 of gina's 64 cores. Widening it does **not** start more
  mpifxcorr ranks — the rank count is `--ntasks` and the roles come from the
  `.input`'s ACTIVE DATASTREAMS (rank 0 manager, 1-10 datastreams, 11 Core) —
  so the extra cores are simply reserved and mostly idle. It does, however,
  need **`--nodes=1`** alongside it: the first wide submission was scheduled
  across *two* nodes, and none of these scripts had ever constrained the node
  count. That matters twice over — `--gres` is a per-node request, so a 2-node
  allocation reserves a GPU on each and uses one; and DataStream ranks on the
  far node stop reaching the Core through intra-node CMA (`process_vm_readv`)
  and go over the fabric, changing the very data-delivery path these
  benchmarks exist to hold constant. All three sbatch scripts now pin
  `--nodes=1` and warn in the log if the allocation spanned more than one node.
  (The captures analysed so far were single-node — their Core ranks show
  `process_vm_readv`, which is CMA and intra-node only — so the existing ledger
  rows are unaffected.)
- `--gres-flags=enforce-binding` is kept in both. On these nodes it is
  measurably worth nothing, but it costs nothing and would matter on hardware
  with a slower cross-socket link.
- `sysutil.{h,cpp}` gained Linux/sysfs placement enquiry — `getCpuAffinity`,
  `getNumaNodesOfCpus`, `getPciNumaNode`, plus `parseIdList`/`formatIdList`.
  They report "unknown" rather than failing where the platform does not expose
  the information, and read sysfs with a plain `ifstream` (not `ifstreamOpen`,
  whose NFS retry-and-warn loop is wrong for a legitimately-absent attribute).
- `GPUCore` logs one line per Core rank: `GPU Core 11 placement: host gina4,
  GPU 0 (0000:41:00.0) on NUMA node 1; rank on CPU 21, NUMA node 2`. It is a
  plain report, **not a warning** — the measurements above show NUMA distance
  does not predict transfer speed on this hardware, so a warning would be
  crying wolf. It earns its place by putting the placement in every difxlog.

**Correctness.** Logging and job-script changes only; no correlator behaviour is
affected. run-local.sh usb / usb-complex / complex-complex / multi PASS
CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes.

**Method worth reusing.** Aggregate GB/s hides the mechanism; the per-copy
percentiles are what separated "this link is slower" from "these copies are
being stalled", and they came from captures already on disk. Any future
transfer anomaly should go straight to the distribution, not the mean.

**Consequence for the ledger.** The two 2026-07-23 A100 rows are not comparable
on wall time — one of them was a contended run. Only the kernel-busy figures
from that pair mean anything. BENCHMARKS.md carries the caveat.

## 15. The materialised conjugate array is gone (2026-08-26)

**What it was.** The pipeline kept a second full array of spectra,
`conj_fftd_gpu`, holding nothing but `conj(fftd_gpu)`, so the XMAC and the
cross-polarisation autocorrelations could read a pre-conjugated operand. The
rationale was reuse: each station's conjugate is read by up to Nstations−1
baselines, so conjugating once looked cheaper than conjugating per use.

**Why that reasoning was wrong.** Conjugating inside the multiply is free:

```
(a+bi)(c+di) = (ac − bd) + (ad + bc)i      4 mul, 2 add
(a+bi)(c−di) = (ac + bd) + (bc − ad)i      4 mul, 2 add
```

Same instruction count either way — only the signs differ, and they fold into
the FMA sign bits. The array was therefore paying, for nothing:

- **an 8-byte global write per spectral point** in `gpu_resultsrotatorMultiply`
  — 82 MB per launch of its ~205 MB, i.e. ~40% of that kernel's DRAM traffic;
- **82 MB of VRAM per Mode** (820 MB across benchprof's 10 datastreams);
- **double the XMAC's working set** — 164 MB of spectra rather than 82 MB, on a
  kernel measured at an effective 3.6 TB/s, i.e. served out of L2, where
  footprint is precisely what matters;
- a per-subint memset of the whole array on the invalid-subint path.

**What changed.** `cuCmulConjf(a, b)` = `a * conj(b)` now lives in
`gpumode_kernels.cuh`, shared by both translation units rather than duplicated
(the existing `atomicAddFloatComplex`/`atomicAddFloatComplex1` pair is a
standing example of what not to do again). Then:

- the XMAC's `m2` pointer array points at `fftd_gpu`, and its product becomes
  `cuCmulConjf(v1, v2)` — same orientation, `v1 * conj(v2)`;
- the **band** autocorrelation is taken from the register the rotator already
  holds, as `v.x*v.x + v.y*v.y`, accumulated into the real component only.
  `v * conj(v)` is real by construction, so the second atomic per element was
  adding a provably-zero imaginary part — this is the real-only optimisation
  identified but left unapplied during the July item-4 investigation;
- the **cross-pol** autocorrelations read both operands from `fftd_gpu`. Each
  value was written by the same thread earlier in its own band loop (same
  window, same channel, different band), so there is no new race — and they get
  cheaper too: two cold reads from two different arrays become two reads from
  one array the band loop has just touched.

**Numerics.** The expressions are unchanged, so differences are at most 1 ulp
from FMA grouping — the same class as the FMA-contraction differences already
documented for this project, and well inside the acceptance bar. Verified
bit-identical in a non-contracted host build across generic, sign-flipped and
subnormal operands. The one real change: the band autocorrelation's imaginary
component was accumulating FMA rounding residue (~1e-8 of the real part) around
a physically-zero value, and is now exactly zero.

**Measured (RTX 2070, benchprof2, 2500 windows x 16 bands x 128 channels).**
`gpu_resultsrotatorMultiply` **536.3 → 396.9 us/call (−26%)**; total kernel busy
1662.9 → 1557.9 ms (−6.3%). T5−T1 **11.8 → 10.8 s (−8.5%)**. Device-memory
estimate for the 2-station `usb` scenario 197.09 → 132.12 MB of modes, exactly
the third of the triple-array term that was removed. The A100 should gain more:
that kernel is 42-44% of its kernel busy, and the probes put traffic
sensitivity at ~100 us per 82 MB removed.

The XMAC itself was flat on the 2070 (33.3 → 34.0 us/call) — it is only 7% of
kernel busy at this scale, so the halved working set had nothing to show. Worth
re-checking on the A100, where it is 23%.

**Also removed, because this change made them wrong or dead:**
`_gpu_processBaselineBased` (an unlaunched XMAC fallback whose `cuCmulf` was
only correct while `m2` pointed at a pre-conjugated array — reviving it after
this change would have silently produced `V1*V2`), and the
`getGpuConjugatedFreqs`/`getGpuConjugatedFreqsHost` virtuals, whose only
overrides went with the array and which would otherwise have returned a silent
`nullptr` to any future caller.

**Correctness.** run-local.sh usb / usb-complex / complex-complex / multi PASS
CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes, and PASS under
`DIFX_GPU_WEIGHTS_HOST=1` (verified genuinely engaged — the fallback's event
timers report non-zero where the device path reports zeros).

Design and the remaining two steps: `docs/gpu-autocorr-design.md`.

## 16. Tiled fused fringe: a shared-memory transpose for the write (2026-08-27)

`ncu` became usable on ar313 on 2026-08-27 (reboot picked up
`NVreg_RestrictProfilingToAdminUsers=0`), and the first thing it said was that
the 2070's kernel mix is nothing like the A100's: `gpu_fused_fringe` **46.5%**,
`vector_fft` 22.8%, `gpu_resultsrotatorMultiply` 20.3%, xmac 7.0%, against the
A100's 21/11/42/23. So the biggest kernel on this card had never been examined,
and it turned out to be **latency-bound on an uncoalesced write**: 29% excessive
sectors, and 58% of 38.6 warp-cycles-per-instruction stalled on L1TEX
scoreboard.

The cause is a genuine conflict between the kernel's two halves rather than an
oversight. `decode_one_gpu` reads `band*nbit` bits into one sample word, so
band-on-lane makes the decode read a single sector per warp (hence L1 hit 91%,
DRAM 21%); but `dest` is band-major, because it is the cuFFT batched input, so
band-on-lane scatters each warp's 8-byte stores across as many sectors as there
are lanes. Neither mapping suits both sides, and which one loses depends on the
band count.

**The fix.** One block now owns `BT` bands x `CT` channels of one FFT window,
with `BT*CT` = 256 threads. Phase 1 decodes and rotates with band on the lane
into a padded shared tile; `__syncthreads()`; phase 2 stores with channel on the
lane, so each warp writes 256 contiguous bytes. `BT` is the largest power of two
<= 8 that *divides* the band count (a divisor, so no block carries idle band
slots - a plain `pow2_floor` rule cost 0-3% at 3 and 6 bands), and `CT = 256/BT`,
so every shape gets the same fully-coalesced 256-byte store run. Dropping the
block from 1024 to 256 threads also took `Block Limit Warps` from **1 to 4**
resident blocks per SM, which is the second half of the win: the remaining
latency now has other warps to hide behind. Gated by
**`DIFX_GPU_FRINGE_TILE`** (default on, `=0` keeps the untiled kernels), and the
launch geometry moved out of `GPUMode::fringeRotation` into `launch_fused_fringe`
where the kernel that needs it lives.

**Results on the 2070.** Kernel 948 -> 703 us/call (**1.35x**), achieved
occupancy 84.2 -> 95.4%, Compute (SM) throughput 53.6 -> 72.4%. End to end,
same binary, same session: T5-T1 **10.9 -> 10.1 s (7.3%)**.

A shared-memory detail worth recording, because the obvious choice was wrong:
padding rows by **one** complex element (the reflex fix) leaves an 8-byte-access
row stride of 1 mod 16 and a **2-way bank conflict on every phase-1 store** -
ncu counted 640k conflicts across 320k store requests. An 8-byte shared access
is serviced in half-warps of 16 lanes, and with `lane = band + BT*ch` the banks
tile only if the row stride is congruent to `16/BT` mod 16, i.e. `PAD = 16/BT`.
With that, conflicts are **0** - but the kernel time did not move (703 us either
way), because it is L1TEX-latency bound, not shared-throughput bound. Kept
anyway: it is free, and a card whose shared throughput binds would pay for it.

**Verification.** All GPU-eligible Synthetic scenarios PASS CPU-vs-GPU in both
`DIFX_GPU_PIPELINE` modes, tiled and untiled. Beyond that, the shape question
("is this a win at every (bands, channels), or only at the benchmark's shape?")
does not fit the DiFX test harness - a job per shape would mean vex surgery per
shape - so `tests/FakeData/fringetile-sweep.{cu,sh}` drives the *real* launcher
(it `#include`s `gpudecode.cu`, so there is no second copy to drift) over 13
shapes from 1x4096 to 128x128 to 16x4096. Both paths do identical arithmetic in identical
order **into `dest`**, so a hash per shape is an exact all-shapes correctness
check there as well as a timing harness. That argument does *not* extend to
`pcal_output`, which is accumulated with `atomicAdd`: the tiling changes the
summation order, so `SWEEP_PCAL=1` dumps the phase-cal bins and the harness
compares them with a tolerance instead (worst relative difference measured
**3.1e-07**, i.e. FP reordering). Results:

- **~1.2x - 1.5x, no shape slower**, and ~1.15x - 1.7x for the
  complex-sampled twin (`SWEEP_COMPLEX=1`, which doubles the bits per sample and
  so halves the samples per frame - a different read pattern for the same
  tiling). Worst real 1.30x (8x256, 2x2048, 4x512); best 1.54x at 1x4096 - the
  shape that was *already* coalesced, which gains from the occupancy half alone.
- `dest` **bit-identical at all 15 shapes in both sampling modes**, including the
  odd band counts (3, 6) that exercise the `BT` fallback, and the phase-cal bins
  agree to **3.1e-07** under `SWEEP_PCAL=1` - which is the only coverage the
  `DOPCAL` path has, since `phaseCalInt = 0` in every in-repo scenario.

**Two things the code review caught, both then measured rather than argued.**
First, the tile policy ignored the channel count, so a job narrower than `CT`
wasted most of each block: at 1 band x 64 channels the tiled path measured
**0.46x**. `CT` is now a template parameter clamped to
`min(256/BT, pow2_floor(fftchannels))` with a floor of 32, so a warp still stores
one contiguous run; that took the shape to 0.98x. Second, at `BT`=1 the two lane
decompositions coincide, so the transpose is an identity and the shared
round-trip is pure cost - `BT`=1 now stores straight to global and skips the
barrier, which took the narrowest shape to 1.00x. Also from the review:
`numBufferedFFTs` is passed explicitly instead of read back from `gridDim.x` (the
`index` arithmetic depended on a grid dimension by convention only), and
`DIFX_GPU_FRINGE_TILE` now disables only on an explicitly off-ish value - an
`atoi(e) == 0` test made `=true` and `=yes` silently select the *old* path.

Coverage note: the local `tests/Synthetic` v2d files are scaled down to
`nChan=1024`, and `fftchannels` is exactly what selects the grid's y extent and
the ragged tail block, so `usb` was also run end-to-end at the committed
`nChan=4096` (PASS, both pipeline modes) before this was called done.

**The A100 leg is done (2026-08-28) and the tiling wins there by more.**
Kernel-level sweep on an A100-SXM4-80GB: 1.07-1.20x at 1-8 bands, **1.48x** at
the benchmark's 16 x 256, **2.1-2.2x at 32-128 bands**, nothing slower, output
bit-identical in both sampling modes. The gain grows with band count because the
untiled lane mapping (`band + nbands*channel`) scatters a warp's stores across
more bands the more there are. So the gate stays on by default everywhere; the
wall-clock A/B that suggested otherwise is written up in `BENCHMARKS.md` as the
measurement error it was. The design argument for
architecture-independence (fewer excessive sectors, more resident blocks - both
properties of NVIDIA memory hierarchies, not of Turing) is an argument, not a
measurement, which is what the env gate is for.
