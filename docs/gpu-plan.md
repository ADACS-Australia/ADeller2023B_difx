# GPU correlator: the plan

Living plan for the GPU work on `adam-performance-gains`. Companion to
`gpu-changes.md` (what has already happened) and
`tests/FakeData/BENCHMARKS.md` (the numbers). Updated as items land.

## Where we are

Correctness is done for the USB scenarios (usb, usb-complex,
complex-complex, and the 5-station/4-subband `multi` scenario PASS
CPU-vs-GPU in both `DIFX_GPU_PIPELINE` modes; LSB deliberately
unimplemented). The current phase is **performance**: the GPU is
gap-dominated because one Core thread alternates host and GPU work.
Landed so far: Lever A (pinned input buffers, direct H2D, 2026-07-18);
de-serialization Increment 1 (set_weights on the device, 2026-07-18);
de-serialization Increment 2 (baseline weights on the device,
2026-07-20). Together these cut the T5-T1 benchmark 66.7 -> 32.4 s. The
biggest remaining host cost was `host_accumulate`'s baseline-weight loop,
now gone; what remains of item 2 is overlapping the per-datastream H2D
and host work with GPU compute.

## Work queue (ordered)

1. ~~**Multi-subband, N>2-station Synthetic scenario.**~~ DONE
   2026-07-18: the `multi` scenario — 5 stations x 4 subbands
   (test-multi.vex/v2d, 4-channel VDIF from generateVDIF) — PASSes
   CPU-vs-GPU in both pipeline modes and is wired into run-local.sh and
   run-slurm.sh (rank counts now sized per scenario).
2. **De-serialize the per-datastream loop** (the main event). Progress:
   Increment 1 (set_weights on the device, 2026-07-18) removed the
   per-datastream stream drains; Increment 2 (baseline weights on the
   device, 2026-07-20) removed `host_accumulate`'s dominant per-window
   loop — together T5-T1 66.7 -> 32.4 s. **Remaining:** overlap
   datastream j+1's input H2D and host-side work with datastream j's GPU
   compute, and overlap `host_accumulate`'s residual (autocorr flush,
   pcal) with the next subint. Machinery anticipated by Lever A: move
   input copies to a dedicated H2D stream and record `h2dInputDone`
   there (see note in `issuegpudata`); relax the per-pass
   `cudaStreamSynchronize`. Candidate within the same effort: reduce
   `fringeRotation`'s double-precision arithmetic (66% of GPU time on
   GeForce, ~25% on data-centre cards) if accuracy analysis allows.
3. **perbandweights on GPU** — currently unused on the GPU path but
   should be active, in analogy with CPUMode/Mk5Mode's interlaced-VDIF
   handling (identified in the de-serialization design review). Shape
   the device weights kernel (item 2, Increment 1) so per-(window, band)
   weights are a natural extension.
4. **LSB on GPU** — unblocks the lsb/lsb-complex/dsb scenarios.
5. **CODIF on GPU** — format-aware frame-validity hook (FIXME in
   `blanker_vdif_gpu`) + widen the `getMode` format gate. MUST add the
   `datalengthbytes <= getMaxDataBytes` clamp in `process_gpu` first:
   non-VDIF datastreams keep the base class's guard-scaled sendbytes,
   which can exceed the packed-data buffer (latent, unreachable today).

## Standing process (adopted 2026-07-18)

- **Benchmark ledger**: every landed change gets a
  `tests/FakeData/run-bench.sh` run (metric T5−T1, best-of-3) recorded
  in `BENCHMARKS.md`. Publishing the ledger to the fork's GitHub wiki is
  intended once the wiki is initialised (the wiki is a git repo:
  `ADACS-Australia/ADeller2023B_difx.wiki.git`).
- **Prose documentation**: `gpu-changes.md` is extended with every
  substantive change, at the same time as the change.
- **Full diff review before every commit.**

## Future infrastructure

- **Continuous integration** on ar313 (and/or the cluster): the natural
  shape is a self-hosted GitHub Actions runner on ar313 executing
  `tests/Synthetic/run-local.sh` (correctness gate) and `run-bench.sh`
  (ledger row) on push/nightly; a cluster leg could submit
  `run-slurm.sh` via a SLURM-side runner for scale testing. Needs: runner
  install (interactive auth), deciding failure notification, and keeping
  GPU jobs serialized on the single 2070.

## Refactor (before merging back into DiFX main)

The fork has diverged significantly from upstream; a structural clean-up
will ease the eventual merge and put GPU code where it belongs:

- `Core` becomes a virtual base with `CPUCore` and `GPUCore` as the
  concrete implementations (today GPUCore inherits a concrete CPU Core).
- `Mode` becomes a virtual base; `CPUMode` concrete; `Mk5Mode` a virtual
  inheritor covering the frame-based formats; `CPUMk5Mode` and
  `GPUMk5Mode` concrete under it. Rationale: the GPU path is restricted
  to frame-based formats by design (no plans for non-frame formats), so
  the GPU implementation belongs under the frame-based branch of the
  hierarchy rather than shadowing all of Mode.
- Same pass: decide keep/remove for the WDEBUG/HDRDEBUG/SPECDEBUG and
  NVTX scaffolding, unify the duplicated pcal reset logic (the agreed
  PCal "ingest folded bins" interface), and clear the latent-issues list
  (heterogeneous-datastream XMAC plans, NOT_SUPPORTED guard for multiple
  phase centres/pulsar binning on GPU, per-call stream create/destroy in
  `processgpudata`).

## Longer-term / opportunistic

- **Subints larger than GPU memory**: today a subint's full set of FFT
  windows must fit on the device, so large station counts or data rates
  force short subints — at the cost of a higher visibility rate from the
  Cores back to the manager. Broaden GPUMode/GPUCore to slice a subint
  into chunks of FFT windows processed a subset at a time (the dormant
  fftloops machinery is the natural seam). Interacts with the VRAM
  budget check, cfg_numBufferedFFTs sizing, and the numfftloops == 1
  assumptions documented in de-serialization Increment 1.
- **Tensor cores for the XMAC**: the cross-multiply-accumulate is a
  complex GEMM at heart, a natural tensor-core target (Turing: FP16
  multiply / FP32 accumulate; data-centre cards add TF32/FP64 paths).
  Needs an accuracy assessment against the correlator's dynamic range.
- NUMA/affinity audit (mattered on the cluster, less on the desktop).
- Profile the fftsPerChunk XMAC grid split if atomics show up.
- GPU pcal regression coverage (currently untested by diffDiFX).
