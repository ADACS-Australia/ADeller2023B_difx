# Design: de-serializing the GPU correlator's subint loop

Status: v2 reviewed and approved 2026-07-18; **Increment 1 LANDED**
(same day: fractional weights bit-identical to CPU, 8/8 scenarios PASS,
bench 66.7 -> 62.4 s, station_processing 83.6 -> 63.6 ms/subint).
Increment 2 is next. Companion to gpu-plan.md item 2.
v2 replaces v1's latency-hiding approach (phase-split host loops + a
deferred-completion worker thread) after review feedback: the host
round-trip is unnecessary, so eliminate it on the device rather than
hide it. Numbers are from the post-Lever-A nsys profile (RTX 2070, 10
stations, 10 ms subints, 400 subints).

## The problem, measured

Wall time per subint today is ~160 ms, spent as:

| phase | ms/subint | what it is |
|---|---|---|
| station_processing | 83.6 | per-datastream process_gpu, serialized |
| xmac_launch (+drain) | 6.8 | fused XMAC + stream drain |
| host_accumulate | 69.7 | autocorr flush cadence + baseline-weight loops (host) |

station_processing is ~62 ms of GPU kernel time plus ~15 ms of host
loops (set_weights 1.40 ms, calculatePre 0.06 ms per datastream),
serialized against each other by two full stream drains inside every
process_gpu call:

1. `packeddata_gpu->sync()` after the input H2D, before the unpack
   launch (gpumode.cu ~:737) — historic, no data dependency;
2. `nearestSamples->sync()` before set_weights (~:777) — exists only to
   bring the device-computed per-frame validity to the HOST so
   set_weights can run there, after which its outputs (window validity,
   sample indexes) are uploaded straight back to the device.

host_accumulate then runs with the GPU idle: its dominant part, the
baseline-weight loops, reads the host-side dataweight[] arrays that
set_weights produced.

## The observation that drives v2

Follow where each input is born:

- `valid_frames` — born on the DEVICE (unpack/blanker kernels). Only
  D2H'd so the host can loop over it.
- `nearestSamples` — born on the HOST (calculatePre_cpu, from the delay
  polynomials) and ALREADY uploaded to the device for the kernels.
- Everything else set_weights uses is constants (framesamples,
  fftchannels, subint-level validity — host-known before any kernel).

set_weights is pure per-window arithmetic with no cross-window
dependency (the one neighbour read, nearestSamples[i+1], is also
device-resident). It belongs on the device; the round-trip is
artificial. And its main host consumer — the baseline-weight loop,
`sum over windows of w1*w2` per (baseline, pol, freq) — is a trivial
device reduction once the weights are device-resident.

## Increment 1: set_weights on the device

A per-window kernel (one thread per FFT window, launched per datastream
after its unpack/blanker) computes, entirely on-device:

- `dataweight_gpu[window]` — the frame-occupancy weight (same formula:
  whole-frame validity, or the two-frame fractional blend);
- `validSamples_gpu[window]` — window validity (subint-level validity
  becomes a kernel parameter or a small uploaded per-window bool array,
  computed on the host from information it already has — no round trip);
- `sampleIndexes_gpu[window]` = nearestSamples[window] - unpackstart.

Consequences:
- the valid_frames D2H and the `nearestSamples->sync()` drain disappear;
- the indices/validity H2D uploads disappear (computed in place);
- the `packeddata_gpu->sync()` before unpack is removed (no dependency);
- process_gpu becomes drain-free: all 10 datastreams' H2D + kernels
  queue back-to-back on the shared stream and the host runs ahead.

Port notes:
- The arithmetic must match the CPU path at FP level; validate with the
  DIFX_WEIGHT_DEBUG parity tooling. Keep the existing host implementation
  as an env-gated fallback (`DIFX_GPU_WEIGHTS_HOST=1`) — it is also what
  the WDEBUG/HDRDEBUG prints run on, so debugging keeps its tooling (a
  lazy D2H of dataweight under the debug gates).
- This is the natural moment to clean up the known-vestigial conditions
  in set_weights (the unpackstartsamples/unpacksamples clause, the
  nearestSamples==0 stale-weight edge, the "why is this happening"
  abort branch) — simplify with intent rather than porting confusion.
- perbandweights: check its consumers; if it is active in any GPU-path
  config it either joins the kernel or forces the host fallback for
  those configs (acceptable initially).
- Audit ALL host readers of dataweight[]/Mode::getDataWeight on the GPU
  path before dropping the host copy: baseline-weight loops (moved in
  Increment 2), autocorrelation weighting (averageAndSendAutocorrs —
  check), WDEBUG. Until Increment 2 lands, keep one small D2H of
  dataweight_gpu per datastream-subint (a few KB, async, no drain) so
  host_accumulate still works unchanged — Increment 1 stays
  independently correct and testable.

## Increment 2: baseline weights on the device

Replace the host baseline-weight loops with a reduction kernel: for each
(baseline, polproduct, freq), sum dataweight1[w] * dataweight2[w] over
the subint's windows (respecting the same per-window validity the CPU
loop applies), accumulating into a small per-subint device buffer
(baselines x polproducts x freqs floats), zeroed per subint,
transferred/folded into procslots[index].results alongside the existing
visibility D2H.

Consequences:
- host_accumulate shrinks to: the autocorrelation flush at its
  1-in-maxacblocks cadence, pcal copies, and O(baselines) folds —
  the ~60 ms/subint window-dimension loop is gone;
- the interim dataweight D2H from Increment 1 can be dropped (except
  under debug gates and for whatever the autocorr audit found);
- no worker thread, no per-subint snapshots, no new lock discipline —
  v1's Increment 2 machinery is unnecessary.

## What deliberately stays as-is

- The shared in-order compute stream (kernel order, and therefore
  arithmetic, unchanged).
- The XMAC drain + pipelined visibility D2H + h2dInputDone slot fence.
- The autocorr flush cadence (amortized; revisit only if it shows up in
  the post-change profile).
- pcal (synchronous, pending its agreed refactor).

## Projection

Per subint: station_processing approaches its GPU kernel content
(~62 ms desktop), xmac unchanged (~7 ms), host_accumulate drops to the
amortized flush (~a few ms typical). Wall/subint ~160 -> ~72-80 ms:
about 2x on the desktop, and substantially better on data-centre GPUs
where fringeRotation (66% of GPU time here, FP64 at 1/32 rate on
GeForce) is far cheaper — there the host was the binding constraint and
it is now nearly out of the per-subint loop.

## Increment 1 implementation notes (from the code audit)

- The set_weights tail's `indices[freq*MAX_INDICIES+k]` band map is
  window-INDEPENDENT (pure config lookup) — build once at GPUMode
  construction, upload once; delete the per-window rebuild.
- `is_data_valid` inputs are all host-known (validflags bit array,
  datalengthbytes, INVALID_SUBINT sentinel): upload the validflags words
  per subint (tiny) and evaluate in-kernel — no round trip.
- The kernel per window w: valid = validflag(w) && subint-valid;
  weight from start/end frame occupancy using nearestSamples[w],
  nearestSamples[w+1] (own extent for the last window), and
  valid_frames[] with the >= nframes stale-frame guard;
  sampleIndexes[w] = nearestSamples[w] (unpackstart is always 0);
  validSamples[w] = valid && weight > 0. Simplifications approved:
  no stale fall-through (always compute), nearestSamples == -1 means
  invalid (no abort), drop unpackstartsamples.
- Host consumers served in the interim by one async D2H of
  dataweight_gpu into the existing host dataweight[] plus a small
  post-drain host loop (moved out of set_weights) doing the
  weights[0/1][band] autocorr-weight accumulation and WDEBUG prints;
  both move to device in Increment 2.
- calculatePre_cpu unchanged (host-born polynomial work, upload-only).
- DIFX_GPU_WEIGHTS_HOST=1 runs the unmodified current path (drains and
  all) for debugging/regression.

## Gates (each increment separately)

- run-local.sh usb usb-complex complex-complex multi: PASS in both
  DIFX_GPU_PIPELINE modes (multi exercises >2 datastreams, 4 subbands);
- DIFX_GPU_WEIGHTS_HOST=1 leg PASSes (fallback stays alive);
- WDEBUG spot-check: identical weight lines CPU vs GPU on a boundary
  subint (the end-of-recording windows are the danger zone);
- run-bench.sh ledger row;
- nsys spot-check: drains gone from station_processing (Increment 1),
  host_accumulate collapsed (Increment 2).

## Open questions — RESOLVED in review (Adam, 2026-07-18)

1. averageAndSendAutocorrs does NOT read per-window weights, only the
   accumulated value — the interim D2H only needs to serve the
   baseline-weight host loop until Increment 2. (Verify in passing
   during the audit anyway.)
2. perbandweights is NOT currently used on the GPU path — but it SHOULD
   be, in analogy with how CPUMode/Mk5Mode uses it for interlaced VDIF.
   That is a correctness gap to fix as its own follow-up work item (the
   device weights kernel should be shaped so a per-(window, band) weight
   is a natural extension); recorded in gpu-plan.md.
3. Simplify the vestigial set_weights conditions during the port, with
   WDEBUG parity as the referee. Approved.
