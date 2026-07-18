# FakeData: 10-station GPU benchmark

`benchsimreal.v2d` + `bd247c3.vex.obs` define a 10-station VLBA benchmark
(16 x 32 MHz, 2 Gbps/station) using `source = fake` datastreams - no
baseband data files needed. This is the performance-timing counterpart to
the correctness tests in `tests/Synthetic`.

The v2d is desktop-adapted (RTX 2070, 8 GB): see the comments in it for
the antenna-window, threads-file, VDIF-format and subintNS choices.

## Run

```sh
source ../../setup.bash
vex2difx benchsimreal.v2d && difxcalc benchsimreal_1.calc
grep 'ACTIVE DATASTREAMS' benchsimreal_1.input   # must be 10!
difxlog benchsimreal_1 difxlog.txt 5 &           # difxmessage eats stdout INFO
/usr/bin/time -v mpirun --oversubscribe --mca mpi_yield_when_idle 1 \
    -machinefile machines -np 12 \
    mpifxcorr benchsimreal_1.input --nocommandthread --usegpu
```

Notes:
- 12 ranks = 1 FxManager + 10 datastreams + 1 (GPU)Core; on a small desktop
  this oversubscribes the cores, hence the yield option.
- mpirun exits nonzero even on success (fake datastreams exit without
  MPI_Finalize at teardown). Judge success by a non-empty
  `benchsimreal_1.difx/DIFX_*` file, not the exit code.
- Wall-time scaling on an RTX 2070: 15 s of data (the default window)
  takes ~260 s.
- **Benchmarking only - do not use for correctness testing.** The fake
  datastream writes valid VDIF headers over untouched (uninitialised)
  payload bytes, so every antenna delivers essentially the same
  pathological stream: correlation sums grow coherently (~N, not sqrt N),
  amplifying FP rounding differences, and the XMAC's atomic adds make the
  output non-bit-reproducible run to run (~1e-6 relative variation).
  Correctness lives in tests/Synthetic.

## Profiling (Nsight Systems)

Wrap mpirun itself - do NOT wrap individual ranks (the teardown above gets
per-rank nsys killed mid-report-write). Set `subintNS = 10000000` in the
v2d first: CUPTI needs VRAM on top of the job's own budget.

```sh
nsys profile --trace=cuda,nvtx,osrt -o bench --force-overwrite true \
    mpirun --oversubscribe --mca mpi_yield_when_idle 1 \
    -machinefile machines -np 12 \
    mpifxcorr benchsimreal_1.input --nocommandthread --usegpu
nsys stats --force-export=true --report nvtx_pushpop_sum \
    --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum bench.nsys-rep
```

Only the GPU core rank emits CUDA/NVTX activity, so tracing all ranks is
cheap. If a profiling run is interrupted, orphaned `nsys --start-agent`
daemons may be left spinning at 100% CPU - find them with
`pgrep -af nsys` and kill them.
