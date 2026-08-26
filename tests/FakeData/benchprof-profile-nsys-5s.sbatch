#!/bin/bash
#
#SBATCH --job-name=difx_benchprof_nsys5s
#SBATCH --output=benchprof-nsys-5s.%j.mpilog
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=4000
#SBATCH --time=15:00
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<gpu-partition>
#
# NUMA: --gres-flags=enforce-binding makes SLURM allocate only CPUs local to
# the selected GPU. Without it the 12 ranks land on arbitrary cores, and if the
# Core rank ends up on the far socket its page-locked receive buffers (first
# touched there, then registered for direct H2D) feed the GPU across the
# inter-socket link at about half bandwidth: two profiles of the identical
# binary and job measured 8.9 vs 18.9 GB/s of pinned H2D for the same 12.9 GB,
# ~760 ms of a ~7.7 s wall. It can lengthen the queue wait, since the scheduler
# now needs free cores on the right node. mpifxcorr logs the resulting
# placement ("GPU Core N placement: ... on NUMA node ...") and warns if the
# rank still ends up remote - check the difxlog before trusting a profile.
#
# Clean GPU-bound Nsight Systems profile of the 10-antenna fake-data
# "benchprof" job (source=fake, ~5 s window = ~500 x 10ms subints). Deliberately
# removes the two things that muddied the earlier profiles:
#   * source=fake  -> no VDIF disk I/O,
#   * --cpus-per-task=1 over 12 ranks (1 manager + 10 datastreams + 1 core)
#     -> one rank per core, no oversubscription.
# Only the Core rank (the last MPI rank = rank 11) is profiled; every other
# rank runs mpifxcorr bare, via the committed nsys-wrapper.sh.
#
# WHY ONLY ~5 s: nsys 2022.2.1's injection library (libToolsInjection64.so)
# SEGFAULTS on a long/heavy capture - a ~20 s cuda,nvtx,osrt trace over ~2000
# subints crashed its worker thread ~13 s in (backtrace entirely inside
# libToolsInjection64.so; the correlator was fine). A short window keeps the
# trace small enough to survive. For a full-length TIMING/soak run, use the
# nsys-free companion benchprof-profile-nonsys-20s.sbatch instead. If nsys still
# falls over, drop osrt (NSYS_TRACE=cuda,nvtx) and/or use a newer nsys.
#
# Submit from the tests/FakeData directory of your DiFX checkout (which holds
# benchprof.v2d, bd247c3.vex.obs and nsys-wrapper.sh):
#     cd <checkout>/tests/FakeData && sbatch benchprof-profile-nsys-5s.sbatch
#
# The report lands at $WORKDIR/nsys/benchprof-cluster-clean.nsys-rep - scp it
# to the desktop for analysis (login-node nsys is too old to `nsys stats` it).

set -e

# --- config (edit WORKDIR if /fred/oz168/... isn't your scratch) -------------
SETUP=/home/adeller/setup_gpudifx.claude                    # DiFX build under test
SRCDIR="${SLURM_SUBMIT_DIR:-$PWD}"                           # holds v2d + vex + wrapper
WORKDIR=/fred/oz168/adeller/claude/testing/benchprof        # scratch, shared filesystem
# -----------------------------------------------------------------------------

. "$SETUP"
module load nsight-systems/2022.2.1 2>/dev/null || true              # ensure nsys is on PATH

# Restrict difxmessage multicast to this node so parallel jobs don't cross-talk.
export DIFX_MESSAGE_GROUP=$(hostname -i)
export DIFX_BINARY_GROUP=$(hostname -i)

# Stage a clean scratch workdir and generate the job (source=fake -> no VDIF).
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/nsys"
cp "$SRCDIR/benchprof.v2d" "$SRCDIR/bd247c3.vex.obs" "$SRCDIR/nsys-wrapper.sh" "$WORKDIR/"
chmod +x "$WORKDIR/nsys-wrapper.sh"
cd "$WORKDIR"

# Profile a short (~5 s) window - long enough for steady-state kernel-mix/idle
# numbers, short enough that the nsys 2022.2.1 trace survives (see the header).
# mjdStart 59675.09606481 + 5 s = 59675.09612270.
sed -i 's|^mjdStop .*|mjdStop  = 59675.09612270|' benchprof.v2d

vex2difx benchprof.v2d          # -> benchprof_1.input/.calc/.threads
difxcalc benchprof_1.calc       # -> benchprof_1.im
nds=$(awk '/^ACTIVE DATASTREAMS/{print $3}' benchprof_1.input)
echo "benchprof: $nds active datastreams, subint = $(awk '/^SUBINT NANOSEC/{print $3}' benchprof_1.input) ns"
if [ "$nds" != 10 ]; then
    echo "ERROR: expected 10 active datastreams (need ntasks=12), got '$nds' - check the mjdStart window in benchprof.v2d" >&2
    exit 1
fi

# Bypass the DataStream interlaced-VDIF corner-turn so this profiles the GPU
# correlator, not the CPU-side thread demux (the current bottleneck - see
# gpu-changes.md sec 10). vex2difx always emits INTERLACEDVDIF for the multi-band
# fake data; rewrite it to single-thread VDIF (16 chans in one thread, the GPU
# unpack de-interleaves them), keeping DATA FRAME SIZE. Needs the mark5access
# VDIF genheaders (commit 191528ff9). Set SINGLE_THREAD_VDIF=0 to keep the
# interlaced demux (e.g. to reproduce the corner-turn-bound profile).
if [ "${SINGLE_THREAD_VDIF:-1}" != 0 ]; then
    sed -i -E 's|(DATA FORMAT:[[:space:]]+)INTERLACEDVDIF/[0-9:]+|\1VDIF|' benchprof_1.input
    echo "benchprof: rewrote INTERLACEDVDIF -> single-thread VDIF (set SINGLE_THREAD_VDIF=0 to disable)"
fi

# Record the node's CPU/GPU topology, so a slow profile can be diagnosed after
# the fact (mpifxcorr's own placement line reports only the Core rank's view).
echo "=== node topology ($(hostname)) ==="
numactl -H 2>/dev/null || echo "numactl unavailable"
nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi topo unavailable"
echo "=== end topology ==="

date
# Collect the correlator log; killed once mpifxcorr returns.
difxlog benchprof_1 "$WORKDIR/benchprof_1.difxlog" 4 &
difxlogpid=$!

# One rank per core (--cpu-bind=cores); nsys wraps only the Core rank
# (CORE_RANK = last rank). The wrapper's rank-0 also clears any stale output.
CORE_RANK=$(( SLURM_NTASKS - 1 )) \
NSYS_OUT="$WORKDIR/nsys/benchprof-cluster-clean" \
    srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores \
         "$WORKDIR/nsys-wrapper.sh" benchprof_1.input --nocommandthread --usegpu

kill "$difxlogpid" 2>/dev/null || true
date
echo "nsys report: $WORKDIR/nsys/benchprof-cluster-clean.nsys-rep"
