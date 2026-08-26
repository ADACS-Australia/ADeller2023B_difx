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
# TWO WAYS TO RUN THIS - pick deliberately:
#
#   sbatch --cpus-per-task=5 benchprof-profile-nsys-5s.sbatch   # LEDGER-QUALITY
#   sbatch                   benchprof-profile-nsys-5s.sbatch   # quick look only
#
# ISOLATION MATTERS for any number that goes in BENCHMARKS.md. On a shared node
# a co-tenant job stalls the input copies: the same 12.9 GB of pinned H2D took
# 1445 ms on one node and 589 ms on another, ~11% of wall, with an IDENTICAL
# binary. Every node reaches the same 26 GB/s per-copy peak - what differs is a
# tail of stalled copies, i.e. interference, not hardware. A shared run can
# silently be 10% slow and look like a regression.
#
# tooarrana REJECTS --exclusive ("Exclusive option not permitted, please request
# the number of cores you need per task with --cpus-per-task=x instead"), so the
# way to get isolation here is to reserve most of the node's cores:
# --cpus-per-task=5 x 12 tasks = 60 of gina's 64 cores. This does NOT change
# how many mpifxcorr ranks start - the rank count is --ntasks (12) and the roles
# come from the .input's ACTIVE DATASTREAMS (rank 0 manager, 1-10 datastreams,
# 11 Core); the extra cores are simply reserved and mostly idle. The cost is
# queue time, which is the whole reason there are two modes. The run prints its
# own "node ownership: N/M cores" verdict below - check it before quoting.
# (--gres-flags=enforce-binding below keeps the ranks on CPUs local to the
# selected GPU. Measured on these nodes it buys nothing - placements from
# GPU-local to cross-socket all hit the same peak - but it is free and would
# matter on hardware with a slower cross-socket link. See gpu-changes.md
# sec 14 for the measurements behind both statements.)
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
# benchprof.v2d, bd247c3.vex.obs and nsys-wrapper.sh) - see the two modes above:
#     cd <checkout>/tests/FakeData && sbatch --cpus-per-task=5 benchprof-profile-nsys-5s.sbatch
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

# Say in the log whether this run owns the node, so a number can be trusted (or
# not) after the fact rather than from memory of how it was submitted.
# tooarrana forbids --exclusive ("please request the number of cores you need
# per task with --cpus-per-task=x instead"), so isolation is measured as the
# fraction of the node's cores this job reserved, not from OverSubscribe.
cores_alloc=$(( ${SLURM_CPUS_PER_TASK:-1} * ${SLURM_NTASKS:-1} ))
cores_node=$(nproc --all)
pct=$(( 100 * cores_alloc / cores_node ))
if [ "$pct" -ge 90 ]; then
    echo "=== node ownership: $cores_alloc/$cores_node cores (${pct}%) - near-exclusive, ledger-quality ==="
else
    echo "=== node ownership: $cores_alloc/$cores_node cores (${pct}%) - SHARED, NOT ledger-quality;" \
         "a co-tenant can cost ~10% of wall. Raise --cpus-per-task. ==="
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
