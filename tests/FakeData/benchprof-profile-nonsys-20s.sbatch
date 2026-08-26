#!/bin/bash
#
#SBATCH --job-name=difx_benchprof_nonsys
#SBATCH --output=benchprof-nonsys-20s.%j.mpilog
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=4000
#SBATCH --time=15:00
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<gpu-partition>
#
# nsys-FREE ~20 s run of the 10-antenna fake-data "benchprof" job. Companion to
# benchprof-profile-nsys-5s.sbatch: that one profiles a SHORT window under
# Nsight Systems; THIS one runs the correlator BARE to (a) confirm a
# full-length run completes and (b) measure wall time / speed vs real time
# without any profiler overhead.
#
# TWO WAYS TO RUN THIS - pick deliberately:
#
#   sbatch --exclusive benchprof-profile-nonsys-20s.sbatch   # LEDGER-QUALITY
#   sbatch             benchprof-profile-nonsys-20s.sbatch   # quick look only
#
# ALWAYS use --exclusive for any number that goes in BENCHMARKS.md - this
# script's whole output is a wall time, and a co-tenant job on a shared node
# costs ~10% of it (measured: identical binary, 1445 vs 589 ms of input H2D).
# See the header of benchprof-profile-nsys-5s.sbatch and gpu-changes.md sec 14.
#
# Why a separate no-nsys script: nsys 2022.2.1's injection library
# (libToolsInjection64.so) segfaults on a long/heavy capture - a ~20 s
# cuda,nvtx,osrt trace over ~2000 subints crashed its worker thread ~13 s into
# the run. The crash backtrace is ENTIRELY inside libToolsInjection64.so (no
# DiFX frame on the stack) - the correlator itself is fine and had reached
# near-real-time speed. So for a full-length timing/soak run, don't profile:
# run bare here, and keep nsys captures short (the -nsys-5s script).
#
# Same clean setup as the profile script: source=fake (no VDIF disk I/O), one
# rank per core (--cpu-bind=cores, 12 ranks = 1 manager + 10 datastreams + 1
# core), single-thread VDIF (bypass the DataStream corner-turn).
#
# Submit from the tests/FakeData directory of your DiFX checkout (which holds
# benchprof.v2d and bd247c3.vex.obs):
#     cd <checkout>/tests/FakeData && sbatch benchprof-profile-nonsys-20s.sbatch

set -e

# --- config (edit WORKDIR if /fred/oz168/... isn't your scratch) -------------
SETUP=/home/adeller/setup_gpudifx.claude                     # DiFX build under test
SRCDIR="${SLURM_SUBMIT_DIR:-$PWD}"                            # holds v2d + vex
WORKDIR=/fred/oz168/adeller/claude/testing/benchprof-nonsys  # scratch, shared FS
# -----------------------------------------------------------------------------

. "$SETUP"

# Restrict difxmessage multicast to this node so parallel jobs don't cross-talk.
export DIFX_MESSAGE_GROUP=$(hostname -i)
export DIFX_BINARY_GROUP=$(hostname -i)

# Stage a clean scratch workdir and generate the job (source=fake -> no VDIF).
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cp "$SRCDIR/benchprof.v2d" "$SRCDIR/bd247c3.vex.obs" "$WORKDIR/"
cd "$WORKDIR"

# ~20 s window (mjdStart 59675.09606481 + 20 s = 59675.09629650) - a full-length
# soak/timing run, longer than the -nsys-5s profile window.
sed -i 's|^mjdStop .*|mjdStop  = 59675.09629650|' benchprof.v2d

vex2difx benchprof.v2d          # -> benchprof_1.input/.calc/.threads
difxcalc benchprof_1.calc       # -> benchprof_1.im
nds=$(awk '/^ACTIVE DATASTREAMS/{print $3}' benchprof_1.input)
echo "benchprof: $nds active datastreams, subint = $(awk '/^SUBINT NANOSEC/{print $3}' benchprof_1.input) ns"
if [ "$nds" != 10 ]; then
    echo "ERROR: expected 10 active datastreams (need ntasks=12), got '$nds' - check the mjdStart window in benchprof.v2d" >&2
    exit 1
fi

# Bypass the DataStream interlaced-VDIF corner-turn so this measures the GPU
# correlator, not the CPU-side thread demux (see gpu-changes.md sec 10). Rewrite
# INTERLACEDVDIF -> single-thread VDIF (needs mark5access genheaders, commit
# 191528ff9). Set SINGLE_THREAD_VDIF=0 to keep the interlaced demux.
if [ "${SINGLE_THREAD_VDIF:-1}" != 0 ]; then
    sed -i -E 's|(DATA FORMAT:[[:space:]]+)INTERLACEDVDIF/[0-9:]+|\1VDIF|' benchprof_1.input
    echo "benchprof: rewrote INTERLACEDVDIF -> single-thread VDIF (set SINGLE_THREAD_VDIF=0 to disable)"
fi

# Say in the log whether this run owns the node, so a wall time can be trusted
# (or not) after the fact rather than from memory of how it was submitted.
oversub=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'OverSubscribe=[A-Za-z]*' | head -1)
case "$oversub" in
    *EXCLUSIVE*) echo "=== node ownership: EXCLUSIVE - ledger-quality ===" ;;
    "")          echo "=== node ownership: UNKNOWN (scontrol unavailable) - verify before using for the ledger ===" ;;
    *)           echo "=== node ownership: SHARED ($oversub) - NOT ledger-quality, a co-tenant can cost ~10% ===" ;;
esac

# Record the node's CPU/GPU topology, so a slow run can be diagnosed after the
# fact (mpifxcorr's own placement line reports only the Core rank's view).
echo "=== node topology ($(hostname)) ==="
numactl -H 2>/dev/null || echo "numactl unavailable"
nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi topo unavailable"
echo "=== end topology ==="

# Collect the correlator log; killed once mpifxcorr returns.
difxlog benchprof_1 "$WORKDIR/benchprof_1.difxlog" 4 &
difxlogpid=$!

# One rank per core (--cpu-bind=cores); NO nsys wrapper - mpifxcorr runs bare on
# every rank. A fresh WORKDIR means there is no stale output for rank 0 to trip
# over, so the wrapper's cleanup is unnecessary here.
echo "=== launching bare (no nsys) at $(date +%T) ==="
t0=$(date +%s.%N)
srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores \
     mpifxcorr benchprof_1.input --nocommandthread --usegpu
t1=$(date +%s.%N)
kill "$difxlogpid" 2>/dev/null || true

wall=$(echo "$t0 $t1" | awk '{printf "%.1f", $2-$1}')
echo "=== mpifxcorr returned at $(date +%T); wall ${wall}s for a ~20 s data window ==="
out=$(ls -S "$WORKDIR"/benchprof_1.difx/DIFX_* 2>/dev/null | head -1)
if [ -n "$out" ] && [ -s "$out" ]; then
    echo "output OK: $out ($(stat -c %s "$out") bytes)"
else
    echo "ERROR: no visibility output produced" >&2
    exit 1
fi
