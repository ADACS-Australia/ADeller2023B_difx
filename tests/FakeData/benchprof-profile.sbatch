#!/bin/bash
#
#SBATCH --job-name=difx_benchprof_gpu
#SBATCH --output=benchprof-profile.%j.mpilog
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=15:00
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<gpu-partition>
#
# Clean GPU-bound Nsight Systems profile of the 10-antenna fake-data
# "benchprof" job (source=fake, 400 x 10ms subints). Deliberately removes the
# two things that muddied the earlier profiles:
#   * source=fake  -> no VDIF disk I/O,
#   * --exclusive + --cpus-per-task=1 over 12 ranks (1 manager + 10 datastreams
#     + 1 core) -> one rank per core, no oversubscription.
# Only the Core rank (the last MPI rank = rank 11) is profiled; every other
# rank runs mpifxcorr bare, via the committed nsys-wrapper.sh.
#
# Submit from the tests/FakeData directory of your DiFX checkout (which holds
# benchprof.v2d, bd247c3.vex.obs and nsys-wrapper.sh):
#     cd <checkout>/tests/FakeData && sbatch benchprof-profile.sbatch
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
module load nsight-systems 2>/dev/null || true              # ensure nsys is on PATH

# Restrict difxmessage multicast to this node so parallel jobs don't cross-talk.
export DIFX_MESSAGE_GROUP=$(hostname -i)
export DIFX_BINARY_GROUP=$(hostname -i)

# Stage a clean scratch workdir and generate the job (source=fake -> no VDIF).
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/nsys"
cp "$SRCDIR/benchprof.v2d" "$SRCDIR/bd247c3.vex.obs" "$SRCDIR/nsys-wrapper.sh" "$WORKDIR/"
chmod +x "$WORKDIR/nsys-wrapper.sh"
cd "$WORKDIR"

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
