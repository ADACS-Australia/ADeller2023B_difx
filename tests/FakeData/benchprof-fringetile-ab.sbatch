#!/bin/bash
#
#SBATCH --job-name=difx_fringetile_ab
#SBATCH --output=benchprof-fringetile-ab.%j.mpilog
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=4000
#SBATCH --time=30:00
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<gpu-partition>
#
# A/B the tiled fused decode+fringe kernel against the untiled one on the
# cluster GPU, in ONE allocation so both legs see the same node, the same clocks
# and the same co-tenants - the comparison is a ratio, and running the legs as
# separate jobs is how a ~10% node-contention difference gets mistaken for a
# kernel result (gpu-changes.md sec 14).
#
# The tiling is 1.2-1.5x on the RTX 2070 at every shape (gpu-fringetile-design.md),
# but that card is FP64-starved and its kernel mix is nothing like the A100's:
# gpu_fused_fringe is 46.5% of 2070 kernel time and only ~21% on the A100. So the
# expected A100 wall win is smaller, and this job is what decides whether
# DIFX_GPU_FRINGE_TILE stays on by default for cluster work.
#
# TWO WAYS TO RUN THIS - pick deliberately:
#
#   sbatch --cpus-per-task=5 benchprof-fringetile-ab.sbatch   # LEDGER-QUALITY
#   sbatch                   benchprof-fringetile-ab.sbatch   # quick look only
#
# --cpus-per-task=5 x 12 tasks = 60 of gina's 64 cores (tooarrana rejects
# --exclusive). It does NOT start more ranks - that is --ntasks. --nodes=1 is
# load-bearing: a wide --cpus-per-task otherwise gets split over two nodes,
# which reserves a GPU on each and pushes DataStream->Core traffic onto the
# fabric. Both are explained at length in benchprof-profile-nsys-5s.sbatch.
#
# Submit from the tests/FakeData directory of the checkout:
#     cd <checkout>/tests/FakeData && sbatch --cpus-per-task=5 benchprof-fringetile-ab.sbatch
#
# For per-kernel numbers rather than wall time, run the nsys script once per
# path instead:
#     sbatch --cpus-per-task=5 --export=ALL,DIFX_GPU_FRINGE_TILE=1 benchprof-profile-nsys-5s.sbatch
#     sbatch --cpus-per-task=5 --export=ALL,DIFX_GPU_FRINGE_TILE=0 benchprof-profile-nsys-5s.sbatch
# then compare gpu_fused_fringe(_tiled) us/call with nsys-analyze.py.

set -e

# --- config (edit WORKDIR if /fred/oz168/... isn't your scratch) -------------
SETUP=/home/adeller/setup_gpudifx.claude                        # DiFX build under test
SRCDIR="${SLURM_SUBMIT_DIR:-$PWD}"                              # holds v2d + vex
WORKDIR=/fred/oz168/adeller/claude/testing/benchprof-fringetile # scratch, shared FS
WINDOW_MJDSTOP=59675.09629650                                   # mjdStart + ~20 s
REPEATS="${REPEATS:-2}"                                         # A/B pairs (interleaved)
# -----------------------------------------------------------------------------

. "$SETUP"

export DIFX_MESSAGE_GROUP=$(hostname -i)
export DIFX_BINARY_GROUP=$(hostname -i)

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cp "$SRCDIR/benchprof.v2d" "$SRCDIR/bd247c3.vex.obs" "$WORKDIR/"
cd "$WORKDIR"

sed -i "s|^mjdStop .*|mjdStop  = $WINDOW_MJDSTOP|" benchprof.v2d
vex2difx benchprof.v2d
difxcalc benchprof_1.calc
nds=$(awk '/^ACTIVE DATASTREAMS/{print $3}' benchprof_1.input)
if [ "$nds" != 10 ]; then
    echo "ERROR: expected 10 active datastreams (ntasks=12), got '$nds'" >&2
    exit 1
fi
if [ "${SINGLE_THREAD_VDIF:-1}" != 0 ]; then
    sed -i -E 's|(DATA FORMAT:[[:space:]]+)INTERLACEDVDIF/[0-9:]+|\1VDIF|' benchprof_1.input
    echo "rewrote INTERLACEDVDIF -> single-thread VDIF (measures the GPU, not the corner-turn)"
fi

# Same node-ownership verdict the other scripts print, so a ratio can be
# trusted (or not) after the fact.
nnodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
[ "$nnodes" = 1 ] || echo "=== WARNING: allocation spans $nnodes nodes - NOT comparable. Keep --nodes=1. ==="
cores_alloc=$(( ${SLURM_CPUS_PER_TASK:-1} * ${SLURM_NTASKS:-1} ))
cores_node=$(nproc --all)
pct=$(( 100 * cores_alloc / cores_node ))
if [ "$pct" -ge 90 ]; then
    echo "=== node ownership: $cores_alloc/$cores_node cores (${pct}%) - near-exclusive, ledger-quality ==="
else
    echo "=== node ownership: $cores_alloc/$cores_node cores (${pct}%) - SHARED, NOT ledger-quality ==="
fi
echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_leg() { # run_leg <tile 0|1> <rep>
    local tile="$1" rep="$2" t0 t1 wall out
    rm -rf benchprof_1.difx
    difxlog benchprof_1 "$WORKDIR/fringetile-tile$tile-rep$rep.difxlog" 4 &
    local logpid=$!
    t0=$(date +%s.%N)
    DIFX_GPU_FRINGE_TILE="$tile" srun --ntasks="$SLURM_NTASKS" --cpu-bind=cores \
        mpifxcorr benchprof_1.input --nocommandthread --usegpu \
        > "fringetile-tile$tile-rep$rep.runlog" 2>&1
    t1=$(date +%s.%N)
    kill "$logpid" 2>/dev/null || true
    wall=$(echo "$t0 $t1" | awk '{printf "%.1f", $2-$1}')
    out=$(ls -S "$WORKDIR"/benchprof_1.difx/DIFX_* 2>/dev/null | head -1)
    if [ -z "$out" ] || [ ! -s "$out" ]; then
        echo "ERROR: tile=$tile rep=$rep produced no visibilities (see fringetile-tile$tile-rep$rep.runlog)" >&2
        exit 1
    fi
    echo "$wall"
}

# Interleave the legs (1,0,1,0,...) so any slow drift in the node hits both.
echo "=== A/B: $REPEATS interleaved pairs, ~20 s data window each leg ==="
tiled=""; untiled=""
for rep in $(seq 1 "$REPEATS"); do
    w1=$(run_leg 1 "$rep"); echo "  rep $rep tiled   : ${w1}s"
    w0=$(run_leg 0 "$rep"); echo "  rep $rep untiled : ${w0}s"
    tiled="$tiled $w1"; untiled="$untiled $w0"
done

echo "=== result ==="
echo "$tiled|$untiled" | awk -F'|' '{
    n=split($1,a," "); m=split($2,b," ");
    bt=1e9; for(i=1;i<=n;i++) if(a[i]<bt) bt=a[i];
    bu=1e9; for(i=1;i<=m;i++) if(b[i]<bu) bu=b[i];
    printf "best tiled   : %.1fs\nbest untiled : %.1fs\nspeedup      : %.3fx\n", bt, bu, bu/bt;
    if (bu/bt < 1.0) print "NOTE: tiled is SLOWER here - keep DIFX_GPU_FRINGE_TILE=0 for this architecture and say so in BENCHMARKS.md.";
}'
echo "Add the numbers to tests/FakeData/BENCHMARKS.md and gpu-fringetile-design.md."
