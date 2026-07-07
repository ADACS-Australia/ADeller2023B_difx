# gen-sbatch.sh - sbatch script generation for run-slurm.sh
#
# Sourced by run-slurm.sh; not meant to be run on its own.  Provides
# write_sbatch(), which emits one ready-to-submit sbatch script per correlation
# mode.  Modelled directly on the hand-written runparallel.benchsimreal /
# runparallel.benchsimrealgpu cluster scripts: source a DiFX setup script, set
# the difxmessage multicast groups to this node, start a difxlog collector in
# the background, then srun mpifxcorr.
#
# The three modes differ only in which setup script is sourced, the SLURM
# resource sizing, and whether "--usegpu" is appended, so a single template
# with a mode switch avoids duplicating near-identical files.
#
# Expects the config variables from slurm.conf to already be in the environment.

# write_sbatch <mode:cpu|gpu|reference> <expname> <jobdir>
#
# Writes <jobdir>/slurm-<mode>.sh.  The generated script cd's into <jobdir> and
# correlates <expname>.input, which run-slurm.sh has already produced there.
write_sbatch() {
    local mode="$1"
    local expname="$2"
    local jobdir="$3"

    local ntasks time mem setup gres="" usegpu=""
    case "$mode" in
        cpu)
            ntasks="$CPU_NTASKS"; time="$CPU_TIME"; mem="$CPU_MEM_PER_CPU"
            setup="$SETUP_SCRIPT"
            ;;
        gpu)
            ntasks="$GPU_NTASKS"; time="$GPU_TIME"; mem="$GPU_MEM_PER_CPU"
            setup="$SETUP_SCRIPT"
            gres="$GPU_GRES"; usegpu="--usegpu"
            ;;
        reference)
            ntasks="$REFERENCE_NTASKS"; time="$REFERENCE_TIME"; mem="$REFERENCE_MEM_PER_CPU"
            setup="$REFERENCE_SETUP_SCRIPT"
            ;;
        *)
            echo "write_sbatch: unknown mode '$mode'" >&2
            return 1
            ;;
    esac

    local script="$jobdir/slurm-$mode.sh"

    {
        echo "#!/bin/bash"
        echo "#"
        echo "#SBATCH --job-name=difx_${expname}_${mode}"
        echo "#SBATCH --output=${jobdir}/${mode}.mpilog"
        echo "#SBATCH --ntasks=${ntasks}"
        echo "#SBATCH --time=${time}"
        echo "#SBATCH --cpus-per-task=1"
        echo "#SBATCH --mem-per-cpu=${mem}"
        echo "#SBATCH --account=${SLURM_ACCOUNT}"
        [ -n "$SLURM_PARTITION" ] && echo "#SBATCH --partition=${SLURM_PARTITION}"
        [ -n "$gres" ]           && echo "#SBATCH --gres=${gres}"
        echo ""
        echo "set -e"
        echo ""
        echo ". ${setup}"
        echo ""
        echo "# Restrict difxmessage multicast to this node so parallel jobs don't cross-talk"
        echo "export DIFX_MESSAGE_GROUP=\`hostname -i\`"
        echo "export DIFX_BINARY_GROUP=\`hostname -i\`"
        echo ""
        # Bake debug variables into the job script: sbatch environment export
        # cannot be relied on (sites configure --export=NONE), and the srun'd
        # tasks must see these.
        if [ -n "${DIFX_WEIGHT_DEBUG:-}" ]; then
            echo "export DIFX_WEIGHT_DEBUG=${DIFX_WEIGHT_DEBUG}"
            echo ""
        fi
        echo "cd ${jobdir}"
        echo "date"
        echo ""
        echo "# Collect the correlator log; killed once mpifxcorr returns"
        echo "difxlog ${expname} ${jobdir}/${expname}.difxlog 4 &"
        echo "difxlogpid=\$!"
        echo ""
        echo "srun -n${ntasks} mpifxcorr ${expname}.input --nocommandthread ${usegpu}"
        echo ""
        echo "kill \$difxlogpid 2>/dev/null || true"
        echo "date"
    } > "$script"

    chmod +x "$script"
    echo "$script"
}
