#!/bin/bash
#SBATCH --job-name=train-worm-seg
#SBATCH --cpus-per-task=26
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_output/run-%j.out
#SBATCH --error=slurm_output/run-%j.err
#SBATCH --partition=main
#SBATCH --mem=196GB
#SBATCH --constraint "infiniband&gpuram80gb"
#SBATCH --time=48:00:00
account="$1"
working_dir="$2"

set -eu

function display_memory_usage() {
        set +eu
        echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
        cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
        echo
}

trap display_memory_usage EXIT

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM job $SLURM_JOB_ID"
echo "[INFO] [$STARTDATE] [$$] Running in $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"

### PUT YOUR CODE IN THIS SECTION
export SBATCH_ACCOUNT="$account"

WD="$working_dir" pixi run --no-lockfile-update train_model

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
