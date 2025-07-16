#!/bin/bash
#SBATCH --job-name=Prepare-Annotation-Data
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_output/run-%j.out
#SBATCH --error=slurm_output/run-%j.err
#SBATCH --partition=main
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a40:1
#SBATCH --constraint infiniband
#SBATCH --time=12:00:00
working_dir="$1"

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

WD="$working_dir" pixi run --no-lockfile-update prepare_annotation_data

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
