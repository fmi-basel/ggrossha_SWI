#!/bin/bash
#SBATCH --job-name=SWI-Analysis
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_output/run-%j.out
#SBATCH --error=slurm_output/run-%j.err
#SBATCH --partition=orchestration
#SBATCH --oversubscribe
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
account="$1"
mail_to="$2"
working_dir="$3"

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
export NXF_OPTS="-Xms500M -Xmx2G"
export NXF_EXECUTOR="slurm"
export NXF_ANSI_LOG=false
export NXF_HOME="$(pwd)/infrastructure/.nxf_home"

# Single threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING="False"

MAIL="$mail_to" WD="$working_dir" pixi --no-lockfile-update run workflow_slurm

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
