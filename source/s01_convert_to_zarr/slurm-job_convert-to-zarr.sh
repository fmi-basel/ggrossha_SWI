#!/bin/bash
#SBATCH --job-name=SWI-Convert2Zarr
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_output/run-%j.out
#SBATCH --error=slurm_output/run-%j.err
#SBATCH --partition=main
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
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
export PATH=$PATH:"$(pwd)/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$(pwd)/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$(pwd)/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

# Single threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING="False"

WD="$working_dir" pixi run convert_to_zarr

### END OF PUT YOUR CODE IN THIS SECTION

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Workflow execution time \(seconds\) : $(( $END-$START ))"
