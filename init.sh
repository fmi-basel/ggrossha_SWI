#!/usr/bin/env bash
# Initialize pixi and set cache and temporary directories.
WD="$(realpath .)"
export PATH=$PATH:"$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

mkdir -p "$WD/slurm_output"

micro_sam="$HOME/.cache/micro_sam/"
if [ ! -d "$micro_sam" ]; then
    echo "Please create the ~/.cache/micro_sam/ directory."
    exit 1
fi

# Single threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING="False"

# Ensure that the latest documentation is built.
pixi run build_docs
