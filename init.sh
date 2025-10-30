#!/usr/bin/env bash
# Initialize pixi and set cache and temporary directories.
WD="$(realpath .)"
export PATH="$WD/infrastructure/apps/pixi/bin":$PATH
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="/tmp/.tmp_$USER"
mkdir -p "$TMPDIR"

mkdir -p "$WD/slurm_output"

# Single threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING="False"

# Ensure that the latest documentation is built.
pixi self-update --version 0.48.0
