#!/usr/bin/env nextflow

params.convert_to_zarr_config = "s01_convert_to_zarr_config.yaml"
params.segmentation_config = "s02_segmentation_config.yaml"

process PREPARE {
    label 'cpu_tiny'

    input:
    path convert_to_zarr_config

    output:
    path "s01_convert_to_zarr_config_*.yaml"

    script:
    """
    pixi run --no-lockfile-update python $baseDir/s01_convert_to_zarr/prepare_parallel_run.py --config $convert_to_zarr_config
    """
}

process CONVERT2ZARR {
    label 'cpu'

    input:
    path convert_to_zarr_config

    output:
    path "s01_convert_to_zarr_result.yaml"

    script:
    """
    pixi run --no-lockfile-update python $baseDir/s01_convert_to_zarr/run.py --config $convert_to_zarr_config
    """
}

process SEGMENT {
    label 'gpu'

    input:
    path segmentation_config
    path inputs

    output:
    path "s02_segment_result.yaml"

    script:
    """
    pixi run --no-lockfile-update -e worm-segmentation python $baseDir/s02_segment/run.py --config $segmentation_config --inputs $inputs
    """
}

workflow {
    configs = PREPARE(params.convert_to_zarr_config)
    zarrs = CONVERT2ZARR(configs.flatten())
//     segmentations = SEGMENT(params.segmentation_config, zarrs)
}
