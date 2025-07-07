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

process PROJECT {
    label 'cpu_tiny'

    input:
    path segmentation_config
    path inputs

    output:
    path "s01_convert_to_zarr_result.yaml"

    script:
    """
    pixi run --no-lockfile-update python $baseDir/s02_segment/generate_projections.py --config $segmentation_config --inputs $inputs
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

process MEASURE {
    label 'cpu_tiny'

    input:
    path segmentation_config
    path inputs

    output:
    path "measurement_files.yaml"

    script:
    """
    pixi run --no-lockfile-update -e worm-segmentation python $baseDir/s03_measure/run.py --config $segmentation_config --datasets $inputs
    """
}

process QC {
    label 'cpu_tiny'

    input:
    path segmentation_config
    path inputs, name: 'measurement_files??.yaml'

    script:
    """
    pixi run --no-lockfile-update python $baseDir/s03_measure/qc.py --config $segmentation_config --measurement_files $inputs
    """
}

workflow {
    configs = PREPARE(params.convert_to_zarr_config)
    zarrs = CONVERT2ZARR(configs.flatten())
    if (new File(params.segmentation_config).exists()) {
        projections = PROJECT(
            params.segmentation_config,
            zarrs
        )
        segmentations = SEGMENT(
            params.segmentation_config,
            projections
        )
        measurements = MEASURE(
            params.segmentation_config,
            segmentations
        )
        QC(
            params.segmentation_config,
            measurements.collect()
        )
    }
}
