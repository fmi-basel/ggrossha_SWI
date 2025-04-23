import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr
from faim_ipa.utils import get_git_root, create_logger

sys.path.append(str(get_git_root()))
from source.s02_segment.config import SegmentationConfig


def main(
    config: SegmentationConfig,
    zarr_containers: list[Path],
):
    logger = create_logger("generate_projections")

    for path in zarr_containers:
        logger.info(f"Processing {path}")
        group = zarr.open(str(path), mode="a")
        data = group["0"]
        shape = data.shape
        yx_size = data.chunks[-2:]
        mip_ds = group.create_dataset(
            "mips",
            shape=(shape[0], shape[1], shape[-2], shape[-1]),
            chunks=(1, 1, *yx_size),
            dtype=data.dtype,
            overwrite=True,
        )
        projection_slices = []
        for i in range(shape[0]):
            bf = data[i, config.brightfield_channel_index]
            yx_std = bf.std((1, 2))
            focus_slice = longest_true_slice(yx_std > yx_std.mean(0))
            logger.info(
                f"Focus slice for plane {i}: {focus_slice.start} - {focus_slice.stop}"
            )
            projection_slices.append(
                {
                    "timepoint": i,
                    "focus_slice_start": focus_slice.start,
                    "focus_slice_stop": focus_slice.stop,
                }
            )
            for ch_idx in range(shape[1]):
                if ch_idx == config.brightfield_channel_index:
                    mip_ds[i, ch_idx] = bf[focus_slice].min(0)
                else:
                    mip_ds[i, ch_idx] = data[i, ch_idx][focus_slice].max(0)

        projection_slices = pd.DataFrame(projection_slices)
        projection_slices.to_csv(path / "projection_slices.csv", index=False)

    logger.info("Done.")


def longest_true_slice(bool_array):
    bool_array = np.pad(bool_array.astype(np.int8), 1)
    diff = np.diff(bool_array)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts

    max_idx = np.argmax(lengths)
    return slice(starts[max_idx], ends[max_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="s02_segmentation_config.yaml")
    parser.add_argument("--inputs", type=str, default="s01_convert_to_zarr_result.yaml")

    args = parser.parse_args()

    config = SegmentationConfig.load(args.config)
    with open(args.inputs, "r") as f:
        inputs = [Path(f) for f in yaml.safe_load(f)]

    main(config, inputs)
