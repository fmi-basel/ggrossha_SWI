import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr
from faim_ipa.utils import get_git_root, create_logger
import dask.array as da

import sys

from ome_zarr.io import parse_url
from rich.pretty import pretty_repr

sys.path.append(str(get_git_root()))
from source.s02_segment.config import SegmentationConfig


def main(config: SegmentationConfig, datasets: list[dict[str, str]]):
    logger = create_logger("measure")
    logger.info("Config:")
    logger.info(pretty_repr(config))
    logger.info("Datasets:")
    logger.info(pretty_repr(datasets))
    output_dir = config.output_dir.parent / "s03_measurements"
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_files = []
    for dataset in datasets:
        raw_path = Path(dataset["raw_data"])
        seg_path = Path(dataset["segmentation"])
        logger.info("Processing dataset:")
        logger.info(f"Raw path: {raw_path}")
        logger.info(f"Segmentation path: {seg_path}")

        measurements = measure(raw_path, seg_path, config, logger)
        output_path = output_dir / f"{Path(raw_path).stem}_measurements.csv"
        logger.info(f"Saving measurements to {output_path}")
        measurements.to_csv(output_path, index=False)
        measurement_files.append(str(output_path))

    with open("measurement_files.yaml", "w") as f:
        yaml.safe_dump(measurement_files, f, sort_keys=False)


def measure(
    raw_path: Path,
    seg_path: Path,
    config: SegmentationConfig,
    logger: logging.Logger,
):
    img = da.from_zarr(raw_path, component="0")
    seg = da.from_zarr(seg_path, component="0")
    channels = list(
        filter(lambda i: i != config.brightfield_channel_index, range(img.shape[1]))
    )
    raw_zarr = zarr.Group(parse_url(raw_path, mode="r").store)
    yx_spacing = raw_zarr.attrs.asdict()["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"][-1]
    rows = []
    for ch in channels:
        logger.info(f"Processing channel {ch}")
        for t in range(img.shape[0]):
            logger.info(f"Processing timepoint {t}")
            data = img[t, ch].compute()
            focus_slice = compute_focus_planes(data)
            logger.info(f"Focus slice: {focus_slice}")
            mip = np.max(data[focus_slice], axis=0)
            mask = seg[t, 0, 0].compute()

            length = np.nan
            area = np.nan
            mean_intensity = np.nan
            bg_intensity = np.nan
            if mask.max() == 2:
                length = np.sum(mask == 2) * yx_spacing
                mask = mask > 0
                area = mask.sum() * yx_spacing**2
                mean_intensity = mip[mask].mean()
                bg_intensity = np.quantile(mip[mask], 0.05)

            rows.append(
                {
                    "channel": ch,
                    "time": t,
                    "length": length,
                    "area": area,
                    "mean_intensity": mean_intensity,
                    "percentile.05_intensity": bg_intensity,
                    "focus_slice_start": focus_slice.start,
                    "focus_slice_stop": focus_slice.stop,
                    "id": raw_path.stem,
                }
            )
    return pd.DataFrame(rows)


def compute_focus_planes(data: np.ndarray) -> slice:
    z_plane_means = np.mean(data, axis=(1, 2))
    return max_cumsum_slice(z_plane_means, np.mean(z_plane_means))


def max_cumsum_slice(values: np.ndarray, threshold: float) -> slice:
    """
    Find the slice which maximizes the sum over all values which are above a given threshold.

    Parameters
    ----------
    values:
        A 1D array of values.
    threshold:
        Threshold to exceed.

    Return
    ------
    slice
    """

    class Interval:
        def __init__(self, start, stop, score=0.0):
            self.start = start
            self.stop = stop
            self.score = score

        def reset(self, idx):
            self.start = idx
            self.stop = idx
            self.score = 0.0

    assert values.ndim == 1

    current = Interval(0, 0)
    best = Interval(0, 0)

    for idx, val in enumerate(np.sign(values - threshold)):
        current.score += val
        if current.score < 0.0:
            current.reset(idx + 1)

        elif current.score > best.score:
            best = Interval(current.start, idx + 1, current.score)

    return slice(best.start, best.stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="s02_segmentation_config.yaml",
        help="Path to the segmentation config file.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="datasets.csv",
        help="Path to the datasets file.",
    )

    args = parser.parse_args()
    config = SegmentationConfig.load(Path(args.config))
    with open(args.datasets, "r") as f:
        datasets = yaml.safe_load(f)

    main(config, datasets)
