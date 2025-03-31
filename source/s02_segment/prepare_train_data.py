from pathlib import Path

import numpy as np
import zarr
from matplotlib import pyplot as plt
from numcodecs import Blosc
from numpy._typing import NDArray
from ome_zarr.io import parse_url
from rich.pretty import pretty_repr
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from tifffile import imread
from tqdm import tqdm

from faim_ipa.utils import create_logger, get_git_root
import sys

sys.path.append(str(get_git_root()))
from source.s02_segment.train_data_config import TrainDataConfig


def main(
    config: TrainDataConfig,
):
    logger = create_logger("prepare-training-data")
    logger.info("Config:")
    logger.info(pretty_repr(config))

    train_data_name = "worm-segmentation-train-data.zarr"
    val_data_name = "worm-segmentation-val-data.zarr"
    train_data_path = config.output_dir / train_data_name
    val_data_path = config.output_dir / val_data_name
    if train_data_path.exists():
        logger.info(f"Found a training data container at {train_data_path}.")
        logger.info("Will append to this container.")
        train_store = parse_url(train_data_path, mode="a").store
        train_store.key_separator = "."
        train_data_zarr = zarr.group(train_store)
        train_x_zarr = train_data_zarr["x"]
        train_y_zarr = train_data_zarr["y"]
        val_store = parse_url(val_data_path, mode="a").store
        val_store.key_separator = "."
        val_data_zarr = zarr.group(val_store)
        val_x_zarr = val_data_zarr["x"]
        val_y_zarr = val_data_zarr["y"]
    else:
        logger.info(f"Creating a new training data container at {train_data_path}.")
        train_store = parse_url(train_data_path, mode="w").store
        train_store.key_separator = "."
        train_data_zarr = zarr.group(train_store)
        train_x_zarr = train_data_zarr.create_group("x")
        train_y_zarr = train_data_zarr.create_group("y")
        val_store = parse_url(val_data_path, mode="w").store
        val_store.key_separator = "."
        val_data_zarr = zarr.group(val_store)
        val_x_zarr = val_data_zarr.create_group("x")
        val_y_zarr = val_data_zarr.create_group("y")

    annotated_files = list(Path(config.annotation_dir).glob("*-SEG.tif"))
    raw_files = [
        config.raw_data_dir / f"{f.name.replace('-SEG.tif', '.zarr')}"
        for f in annotated_files
    ]
    logger.info(f"Found {len(annotated_files)} annotated files.")

    for raw_file, annotated_file in tqdm(
        zip(raw_files, annotated_files), total=len(raw_files)
    ):
        logger.info(f"Adding annotations from {annotated_file}.")
        store = parse_url(raw_file, mode="w").store
        store.key_separator = "."
        raw_zarr = zarr.group(store)["0"]
        annotated_data = imread(annotated_file)

        for i in tqdm(range(raw_zarr.shape[0]), leave=False):
            annotated_plane = annotated_data[i : i + 1]
            if (annotated_plane > 0).sum() > 0:
                if i % 10 == 0:
                    add_to_zarr(
                        x_zarr_container=val_x_zarr,
                        y_zarr_container=val_y_zarr,
                        raw_data=raw_zarr[i : i + 1, config.brightfield_channel],
                        seg_data=annotated_plane,
                    )
                else:
                    add_to_zarr(
                        x_zarr_container=train_x_zarr,
                        y_zarr_container=train_y_zarr,
                        raw_data=raw_zarr[i : i + 1, config.brightfield_channel],
                        seg_data=annotated_plane,
                    )

    logger.info("Done.")


def pad_z_to_25(raw_data):
    """
    Ensure that the z-dimension of the raw data is 25 slices.

    Parameters
    ----------
    raw_data :
        The raw data.
    """
    z_shape = raw_data.shape[1]
    if z_shape == 25:
        return raw_data
    elif z_shape < 25:
        pre_pad = (25 - z_shape) // 2
        post_pad = 25 - z_shape - pre_pad
        return np.pad(
            raw_data, ((0, 0), (pre_pad, post_pad), (0, 0), (0, 0)), mode="constant"
        )
    else:
        pre = (z_shape - 25) // 2
        post = z_shape - 25 - pre
        return raw_data[:, pre:-post]


def visualize_sample(x, y, i, output_dir):
    idx = max(6, np.argmax(np.std(x, axis=(1, 2))))
    fig = plt.figure(figsize=(10, 3.2))
    plt.subplot(1, 4, 1)
    plt.imshow(x[idx - 6 : idx + 1].mean(0), cmap="gray")
    plt.title("Raw data")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    plt.subplot(1, 4, 2)
    plt.imshow(y[0], cmap="gray", vmin=0, vmax=1)
    plt.title("Foreground")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    plt.subplot(1, 4, 3)
    plt.imshow(y[1], cmap="gray", vmin=0, vmax=1)
    plt.title("Background")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    plt.subplot(1, 4, 4)
    plt.imshow(y[2], cmap="gray", vmin=0, vmax=1)
    plt.title("Center line")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    plt.suptitle(f"Sample {i}")
    plt.tight_layout()
    fig.savefig(output_dir / f"sample_{i}.png", dpi=150)
    plt.close(fig)


def add_to_zarr(x_zarr_container, y_zarr_container, raw_data, seg_data):
    """
    Add data to the zarr container.

    Parameters
    ----------
    x_zarr_container :
        The x container.
    y_zarr_container :
        The y container.
    raw_data :
        The raw data.
    seg_data :
        The segmentation mask.
    """
    seg_data = clean_segmentation_mask(seg_data)
    raw_data = pad_z_to_25(raw_data)
    if "0" in x_zarr_container:
        x = x_zarr_container["0"]
        y = y_zarr_container["0"]
        x.append(raw_data, axis=0)
        y.append(seg_data, axis=0)
    else:
        x = x_zarr_container.create_dataset(
            "0",
            shape=(1, 25, 1024, 1024),
            chunks=(1, 25, 1024, 1024),
            dtype=raw_data.dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            dimension_separator=".",
        )
        y = y_zarr_container.create_dataset(
            "0",
            shape=(1,) + seg_data.shape[-2:],
            chunks=(100,) + seg_data.shape[-2:],
            dtype=seg_data.dtype,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            dimension_separator=".",
        )
        x[0] = raw_data[0]
        y[0] = seg_data[0]

    proof_read_dir = config.output_dir / "proof_read"
    proof_read_dir.mkdir(exist_ok=True)
    visualize_sample(raw_data, seg_data, x.shape[0] - 1, proof_read_dir)


def clean_segmentation_mask(seg_data: NDArray) -> NDArray:
    """
    Keep only largest connected component.

    During manual annotation it can happen that single pixels are annotated
    by accident. But we know that each image contains only one worm. Hence,
    we can remove every other connectecd component.

    Parameters
    ----------
    seg_data :
        The segmentation mask.

    Returns
    -------
        Clean segmentation mask.
    """
    seg_data = label(seg_data).astype(np.uint16)
    roi_ids, counts = np.unique(seg_data, return_counts=True)
    roi_ids = roi_ids[1:]
    counts = counts[1:]
    if len(roi_ids) > 1:
        # Contains more than one connected component.
        # Keep only largest non-background component.
        seg_data = seg_data == roi_ids[np.argmax(counts)]
    return binary_fill_holes(seg_data).astype(np.uint16)


if __name__ == "__main__":
    config = TrainDataConfig.load()

    main(config)
