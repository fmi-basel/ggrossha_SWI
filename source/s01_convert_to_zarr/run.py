"""
Run.py
======

This script expects config.yaml to be present in the current working directory.
"""

import argparse
import glob
import os
import pathlib
import re
from pathlib import Path
from typing import Optional, Union, Any

import distributed
import numpy as np
import pandas as pd
import yaml

import zarr
from distributed import wait
from faim_ipa import dask_utils
from faim_ipa.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_ipa.stitching import DaskTileStitcher
from faim_ipa.stitching.tile import Tile
from faim_ipa.utils import create_logger
from faim_ipa.visiview.acquisition import RegionAcquisitionOME, RegionAcquisitionSTK
import dask.array as da
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import _get_valid_axes, write_multiscales_metadata
from tqdm import tqdm
from rich.pretty import pretty_repr

import dask

from config import ConvertToZarrConfig

dask.config.set({"logging.distributed": "error"})


def init():
    import numcodecs

    numcodecs.blosc.use_threads = True
    numcodecs.blosc.set_nthreads(4)


class CompressedTiff(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, np.ndarray]],
        illumination_correction_matrices: Optional[dict[str, np.ndarray]],
        yx_spacing: tuple[float, float],
        z_spacing: float,
        axes: list[str] = ["c", "z", "y", "x"],
        memmap: bool = True,
    ):
        self._yx_spacing = yx_spacing
        self._z_spacing = z_spacing
        from tifffile import imread

        self.tile_shape = imread(files.iloc[0]["path"]).shape
        self._axes = axes
        self._memmap = memmap
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["time"]
            channel = row["channel"]

            from faim_ipa.visiview.acquisition import StackedTile
            from faim_ipa.stitching.tile import TilePosition

            tiles.append(
                StackedTile(
                    path=file,
                    shape=self.tile_shape,
                    position=TilePosition(
                        time=time_point,
                        channel=channel,
                        z=0,
                        y=int(row["Y"] / self._yx_spacing[0]),
                        x=int(row["X"] / self._yx_spacing[1]),
                    ),
                    memmap=self._memmap,
                )
            )

        return tiles

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def get_yx_spacing(self) -> tuple[float, float]:
        return self._yx_spacing

    def get_axes(self) -> list[str]:
        return self._axes


def main(config: ConvertToZarrConfig) -> None:
    """
    Convert raw data to zarr format.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger = create_logger(name="convert-to-zarr")
    logger.info("Config:")
    logger.info(pretty_repr(config))
    os.makedirs(config.output_dir, exist_ok=True)

    files = parse_files(config.raw_data_dir)
    output_dir = config.output_dir
    if config.preview:
        files = files[files["channel"].isin(config.channels)]
        logger.info(f"Found {len(files)} files.")

        output_dir = os.path.join(output_dir, "preview")
        os.makedirs(output_dir, exist_ok=True)
    else:
        files = filter_files(files, config.selection_csv)
        logger.info(f"Found {len(files)} files.")

    client = distributed.Client(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
        memory_limit="16GB",
        local_directory="./dask_tmp/",
    )
    logger.info(client.dashboard_link)
    client.run(init)

    outputs = []
    wells = sorted([int(w) for w in files["well"].unique()])
    ome_xml = glob.glob(os.path.join(config.raw_data_dir, "*.companion.ome"))[0]
    logger.info(f"Found ome-companion file: {ome_xml}")
    for well in tqdm(wells):
        logger.info(f"Processing well {well}...")
        if not config.legacy_compressed_tif:
            if files.iloc[0]["path"].endswith(".ome.tif"):
                worm_acquisition = RegionAcquisitionOME(
                    files=files.query(f"well == '{well}'"),
                    ome_xml=ome_xml,
                    alignment=TileAlignmentOptions.STAGE_POSITION,
                    background_correction_matrices=None,
                    illumination_correction_matrices=None,
                    axes=["t", "c", "z", "y", "x"],
                    memmap=False,
                )
            elif files.iloc[0]["path"].endswith(".stk"):
                RegionAcquisitionSTK(
                    files=files.query(f"well == '{well}'"),
                    alignment=TileAlignmentOptions.STAGE_POSITION,
                    background_correction_matrices=None,
                    illumination_correction_matrices=None,
                    axes=["t", "c", "z", "y", "x"],
                    memmap=False,
                )
            else:
                logger.warning("Unknown file format.")
        else:
            worm_acquisition = CompressedTiff(
                files=files.query(f"well == '{well}'"),
                alignment=TileAlignmentOptions.STAGE_POSITION,
                background_correction_matrices=None,
                illumination_correction_matrices=None,
                axes=["t", "c", "z", "y", "x"],
                memmap=False,
                yx_spacing=(config.legacy_yx_spacing,) * 2,
                z_spacing=config.legacy_z_spacing,
            )
        tile_shape = worm_acquisition.get_tiles()[0].load_data().shape
        stitcher = DaskTileStitcher(
            tiles=worm_acquisition.get_tiles(),
            chunk_shape=tile_shape,
            output_shape=worm_acquisition.get_shape(),
            dtype=worm_acquisition.get_dtype(),
        )
        stitched_da = stitcher.get_stitched_dask_array()

        if config.preview:
            stitched_da = da.coarsen(
                reduction=dask_utils.mean_cast_to(np.uint16),
                x=stitched_da,
                axes={3: config.bin, 4: config.bin},
                trim_excess=True,
            )
            if config.mip:
                stitched_da = stitched_da.max(axis=2)
                tile_shape = tile_shape[1:]

        stitched_da = da.rechunk(
            stitched_da,
            chunks=(
                1,
                1,
            )
            + tile_shape,
        )

        name = files.iloc[0]["name"]
        out_zarr = os.path.join(output_dir, f"{name}_s{well}.zarr")
        if os.path.exists(out_zarr):
            logger.warning(f"Zarr file {out_zarr} already exists. Skipping...")
        else:
            write_zarr(client, out_zarr, stitched_da, worm_acquisition)

        outputs.append(out_zarr)

    with open("s01_convert_to_zarr_result.yaml", "w") as f:
        yaml.safe_dump(outputs, f)

    logger.info("Done!")


def parse_files(acquisition_dir: Union[Path, str]) -> pd.DataFrame:
    """
    Parse all files in the acquisition directory.

    Parameters
    ----------
    acquisition_dir :
        Directory containing all raw data.

    Returns
    -------
    DataFrame
        Table of all files in the acquisition.
    """
    filename_re = re.compile(
        r"(?P<name>.*)_w(?P<channel>\d+)(?P<channel_name>.*)_s("
        r"?P<well>\d+)_t("
        r"?P<time>\d+).(ome.tif|tif|stk)"
    )

    files = []
    for root, _, filenames in os.walk(acquisition_dir):
        for f in filenames:
            m_filename = filename_re.fullmatch(f)
            if m_filename:
                row = m_filename.groupdict()
                row["path"] = str(Path(root) / f)
                row["channel"] = int(row["channel"])
                row["time"] = int(row["time"])
                row["X"] = 0
                row["Y"] = 0
                files.append(row)

    return pd.DataFrame(files)


def filter_files(files, selection_csv):
    """
    Filter files based on selection CSV.

    Parameters
    ----------
    files :
        All files belonging to the acquisition.
    selection_csv :
        CSV indicating if a position should be skipped and the start and end
        time point.

    Returns
    -------
        Filtered files.
    """
    selection = pd.read_csv(selection_csv, index_col=False)
    positions = selection[selection["skip"] == False]["position"].values  # noqa
    files = files[files["well"].isin([str(p) for p in positions])]
    selected = []
    for pos in positions:
        subset = files[files["well"] == str(pos)]
        start_t = selection[selection["position"] == pos]["start_t"].values[0]
        end_t = selection[selection["position"] == pos]["end_t"].values[0]
        subset = subset[subset["time"] >= start_t]
        subset = subset[subset["time"] <= end_t]
        selected.append(subset)
    if len(selected) == 0:
        raise ValueError("No files selected.")
    files = pd.concat(selected)
    return files


def write_zarr(client, zarr_path, stitched_da, worm_acquisition):
    """
    Write dask-array to ome-zarr multiscale.

    Parameters
    ----------
    client :
        Dask client used for the computation.
    zarr_path :
        Path to the output zarr file.
    stitched_da :
        Dask array containing the stitched data.
    worm_acquisition :
        Acquisition object.
    """
    store = parse_url(zarr_path, mode="w").store
    group = zarr.group(store)
    wait(
        client.persist(
            da.to_zarr(
                arr=stitched_da,
                url=group.store,
                compute=False,
                component=str(Path(group.path, "0")),
                storage_options=dict(
                    dimension_separator="/",
                    chunks=(
                        1,
                        1,
                    )
                    + stitched_da.shape[2:],
                    write_empty_chunks=False,
                ),
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                dimension_separator=group._store._dimension_separator,
            )
        )
    )
    shapes = [stitched_da.shape]
    datasets = [{"path": "0"}]
    coordinate_transformations = get_coordinate_transformations(
        acquisition=worm_acquisition,
        max_layer=0,
        mip=stitched_da.ndim == 4,
    )
    fmt = CurrentFormat()
    dims = len(shapes[0])
    fmt.validate_coordinate_transformations(
        dims, len(datasets), coordinate_transformations
    )
    for dataset, transform in zip(datasets, coordinate_transformations):
        dataset["coordinateTransformations"] = transform

    if stitched_da.ndim == 4:
        axes = ["t", "c", "y", "x"]
    else:
        axes = ["t", "c", "z", "y", "x"]

    axes = _get_valid_axes(dims, axes, fmt)
    write_multiscales_metadata(
        group,
        datasets,
        fmt,
        axes,
    )


def get_coordinate_transformations(
    acquisition: WellAcquisition,
    max_layer: int,
    yx_binning: int = 1,
    mip: bool = False,
) -> list[dict[str, Any]]:
    """
    Get ome-zarr conform coordinate transformations.

    Parameters
    ----------
    acquisition :
        Acquisition object.
    max_layer :
        Maximum layer of resolution levels.
    yx_binning :
        Binning factor for YX dimension.
    mip :
        Indicate if the data is a maximum intensity projection.

    Returns
    -------
        List of coordinate transformations.
    """
    transformations = []
    for s in range(max_layer + 1):
        if acquisition.get_z_spacing() is not None:
            if mip:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                1.0,
                                acquisition.get_yx_spacing()[0] * yx_binning * 2**s,
                                acquisition.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )
            else:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                1.0,
                                acquisition.get_z_spacing(),
                                acquisition.get_yx_spacing()[0] * yx_binning * 2**s,
                                acquisition.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )

    return transformations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="01_convert_to_zarr_config.yaml")

    args = parser.parse_args()

    config = ConvertToZarrConfig.load(config_file=pathlib.Path(args.config))

    main(config)
