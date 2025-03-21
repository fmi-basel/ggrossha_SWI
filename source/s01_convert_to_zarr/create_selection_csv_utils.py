from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import zarr
from glob import glob

import sys

from napari import Viewer
from faim_ipa.utils import get_git_root
from napari.utils.notifications import show_warning

sys.path.append(str(get_git_root()))
from source.s01_convert_to_zarr.config import ConvertToZarrConfig
from source.s01_convert_to_zarr.run import parse_files


def get_experiment_widget(root_dir: Path = Path.cwd()):
    """
    Returns a dropdown widget with all experiments in the root_dir/runs directory.

    Parameters
    ----------
    root_dir : Path
        The root directory of the experiments.
    """
    return widgets.Dropdown(
        options=[
            (Path(p).name, Path(p))
            for p in sorted(glob(str(root_dir / "runs" / "*")))
            if Path(p).is_dir()
        ],
        description="Experiment:",
        disabled=False,
    )


def load_positions(exp, root_dir: Path = Path.cwd(), from_input_dir: bool = True):
    """
    Load all worm positions for the selected experiment.

    Parameters
    ----------
    exp : Path
        The selected experiment.
    root_dir : Path
        The root directory of the experiments
    from_input_dir : bool
        Whether to load the positions from the input directory or processed
        output directory.

    Returns
    -------
    config : ConvertToZarrConfig
        The config file for the selected experiment.
    position_zarrs : list[Path]
        The paths to the zarr files for each worm position
    """
    ConvertToZarrConfig.reference_dir = staticmethod(lambda: root_dir)
    try:
        config = ConvertToZarrConfig.load(exp.value / ConvertToZarrConfig.config_name())
        files = parse_files(config.raw_data_dir)
        if from_input_dir:
            load_dir = config.output_dir / "preview"
            positions = sorted([int(p) for p in files["well"].unique()])
            positions_zarr = [
                load_dir / f"{files.iloc[0]["name"]}_s{p}.zarr" for p in positions
            ]
        else:
            load_dir = config.output_dir
            positions = sorted(
                glob(str(config.output_dir / "*.zarr")),
                key=lambda x: int(Path(x).name.split("_")[-1].split(".zarr")[0][1:]),
            )
            positions_zarr = [load_dir / f"{Path(p).name}" for p in positions]

        return config, positions_zarr
    except FileNotFoundError:
        print("Selected experiment has no config file.")


class Annotator:
    """
    Annotator state for selection of start and end times for each worm position.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    position_zarrs : list[Path]
        List of paths to the zarr files for each worm position.
    """

    def __init__(self, viewer: Viewer, position_zarrs: list[Path]):
        self.viewer = viewer
        self.position_zarrs = position_zarrs
        self.current_position = -1
        self.start_end_times = {
            p.name: {"start_t": 0, "end_t": 0, "skip": True}
            for p in self.position_zarrs
        }
        self.add_next()
        self.text_overlay()

    def set_start_time(self):
        """
        Record the current time-point as the start time for the current worm position.
        """
        self.start_end_times[self.viewer.layers[0].name]["start_t"] = (
            self.viewer.dims.current_step[0]
        )
        self.text_overlay()

    def set_end_time(self):
        """
        Record the current time-point as the end time for the current worm position.
        """
        self.start_end_times[self.viewer.layers[0].name]["end_t"] = (
            self.viewer.dims.current_step[0]
        )
        self.text_overlay()

    def skip(self):
        """
        Invert the skip value for the current worm position.
        """
        skip_val = self.start_end_times[self.viewer.layers[0].name]["skip"]
        self.start_end_times[self.viewer.layers[0].name]["skip"] = not skip_val
        self.text_overlay()

    def text_overlay(self):
        """
        Update the current state of the current worm in the napari text-overlay.
        """
        self.viewer.text_overlay.text = str(
            self.start_end_times[self.viewer.layers[0].name]
        )
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.color = "gold"
        self.viewer.text_overlay.font_size = 18

    def add_next(self):
        """
        Remove the current displayed worm and add the next one.
        """
        if self.current_position + 1 < len(self.position_zarrs):
            self.viewer.layers.clear()
            current_path = self.position_zarrs[self.current_position + 1]
            if current_path.exists():
                da_img = zarr.open(current_path, mode="r", path="0")
                self.viewer.add_image(
                    da_img,
                    contrast_limits=np.quantile(da_img[0, 0], (0, 0.998)),
                    multiscale=False,
                    name=current_path.name,
                )
                self.viewer.dims.set_current_step(0, 0)
                self.current_position = min(
                    self.current_position + 1, len(self.position_zarrs) - 1
                )
                self.start_end_times[current_path.name]["end_t"] = da_img.shape[0] - 1
            else:
                show_warning("File does not exist.")
        else:
            show_warning("Last position reached.")
        self.text_overlay()

    def add_previous(self):
        """
        Remove the current displayed worm and add the previous one.
        """
        if self.current_position > 0:
            self.viewer.layers.clear()
            current_path = self.position_zarrs[self.current_position - 1]
            da_img = zarr.open(current_path, mode="r", path="0")
            self.viewer.add_image(
                da_img,
                contrast_limits=np.quantile(da_img[0, 0], (0, 0.998)),
                multiscale=False,
                name=current_path.name,
            )
            self.viewer.dims.set_current_step(0, 0)
            self.current_position = max(self.current_position - 1, 0)
        else:
            show_warning("First position reached.")
        self.text_overlay()

    def get_selection_df(self):
        """
        Convert the recorded annotations into a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the recorded annotations.
        """
        df = pd.DataFrame(
            [
                {"position": k, **v, "condition": ""}
                for k, v in self.start_end_times.items()
            ]
        )
        df["skip"] = df["skip"] | (df["start_t"] == df["end_t"])
        return df
