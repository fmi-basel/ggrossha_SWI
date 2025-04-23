import dask.array
import numpy as np
import pandas as pd
from napari import Viewer
from napari.utils.notifications import show_warning
from pathlib import Path


class MoltAnnotator:
    """
    Annotator state for molt points for each worm position.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    position_zarrs : list[Path]
        List of paths to the zarr files for each worm position.
    """

    def __init__(
        self,
        viewer: Viewer,
        position_zarrs: list[Path],
        brightfield_channel_index: int = 0,
        exp_path: Path = None,
    ):
        self.viewer = viewer
        self.brightfield_channel_index = brightfield_channel_index
        self.exp_path = exp_path
        self.position_zarrs = position_zarrs
        self.current_position = -1
        if (exp_path / "molt.csv").exists():
            self.annotations = pd.read_csv(exp_path / "molt.csv", index_col=0).to_dict(
                orient="index"
            )
        else:
            self.annotations = {
                p.name: {"m1": np.nan, "m2": np.nan, "m3": np.nan, "m4": np.nan}
                for p in self.position_zarrs
            }
        self.add_next()
        self.text_overlay()

    def set_m1(self):
        """
        Record the current time-point as the first molt point.
        """
        self._set("m1")

    def set_m2(self):
        """
        Record the current time-point as the 2nd molt point.
        """
        self._set("m2")

    def set_m3(self):
        """
        Record the current time-point as the 3rd molt point.
        """
        self._set("m3")

    def set_m4(self):
        """
        Record the current time-point as the 4th molt point.
        """
        self._set("m4")

    def _set(self, molt_point: str):
        """
        Record the current time-point as a molt point.

        Parameters
        ----------
        molt_point : str
            The name of the molt point to set.
        """
        timepoint = self.viewer.dims.current_step[0]
        if self.annotations[self.viewer.layers[0].name][molt_point] == timepoint:
            self.annotations[self.viewer.layers[0].name][molt_point] = np.nan
        else:
            self.annotations[self.viewer.layers[0].name][molt_point] = timepoint
        self.text_overlay()
        self.save()

    def text_overlay(self):
        """
        Update the current state of the current worm in the napari text-overlay.
        """
        self.viewer.text_overlay.text = str(
            self.annotations[self.viewer.layers[0].name]
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
                da_img = dask.array.from_zarr(current_path, mode="r", component="mips")
                self.viewer.add_image(
                    da_img[:, self.brightfield_channel_index],
                    contrast_limits=np.quantile(
                        da_img[0, self.brightfield_channel_index].compute(), (0, 0.998)
                    ),
                    multiscale=False,
                    name=current_path.name,
                )
                self.viewer.dims.set_current_step(0, 0)
                self.current_position = min(
                    self.current_position + 1, len(self.position_zarrs) - 1
                )
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
            da_img = dask.array.from_zarr(current_path, mode="r", component="mips")
            self.viewer.add_image(
                da_img[:, self.brightfield_channel_index],
                contrast_limits=np.quantile(
                    da_img[0, self.brightfield_channel_index].compute(), (0, 0.998)
                ),
                multiscale=False,
                name=current_path.name,
            )
            self.viewer.dims.set_current_step(0, 0)
            self.current_position = max(self.current_position - 1, 0)
        else:
            show_warning("First position reached.")
        self.text_overlay()

    def get_molt_df(self):
        """
        Convert the recorded annotations into a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the recorded annotations.
        """
        df = pd.DataFrame([{"position": k, **v} for k, v in self.annotations.items()])
        return df

    def save(self):
        """
        Save the annotations to a CSV file.
        """
        self.get_molt_df().to_csv(self.exp_path / "molt.csv", index=False)
