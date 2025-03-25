import os
from pathlib import Path

import questionary
from faim_ipa.utils import IPAConfig, get_git_root

import sys

sys.path.append(str(get_git_root()))

from source.s01_convert_to_zarr.config import ConvertToZarrConfig


class SegmentationConfig(IPAConfig):
    data_dir: Path
    output_dir: Path
    checkpoint: Path
    brightfield_channel_index: int
    batch_size: int

    @staticmethod
    def config_name() -> str:
        return "s02_segmentation_config.yaml"

    @classmethod
    def prompt(cls, convert_to_zarr_cfg: ConvertToZarrConfig) -> "SegmentationConfig":
        try:
            loaded_config = cls.load()
        except FileNotFoundError:
            loaded_config = cls(
                data_dir=convert_to_zarr_cfg.output_dir,
                output_dir=convert_to_zarr_cfg.output_dir.parent,
                checkpoint=get_git_root(),
                brightfield_channel_index=0,
                batch_size=10,
            )

        data_dir = questionary.path(
            "[s02]: Path to data directory:",
            default=str(loaded_config.data_dir),
        ).ask()
        output_dir = questionary.path(
            "[s02]: Path to output directory:",
            default=str(loaded_config.output_dir).replace("s02_segment", ""),
        ).ask()
        model_checkpoint = questionary.path(
            "[s02]: Path to model checkpoint:",
            validate=lambda x: os.path.isfile(x) and x.endswith(".ckpt"),
            default=str(loaded_config.checkpoint),
        ).ask()
        brightfield_channel_index = int(
            questionary.text(
                "[s02]: Brightfield channel index (zero-indexed):",
                validate=lambda x: x.isdigit() and int(x) >= 0,
                default=str(loaded_config.brightfield_channel_index),
            ).ask()
        )
        batch_size = int(
            questionary.text(
                "[s02]: Batch size:",
                validate=lambda x: x.isdigit() and int(x) >= 1,
                default=str(loaded_config.batch_size),
            ).ask()
        )

        config = SegmentationConfig(
            data_dir=Path(data_dir),
            output_dir=Path(output_dir) / "s02_segment",
            checkpoint=Path(model_checkpoint),
            brightfield_channel_index=brightfield_channel_index,
            batch_size=batch_size,
        )
        config.output_dir.mkdir(exist_ok=True)
        config.output_dir.joinpath("focus-planes").mkdir(exist_ok=True)
        config.output_dir.joinpath("quality-control").mkdir(exist_ok=True)

        config.save()
        return config
