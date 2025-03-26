from pathlib import Path

import questionary
from faim_ipa.utils import IPAConfig, get_git_root

import sys

sys.path.append(str(get_git_root()))


class PrepareAnnotationDataConfig(IPAConfig):
    zarr_data_dir: Path
    output_dir: Path
    base_name: str
    positions: list[int]
    brightfield_channel_index: int

    @staticmethod
    def config_name() -> str:
        return "prepare_training_data_config.yaml"

    @classmethod
    def prompt(cls) -> "PrepareAnnotationDataConfig":
        try:
            loaded_config = cls.load()
        except FileNotFoundError:
            loaded_config = cls(
                zarr_data_dir=get_git_root() / "processed_data",
                output_dir=get_git_root() / "processed_data",
                base_name="worm_s",
                positions=[1],
                brightfield_channel_index=0,
            )

        zarr_data_dir = questionary.path(
            "[prepare_training_data]: Path to zarr data directory:",
            default=str(loaded_config.zarr_data_dir),
        ).ask()
        output_dir = questionary.path(
            "[prepare_training_data]: Path to output directory:",
            default=str(loaded_config.output_dir).replace("prepare_training_data", ""),
        ).ask()
        base_name = questionary.text(
            "[prepare_training_data]: Base name of the datasets:",
            default=loaded_config.base_name,
        ).ask()
        positions = [
            int(c)
            for c in questionary.text(
                "[prepare_training_data]: List positions to process (comma separated):",
                validate=lambda x: x.replace(",", "").isdigit(),
                default=",".join([str(c) for c in loaded_config.positions]),
            )
            .ask()
            .split(",")
        ]
        brightfield_channel_index = int(
            questionary.text(
                "[prepare_training_data]: Brightfield channel index (zero-indexed):",
                validate=lambda x: x.isdigit() and int(x) >= 0,
                default=str(loaded_config.brightfield_channel_index),
            ).ask()
        )
        output_dir = Path(output_dir) / "annotation_data"
        output_dir.mkdir(exist_ok=True, parents=True)

        config = PrepareAnnotationDataConfig(
            zarr_data_dir=Path(zarr_data_dir),
            output_dir=output_dir,
            base_name=base_name,
            positions=positions,
            brightfield_channel_index=brightfield_channel_index,
        )

        config.save()
        return config


if __name__ == "__main__":
    config = PrepareAnnotationDataConfig.prompt()
    config.save()
