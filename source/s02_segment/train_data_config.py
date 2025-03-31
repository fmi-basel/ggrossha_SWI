from pathlib import Path

import questionary
from faim_ipa.utils import IPAConfig, get_git_root


class TrainDataConfig(IPAConfig):
    raw_data_dir: Path
    annotation_dir: Path
    output_dir: Path
    brightfield_channel: int

    @staticmethod
    def config_name():
        return "train_data_config.yaml"

    @classmethod
    def prompt(cls):
        try:
            loaded_config = cls.load()
        except FileNotFoundError:
            loaded_config = cls(
                raw_data_dir=get_git_root() / "processed_data",
                annotation_dir=get_git_root() / "processed_data",
                output_dir=get_git_root() / "processed_data",
                brightfield_channel=0,
            )

        raw_data_dir = Path(
            questionary.path(
                "[train_data]: Path to raw data directory:",
                default=str(loaded_config.raw_data_dir),
            ).ask()
        )
        annotation_dir = Path(
            questionary.path(
                "[train_data]: Path to annotation directory:",
                default=str(loaded_config.annotation_dir),
            ).ask()
        )
        brightfield_channel = int(
            questionary.text(
                "[train_data]: Brightfield channel index (zero-indexed):",
                validate=lambda x: x.isdigit() and int(x) >= 0,
                default=str(loaded_config.brightfield_channel),
            ).ask()
        )
        output_dir = annotation_dir.parent / "train_data"
        output_dir.mkdir(exist_ok=True, parents=True)

        config = TrainDataConfig(
            raw_data_dir=raw_data_dir,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            brightfield_channel=brightfield_channel,
        )
        return config


if __name__ == "__main__":
    config = TrainDataConfig.prompt()
    config.save()
