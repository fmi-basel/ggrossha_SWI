from pathlib import Path
from typing import Optional

import questionary
from faim_ipa.utils import IPAConfig, get_git_root


class TrainConfig(IPAConfig):
    data_zarr: Path
    output_dir: Path
    checkpoint: Optional[Path] = None
    val_split: float = 0.1
    random_seed: int = 42
    max_epochs: int
    batch_size: int
    augment: bool
    patch_size: tuple[int, int]
    depth: int
    lr: float
    zero_pad_z: tuple[int, int]

    @staticmethod
    def config_name():
        return "train_config.yaml"

    @classmethod
    def prompt(cls):
        try:
            loaded_config = cls.load()
        except FileNotFoundError:
            loaded_config = cls(
                        data_zarr=get_git_root() / "processed_data",
                output_dir=get_git_root() / "processed_data",
                checkpoint="",
                val_split=0.1,
                random_seed=42,
                max_epochs=100,
                batch_size=12,  # TODO increase batch size?
                augment=True,
                patch_size=(1024, 1024),
                depth=4,
                lr=0.0004,
            )

        data_zarr = questionary.path(
            "Data zarr:", default=str(loaded_config.data_zarr)
        ).ask()
        val_split = float(
            questionary.text(
                "Validation split ratio (0-1):",
                default=str(loaded_config.val_split),
                validate=lambda x: x.replace(".", "", 1).isdigit() and 0 <= float(x) <= 1,
            ).ask()
        )
        random_seed = int(
            questionary.text(
                "Random seed:",
                default=str(loaded_config.random_seed),
                validate=lambda x: x.isdigit(),
            ).ask()
        )
        output_dir = questionary.path(
            "Output directory:", default=str(loaded_config.output_dir)
        ).ask()
        checkpoint = questionary.path(
            "Checkpoint:",
            default=str(loaded_config.checkpoint) if loaded_config else "",
        ).ask()
        if checkpoint == "":
            checkpoint = None

        max_epochs = int(
            questionary.text(
                "Max epochs:",
                default=str(loaded_config.max_epochs),
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        batch_size = int(
            questionary.text(
                "Batch size:",
                default=str(loaded_config.batch_size),
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        augment = questionary.confirm(
            "Augment data:", default=loaded_config.augment
        ).ask()
        patch_size = tuple(
            map(
                int,
                questionary.text(
                    "Patch size:",
                    default=", ".join(map(str, loaded_config.patch_size)),
                    validate=lambda x: all(map(lambda y: y.isdigit(), x.split(", "))),
                )
                .ask()
                .split(", "),
            )
        )
        depth = int(
            questionary.text(
                "Depth:",
                default=str(loaded_config.depth),
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        lr = float(
            questionary.text(
                "Learning rate:",
                default=str(loaded_config.lr),
                validate=lambda x: x.replace(".", "", 1).isdigit(),
            ).ask()
        )

        output_dir = Path(output_dir) / "segmentation_model"
        output_dir.mkdir(exist_ok=True, parents=True)

        config = TrainConfig(
            data_zarr=Path(data_zarr),
            output_dir=output_dir,
            checkpoint=Path(checkpoint) if checkpoint else None,
            val_split=val_split,
            random_seed=random_seed,
            max_epochs=max_epochs,
            batch_size=batch_size,
            augment=augment,
            patch_size=patch_size,
            depth=depth,
            lr=lr,
        )

        return config


if __name__ == "__main__":
    config = TrainConfig.prompt()
    config.save()
