from pathlib import Path

import questionary
from faim_ipa.utils import IPAConfig, get_git_root


class TrainConfig(IPAConfig):
    train_data_zarr: Path
    val_data_zarr: Path
    output_dir: Path
    checkpoint: Path
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
                train_data_zarr=get_git_root() / "processed_data",
                val_data_zarr=get_git_root() / "processed_data",
                output_dir=get_git_root() / "processed_data",
                checkpoint=get_git_root() / "processed_data",
                max_epochs=100,
                batch_size=12,
                augment=True,
                patch_size=(1024, 1024),
                depth=4,
                lr=0.0004,
                zero_pad_z=(0, 0),
            )

        train_data_zarr = questionary.path(
            "Train data zarr:", default=str(loaded_config.train_data_zarr)
        ).ask()
        val_data_zarr = questionary.path(
            "Validation data zarr:", default=str(loaded_config.val_data_zarr)
        ).ask()
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
                default=str(loaded_config["max_epochs"]) if loaded_config else "100",
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        batch_size = int(
            questionary.text(
                "Batch size:",
                default=str(loaded_config["batch_size"]) if loaded_config else "12",
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        augment = questionary.confirm(
            "Augment data:", default=loaded_config["augment"] if loaded_config else True
        ).ask()
        patch_size = tuple(
            map(
                int,
                questionary.text(
                    "Patch size:",
                    default=", ".join(map(str, loaded_config["patch_size"]))
                    if loaded_config
                    else "1024, 1024",
                    validate=lambda x: all(map(lambda y: y.isdigit(), x.split(", "))),
                )
                .ask()
                .split(", "),
            )
        )
        depth = int(
            questionary.text(
                "Depth:",
                default=str(loaded_config["depth"]) if loaded_config else "4",
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
        )
        lr = float(
            questionary.text(
                "Learning rate:",
                default=str(loaded_config["lr"]) if loaded_config else "0.0004",
                validate=lambda x: x.replace(".", "", 1).isdigit(),
            ).ask()
        )
        zero_pad_z = tuple(
            map(
                int,
                questionary.text(
                    "Zero pad z:",
                    default=", ".join(map(str, loaded_config["zero_pad_z"]))
                    if loaded_config
                    else "0, 0",
                    validate=lambda x: all(map(lambda y: y.isdigit(), x.split(", "))),
                )
                .ask()
                .split(", "),
            )
        )

        output_dir = Path(output_dir) / "segmentation_model"
        output_dir.mkdir(exist_ok=True, parents=True)

        config = TrainConfig(
            train_data_zarr=Path(train_data_zarr),
            val_data_zarr=Path(val_data_zarr),
            output_dir=output_dir,
            checkpoint=Path(checkpoint) if checkpoint else None,
            max_epochs=max_epochs,
            batch_size=batch_size,
            augment=augment,
            patch_size=patch_size,
            depth=depth,
            lr=lr,
            zero_pad_z=zero_pad_z,
        )

        return config


if __name__ == "__main__":
    config = TrainConfig.prompt()
    config.save()
