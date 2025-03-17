import os
from pathlib import Path

import questionary
from faim_ipa.utils import IPAConfig, get_git_root


class ConvertToZarrConfig(IPAConfig):
    raw_data_dir: Path = Path("/")
    output_dir: Path = Path("/")
    preview: bool = False
    channels: list[int] = [1]
    bin: int = 2
    mip: bool = True
    selection_csv: Path = Path("/selection.csv")
    legacy_compressed_tif: bool = False
    legacy_yx_spacing: float = 0.65
    legacy_z_spacing: float = 2.0

    @staticmethod
    def config_name() -> str:
        return "s01_convert_to_zarr_config.yaml"

    @classmethod
    def prompt(cls) -> "ConvertToZarrConfig":
        try:
            loaded_config = cls.load()
        except FileNotFoundError:
            loaded_config = cls(
                raw_data_dir=get_git_root() / "raw_data",
                output_dir=get_git_root() / "processed_data",
                selection_csv=Path(os.getcwd()) / "selection.csv",
            )

        raw_data_dir = questionary.path(
            "[s01]: Path to raw data directory:",
            default=str(loaded_config.raw_data_dir),
        ).ask()
        preview = questionary.confirm(
            "[s01]: Create preview?",
            default=loaded_config.preview,
        ).ask()

        channels = loaded_config.channels
        bin = loaded_config.bin
        mip = loaded_config.mip
        selection_csv = loaded_config.selection_csv
        if preview:
            channels = [
                int(c)
                for c in questionary.text(
                    "[s01]: List channels to process (comma separated):",
                    validate=lambda x: x.replace(",", "").isdigit(),
                    default=",".join([str(c) for c in loaded_config.channels]),
                )
                .ask()
                .split(",")
            ]

            bin = int(
                questionary.text(
                    "[s01]: YX bin factor:",
                    validate=lambda x: x.isdigit() and int(x) >= 1,
                    default=str(loaded_config.bin),
                ).ask()
            )

            mip = questionary.confirm(
                "[s01]: Create MIPs?",
                default=loaded_config.mip,
            ).ask()
        else:
            selection_csv = questionary.path(
                "[s01]: Path to selection.csv:",
                validate=lambda x: os.path.isfile(x),
                default=str(loaded_config.selection_csv),
            ).ask()

        legacy_compressed_tiff = questionary.confirm(
            "[s01]: Convert legacy compressed TIFF format?",
            default=loaded_config.legacy_compressed_tif,
        ).ask()

        yx_spacing = loaded_config.legacy_yx_spacing
        z_spacing = loaded_config.legacy_z_spacing
        if legacy_compressed_tiff:
            yx_spacing = float(
                questionary.text(
                    "[s01]: YX spacing (microns):",
                    validate=lambda x: x.replace(".", "").isdigit(),
                    default=str(loaded_config.legacy_yx_spacing),
                ).ask()
            )
            yx_spacing = (yx_spacing, yx_spacing)
            z_spacing = float(
                questionary.text(
                    "[s01]: Z spacing (microns):",
                    validate=lambda x: x.replace(".", "").isdigit(),
                    default=str(loaded_config.legacy_z_spacing),
                ).ask()
            )

        raw_data_dir = Path(raw_data_dir)
        output_dir = (
            get_git_root() / "processed_data" / raw_data_dir.name / "s01_zarr_data"
        )

        config = ConvertToZarrConfig(
            raw_data_dir=Path(raw_data_dir),
            output_dir=Path(output_dir),
            preview=preview,
            channels=channels,
            bin=bin,
            mip=mip,
            selection_csv=Path(selection_csv),
            legacy_compressed_tif=legacy_compressed_tiff,
            legacy_yx_spacing=yx_spacing,
            legacy_z_spacing=z_spacing,
        )

        config.output_dir.mkdir(exist_ok=True)
        if config.preview:
            config.output_dir.joinpath("preview").mkdir(exist_ok=True)

        config.save()

        return config
