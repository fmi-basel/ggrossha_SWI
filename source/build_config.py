import questionary

import sys

from faim_ipa.utils import get_git_root

sys.path.append(str(get_git_root()))
from source.s01_convert_to_zarr.config import ConvertToZarrConfig
from source.s02_segment.config import SegmentationConfig


def main() -> None:
    """
    Create config.yaml from user inputs.
    """
    create_config_for = questionary.checkbox(
        "Create config for:",
        choices=[
            "s01_convert_to_zarr",
            "s02_segment",
        ],
    ).ask()

    asked_for_config = False
    if "s01_convert_to_zarr" in create_config_for:
        s01_convert_to_zarr = ConvertToZarrConfig.prompt()
        asked_for_config = True

    if "s02_segment" in create_config_for:
        if asked_for_config:
            questionary.press_any_key_to_continue(
                "Press any key to continue to 's02_segment' config..."
            ).ask()
        _ = SegmentationConfig.prompt(s01_convert_to_zarr)


if __name__ == "__main__":
    main()
