from pathlib import Path
from typing import Union

import pandas as pd

from config import ConvertToZarrConfig
from run import parse_files


def main(
    raw_data_dir: Union[Path, str],
) -> None:
    """
    Create selection.csv file for the given raw_data_dir.

    The selection.csv file can be used to indicate which raw data should be
    converted.
    """
    files = parse_files(raw_data_dir)
    if len(files) == 0:
        raise ValueError(f"No files found in {raw_data_dir}.")

    positions = [int(p) for p in files["well"].unique()]
    t_start = files["time"].min()
    t_end = files["time"].max()
    selection = pd.DataFrame(
        {
            "position": sorted(positions),
            "start_t": [t_start] * len(positions),
            "end_t": [t_end] * len(positions),
            "skip": [False] * len(positions),
            "condition": [""] * len(positions),
        }
    )

    selection.to_csv("selection.csv", index=False)


if __name__ == "__main__":
    config = ConvertToZarrConfig.load()

    main(raw_data_dir=config.raw_data_dir)
