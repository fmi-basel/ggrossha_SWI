import argparse
import os
from pathlib import Path

import pandas as pd
from rich.pretty import pretty_repr

import sys
from faim_ipa.utils import get_git_root, create_logger

sys.path.append(str(get_git_root()))

from source.s01_convert_to_zarr.config import ConvertToZarrConfig


def main(config: ConvertToZarrConfig):
    logger = create_logger(name="prepare-parallel-run")
    logger.info(pretty_repr(config))

    os.makedirs(config.output_dir, exist_ok=True)

    selection = pd.read_csv(config.selection_csv)
    selection = selection[~selection["skip"]]

    logger.info(f"Number of positions: {len(selection)}")

    chunk_size = max(5, len(selection) // 10)

    for i, chunk in enumerate(range(0, len(selection), chunk_size)):
        chunk_selection = selection.iloc[chunk : chunk + chunk_size]
        selection_file = Path.cwd() / f"selection_{i}.csv"
        chunk_selection.to_csv(selection_file, index=False)
        config = config.model_copy()
        config.selection_csv = selection_file
        config.save(Path.cwd() / config.config_name().replace(".yaml", f"_{i}.yaml"))

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = ConvertToZarrConfig.load(args.config)
    main(config)
