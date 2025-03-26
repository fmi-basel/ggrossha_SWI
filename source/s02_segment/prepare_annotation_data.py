from os.path import basename, join

import numpy as np
import zarr
from ome_zarr.io import parse_url
from tifffile import imwrite
from tqdm import tqdm
from micro_sam.precompute_state import precompute_state
from rich.pretty import pretty_repr

from faim_ipa.utils import create_logger
import sys
from faim_ipa.utils import get_git_root

sys.path.append(str(get_git_root()))
from source.s02_segment.annotation_data_config import PrepareAnnotationDataConfig


def main(
    config: PrepareAnnotationDataConfig,
):
    logger = create_logger("prepare-annotation-data")
    logger.info("Config:")
    logger.info(pretty_repr(config))

    zarr_dirs = list(config.zarr_data_dir.glob("*.zarr"))
    zarr_dirs = list(
        filter(
            lambda x: int(
                basename(x).replace(config.base_name, "").replace(".zarr", "")
            )
            in config.positions,
            zarr_dirs,
        )
    )

    logger.info(f"Found {len(zarr_dirs)} zarr files.")

    for zarr_dir in tqdm(zarr_dirs):
        logger.info(f"Processing {zarr_dir}")
        data = zarr.Group(parse_url(zarr_dir, mode="r").store)[0]

        logger.info("Extract raw data projections...")
        annotation_planes = []
        for i in tqdm(range(data.shape[0]), leave=False):
            stack = data[i, config.brightfield_channel_index]
            idx = max(6, np.argmax(np.std(stack, axis=(1, 2))))
            annotation_planes.append(stack[idx - 6 : idx + 1].mean(0).astype(np.uint16))

        annotation_planes = np.stack(annotation_planes)
        out_path = join(
            config.output_dir, f"{basename(zarr_dir).replace('.zarr', '.tif')}"
        )
        imwrite(out_path, annotation_planes)

        logger.info("Precomputing SAM embeddings...")
        precompute_state(
            input_path=out_path,
            output_path=out_path.replace(".tif", "-embedding.zarr"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    config = PrepareAnnotationDataConfig.load()
    main(config)
