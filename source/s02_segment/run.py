import argparse
import logging
import os
from os.path import basename, join, exists
from pathlib import Path


from typing import Union

import cv2
import distributed
import numpy as np
import yaml
import zarr
from faim_ipa.utils import create_logger
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from ome_zarr.io import parse_url
from skimage.measure import regionprops, label
from skimage.filters.rank import sum as sum_filter
from skimage.morphology import skeletonize, dilation, disk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import dask.array as da
import dask

import matplotlib
from config import SegmentationConfig

matplotlib.use("agg")


dask.config.set({"logging.distributed": "error"})


class ZarrDataset(Dataset):
    def __init__(self, zarr_container, brightfield_channel_index):
        self.zarr = zarr.open(
            parse_url(zarr_container, mode="r").store,
        )[0]
        self.zarr.chunk_store.key_separator = "."
        self.shape = self.zarr.shape
        self.brightfield_channel_index = brightfield_channel_index

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        data = self.zarr[item, self.brightfield_channel_index].astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        normalized = ((data - mean) / std).astype(np.float32)
        z_shape = normalized.shape[0]
        pre_pad = (25 - z_shape) // 2
        post_pad = 25 - z_shape - pre_pad
        return np.pad(
            normalized, ((pre_pad, post_pad), (0, 0), (0, 0)), mode="constant"
        )


def load_model(model_checkpoint: Union[Path, str]):
    from network import WormSegmentationModule

    model = WormSegmentationModule.load_from_checkpoint(model_checkpoint)
    model.eval()
    model = model.to("cuda")
    return model


def init():
    import numcodecs

    numcodecs.blosc.set_nthreads(1)


def main(
    config: SegmentationConfig,
    zarr_containers: list[str],
):
    """
    Segment worms in the brightfield channel with a DynUNet.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger = create_logger(name="segment")
    logger.info(f"Config: {config}")
    logger.info(f"Inputs: {inputs}")
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(join(config.output_dir, "focus-planes"), exist_ok=True)
    os.makedirs(join(config.output_dir, "quality-control"), exist_ok=True)

    logger.info(f"Processing {len(zarr_containers)} zarr containers.")

    client = distributed.Client(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
    )
    client.run(init)

    logger.info(f"Client: {client}")
    logger.info(f"Dashboard: {client.dashboard_link}")

    model = load_model(config.checkpoint)
    outputs = []
    for zarr_container in tqdm(zarr_containers):
        name = basename(zarr_container)
        logger.info(f"Processing {name}...")

        raw_data_zarr = zarr.open(
            parse_url(zarr_container, mode="r").store,
        )[0]
        raw_shape = raw_data_zarr.shape
        output = run_worm_segmentation(
            config.batch_size,
            config.brightfield_channel_index,
            model,
            name,
            config.output_dir,
            raw_shape,
            zarr_container,
            logger=logger,
        )

        compute_focus_plane_and_qc(
            config.brightfield_channel_index, name, config.output_dir, zarr_container
        )
        outputs.append(
            {
                "raw_data": zarr_container,
                "segmentation": output,
            }
        )

    with open("s02_segment_result.yaml", "w") as f:
        yaml.safe_dump(outputs, f)

    logger.info("Done.")


def compute_focus_plane_and_qc(
    brightfield_channel_index, name, output_dir, zarr_container
):
    output_name = join(output_dir, "focus-planes", f"{name}-focus-planes.csv")
    if not exists(output_name):
        raw_data = da.from_zarr(
            zarr_container,
            component="0",
            chunks=(1, 1, 25, 1024, 1024),
        )
        post_processed = da.from_zarr(
            join(output_dir, name),
            component="0",
            chunks=(1, 1, 1, 1024, 1024),
        )
        focus_planes = da.map_blocks(
            compute_focus_plane,
            raw_data[:, brightfield_channel_index : brightfield_channel_index + 1],
            post_processed[:, :1],
            dtype=np.int8,
            chunks=(1,),
            drop_axis=(1, 2, 3, 4),
        ).persist()
        qc_images = da.map_blocks(
            create_QC_image,
            raw_data[:, brightfield_channel_index : brightfield_channel_index + 1],
            post_processed,
            da.reshape(focus_planes, (focus_planes.shape[0], 1, 1, 1, 1)),
            dtype=np.uint8,
            chunks=(1, 4, 600, 600),
            drop_axis=(1,),
        ).compute()
        writer = cv2.VideoWriter(
            join(output_dir, "quality-control", f"{name}-QC.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            1,
            qc_images.shape[-2:],
        )
        for frame in qc_images:
            writer.write(np.moveaxis(frame, 0, -1)[..., 2::-1])
        writer.release()
        fp_res = focus_planes.compute()
        df = pd.DataFrame({"time": list(range(len(fp_res))), "focus_plane": fp_res})
        df.to_csv(
            output_name,
            index=False,
        )


def run_worm_segmentation(
    batch_size,
    brightfield_channel_index,
    model,
    name,
    output_dir,
    raw_shape,
    zarr_container,
    logger: logging.Logger,
) -> str:
    output_name = join(output_dir, name)
    if not exists(output_name):

        def _worker_init_fn(worker_id):
            # Otherwise workers stay on the same physical core.
            os.sched_setaffinity(0, range(os.cpu_count()))

        dl = DataLoader(
            ZarrDataset(
                zarr_container, brightfield_channel_index=brightfield_channel_index
            ),
            batch_size=2,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )
        pp = []
        for i, batch in enumerate(dl):
            pred = predict(batch, model)
            prediction = da.from_array(
                np.array(pred[:, ::2]),
                chunks=(1, 2, 1024, 1024),
            )
            logger.info(f"prediction.shape = {prediction.shape}")

            post_processed = prediction.map_blocks(
                post_process,
                dtype=np.uint8,
                chunks=(1, 1, 1024, 1024),
            )

            sum_pp = da.expand_dims(
                post_processed.sum(1, keepdims=True, dtype=np.uint8),
                axis=1,
            )

            logger.info(f"sum_pp.shape = {sum_pp.shape}")
            pp.append(sum_pp.persist())

        store = parse_url(output_name, mode="w").store
        store.chunk_store.key_separator = "."
        da.to_zarr(
            da.concatenate(pp, axis=0).rechunk(
                (50, 1, 1, raw_shape[-2], raw_shape[-1])
            ),
            store,
            component="0",
            overwrite=True,
            write_empty_chunks=False,
            dimension_separator=".",
            compute=True,
        )
    return output_name


def parse_dirs(data_dir: Union[Path, str]) -> list[Path]:
    data_dir = Path(data_dir)
    zarr_containers = []
    for dir in data_dir.iterdir():
        if dir.is_dir():
            if dir.name.endswith(".zarr"):
                zarr_containers.append(dir)
    return zarr_containers


def predict(batch, model):
    pred = model(batch.to(model.device)).detach().cpu().numpy()
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

    return pred


def post_process(pred):
    cl_intensity = pred[0, 1]
    segmentation = label(pred[0, 0] > (0.5 * 255)).astype(np.uint8)
    unique = np.unique(segmentation)
    if len(unique) == 1 and unique[0] == 0:
        # Background only
        return np.stack([segmentation, np.zeros_like(segmentation)], axis=0)[np.newaxis]
    else:
        bg_index = np.argmax(unique == 0)
        if bg_index == 0:
            roi_index = unique[1]
        else:
            roi_index = unique[0]

        segmentation = (segmentation == roi_index).astype(np.uint8)

        center_line = skeletonize(segmentation).astype(np.uint8)

        # Remove T-join
        # 1 1 1     1 1 1
        # 0 1 0 --> 0 0 0
        # 0 1 0     0 1 0
        cl_connectivity_map = sum_filter(center_line, footprint=np.ones((3, 3)))
        cl_connectivity_map[center_line == 0] = 0

        if np.max(cl_connectivity_map) == 3:
            return np.stack([segmentation, center_line], axis=0)[np.newaxis]

        center_line[cl_connectivity_map >= 5] = 0

        # remove shortcuts
        cl_connectivity_map = sum_filter(center_line, footprint=np.ones((3, 3)))
        cl_connectivity_map[center_line == 0] = 0

        if np.max(cl_connectivity_map) == 3:
            return np.stack([segmentation, center_line], axis=0)[np.newaxis]

        nodes = (cl_connectivity_map == 4).astype(np.uint8)
        nodes[center_line == 0] = 0

        cl_segments = np.zeros_like(cl_connectivity_map)
        cl_segments[cl_connectivity_map < 4] = cl_connectivity_map[
            cl_connectivity_map < 4
        ]
        segments = label(cl_segments > 0)
        node_attribution = np.zeros_like(segments)
        for y, x in zip(*np.where(nodes > 0)):
            target_id = 0
            for i, j in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]:
                target_id = max(target_id, segments[y + i, x + j])

            node_attribution[y, x] = target_id

        segments += node_attribution
        n_segments = list(filter(None, np.unique(segments)))
        if len(n_segments) == 1:
            return np.stack([segmentation, (segments > 0).astype(np.uint8)], axis=0)[
                np.newaxis
            ]
        else:
            middle_segments = np.zeros_like(segments)
            end_segments = np.zeros_like(segments)
            for seg_id in filter(None, np.unique(segments)):
                if np.min(cl_connectivity_map[segments == seg_id]) == 3:
                    middle_segments[segments == seg_id] = seg_id
                else:
                    end_segments[segments == seg_id] = seg_id

            final_segments = np.zeros_like(segments)
            for seg in regionprops(middle_segments, intensity_image=cl_intensity):
                if seg.mean_intensity > (0.5 * 255):
                    final_segments[segments == seg.label] = 1

            # Select end segments
            cl_connectivity_map = sum_filter(
                (final_segments > 0).astype(np.uint8), footprint=np.ones((3, 3))
            )
            cl_connectivity_map[final_segments == 0] = 0
            end_points = np.where(cl_connectivity_map == 2)
            for y, x in zip(*end_points):
                target_id = 0
                for i, j in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    target_id = max(target_id, end_segments[y + i, x + j])

                final_segments[end_segments == target_id] = 1

            center_line = (final_segments > 0).astype(np.uint8)

            if center_line.max() == 0 or segmentation.max() == 0:
                return np.stack(
                    [np.zeros_like(segmentation), np.zeros_like(segmentation)], axis=0
                )[np.newaxis]

            else:
                return np.stack([segmentation, center_line], axis=0)[np.newaxis]


def compute_focus_plane(in_data, segmentation):
    img = in_data[0, 0, :]
    seg = segmentation[0, 0, 0] > 0
    if segmentation.max() > 0:
        var_roi = np.var(img[:, seg], axis=1)
        var_roi[var_roi < np.median(var_roi)] = np.median(var_roi)
        fp = np.argmax(np.diff(var_roi) < 0)
        return fp
    else:
        return np.array([-1])


def create_QC_image(in_data, segmentation, focus_plane):
    outline_cmap = ListedColormap([[0, 0, 0, 0], [0, 1, 1, 1]])
    centerline_cmap = ListedColormap([[0, 0, 0, 0], [1, 0, 1, 1]])

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(in_data[0, 0, focus_plane.squeeze()], cmap="gray")
    fg = (segmentation[0, 0, 0] >= 1).astype(np.uint8)
    outline = dilation(fg, disk(3)) - fg
    plt.imshow(outline, cmap=outline_cmap)
    cl = (segmentation[0, 0, 0] == 2).astype(np.uint8)
    plt.imshow(dilation(cl, disk(1)), cmap=centerline_cmap)
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    plt.tight_layout()
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    plt.close()
    out = np.moveaxis(image_from_plot, -1, 0)[np.newaxis]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="s02_segmentation_config.yaml")
    parser.add_argument("--inputs", type=str, default="s01_convert_to_zarr_result.yaml")

    args = parser.parse_args()

    config = SegmentationConfig.load(args.config)
    with open(args.inputs, "r") as f:
        inputs = yaml.safe_load(f)

    main(config, inputs)
