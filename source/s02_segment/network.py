import os
from pathlib import Path
from typing import Union, Any

import numpy as np
import torch
import zarr
from ome_zarr.io import parse_url
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from monai.data import DataLoader
from monai.metrics import DiceHelper
from monai.networks.nets import DynUNet
from pydantic import PositiveInt, PositiveFloat
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import dilation, disk, skeletonize
from torch.utils.data import Dataset


class WormDataset(Dataset):
    def __init__(
        self,
        zarr_file: Union[Path, str],
        patch_size: tuple[int, int] = (1024, 1024),
        overlap: bool = True,
        augment: bool = False,
        shuffle: bool = True,
    ):
        store = parse_url(zarr_file, mode="r").store
        store.key_separator = "."
        self.zarr_x = zarr.group(store)["x"]["0"]
        self.zarr_y = zarr.group(store)["y"]["0"]
        self.patches = self.select_patches(
            patch_size, overlap, augment, shuffle=shuffle
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i, y_slice, x_slice, k = self.patches[idx]
        raw = self.normalize_raw(self.zarr_x[i])[:, y_slice, x_slice]
        target = self.create_target(self.zarr_y[i][y_slice, x_slice])
        weights = self.compute_weights(target)
        if k > 0:
            raw = np.rot90(raw, k, axes=(1, 2))
            target = np.rot90(target, k, axes=(1, 2))

        return raw.copy(), (target.copy(), weights)

    @staticmethod
    def normalize_raw(img):
        return ((img - img.mean()) / img.std()).astype(np.float32)

    def select_patches(self, patch_size, overlap, augment, shuffle):
        """Extract overlapping patches with at least one pixel of
        foreground."""
        if overlap:
            overlap_factor = 2
        else:
            overlap_factor = 1

        patches = []
        for sample_index, mask in enumerate(self.zarr_y):
            for y in range(
                0, mask.shape[0] - patch_size[0] + 1, patch_size[0] // overlap_factor
            ):
                for x in range(
                    0,
                    mask.shape[1] - patch_size[1] + 1,
                    patch_size[1] // overlap_factor,
                ):
                    mask_patch = mask[y : y + patch_size[0], x : x + patch_size[1]]
                    if np.any(mask_patch):
                        patches.append(
                            (
                                sample_index,
                                slice(y, y + patch_size[0]),
                                slice(x, x + patch_size[1]),
                                0,
                            )
                        )
                        if augment:
                            for k in [1, 2, 3]:
                                patches.append(
                                    (
                                        sample_index,
                                        slice(y, y + patch_size[0]),
                                        slice(x, x + patch_size[1]),
                                        k,
                                    )
                                )

        np.random.seed(42)
        if shuffle:
            idx = np.random.permutation(len(patches))
            return [patches[i] for i in idx]
        else:
            return patches

    @staticmethod
    def compute_weights(target):
        n_fg_pixels = np.sum(target[0] > 0.5)
        fg_fraction = n_fg_pixels / target[0].size + 1e-6
        bg_fraction = np.sum(target[1] > 0.5) / target[1].size + 1e-6
        if n_fg_pixels > 0:
            cl_fraction_of_fg = np.sum(target[2] == 1) / n_fg_pixels + 1e-6
        else:
            cl_fraction_of_fg = 1e-6

        return np.array(
            [1 - fg_fraction - cl_fraction_of_fg, 1 - bg_fraction, cl_fraction_of_fg],
            dtype=np.float32,
        )[..., np.newaxis, np.newaxis]

    @staticmethod
    def encode_center_line(mask):
        cl_weighted = np.zeros_like(mask, dtype=np.uint16)
        cl_edt = distance_transform_edt(mask)
        max_distance = int(cl_edt.max())
        center_line = skeletonize(mask)
        cl_weighted[center_line] = max_distance
        for i in range(max_distance):
            next_ring = dilation(cl_weighted > 0, disk(1)).astype(np.uint8) - (
                cl_weighted > 0
            ).astype(np.uint8)
            cl_weighted[next_ring > 0] = (
                cl_edt[next_ring > 0] * (max_distance - i - 1) / max_distance
            )

        cl_weighted = cl_weighted / cl_weighted.max()
        return cl_weighted

    @staticmethod
    def one_hot_encoding(mask):
        background = np.ones_like(mask)
        fg = gaussian_filter(dilation(mask.astype(np.float32)), 2)
        one_hot = np.stack([fg, np.clip(background - fg, 0, 1)], axis=0)
        return one_hot / np.sum(one_hot, 0)

    @staticmethod
    def create_target(mask):
        one_hot = WormDataset.one_hot_encoding(mask)
        cl_weighted = WormDataset.encode_center_line(mask)
        return np.concatenate([one_hot, cl_weighted[np.newaxis]], 0).astype(np.float32)


class WormSegmentationModule(LightningModule):
    def __init__(
        self,
        batch_size: int,
        augment: bool,
        train_data_zarr: Union[Path, str],
        val_data_zarr: Union[Path, str],
        patch_size: tuple[PositiveInt, PositiveInt] = (1024, 1024),
        depth: PositiveInt = 4,
        lr: PositiveFloat = 0.0004,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unet = DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            kernel_size=(
                (3, 3),
                (3, 3),
                *((3, 3),) * (self.hparams.depth - 1),
            ),
            strides=(
                (1, 1),
                (2, 2),
                *((2, 2),) * (self.hparams.depth - 1),
            ),
            upsample_kernel_size=(
                (2, 2),
                *((2, 2),) * (self.hparams.depth - 1),
            ),
        )

        def weighted_mse(pred, gt, weights):
            return torch.mean(weights * (pred - gt) ** 2)

        self.loss = weighted_mse

    def forward(self, image):
        """
        Applies the neural network to the given image.

        Parameters
        ----------
        image : Normalized raw image data.

        Returns
        -------
        prediction
        """
        return self.unet(image)

    def training_step(self, batch, batch_idx):
        """
        A single training step which is performed on a batch.

        The input batch is a tuple of two arrays of the form [N, 25, Y, X].

        The first entry is the raw image data with the shape [N, 1, Y, X].

        The second entry is the target data with the shape [N, 3, Y, X].
        Where the last channel corresponds to the ground truth mask
        indicating where the ground truth is annotated. The second last
        channel contains a weight matrix for the loss re-weighting.

        Parameters
        ----------
        batch : (image, target)
        batch_idx : index of the batch (Not used.)

        Returns
        -------
        masked loss
        """
        raw, (gt, weights) = batch
        pred = self.unet(raw)
        loss = self.loss(pred, gt, weights)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single validation step which is performed on a validation batch.

        The Dice score is computed for the three foreground labels and
        logged. Additionally, the average Dice score over all foreground
        labels is logged and returned.

        Additionally, up to three validation images are logged from the first
        validation batch.

        Parameters
        ----------
        batch : (input, target)
        batch_idx : not used

        Returns
        -------
        Average Dice score for the foreground labels.
        """
        raw, (gt, weights) = batch
        pred = self(raw)
        pred_sm = torch.nn.functional.softmax(pred, dim=1)

        dice_score = self._compute_dice_score(gt[:, :2], pred_sm[:, :2])
        center_score = 1 - torch.mean((pred[:, 2] - gt[:, 2]) ** 2)

        avg_val_dic = self._log_validation_scores(dice_score, center_score, weights)
        return avg_val_dic

    def _log_validation_scores(self, dice_score, center_score, weights):
        w_val_dice_fg = dice_score[0] * weights[0, 0]
        w_val_dice_bg = dice_score[1] * weights[0, 1]
        w_val_score_center_line = center_score * weights[0, 2]
        self.log(
            "val_dice_foreground",
            dice_score[0],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_dice_background",
            dice_score[1],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_score_center-line",
            center_score,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        val_score_avg = w_val_dice_fg + w_val_dice_bg + w_val_score_center_line
        self.log(
            "val_score_avg",
            val_score_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_score_avg

    def _compute_dice_score(self, gt, pred):
        dice = DiceHelper(
            include_background=True,
            sigmoid=False,
            softmax=True,
            get_not_nans=False,
        )
        dice_score = dice(pred, gt)
        return dice_score

    def configure_optimizers(self):
        """
        Adam optimizer with a Reduce LR on Plateau scheduler is used.

        Returns
        -------
        dict containing optimizer and lr scheduler.
        """
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode="max", factor=0.5, patience=25
                ),
                "monitor": "val_score_avg",
                "frequency": 500,
            },
        }

    def train_dataloader(self):
        """
        Creates a pytorch data loader which consumes a `SynapseDataset`.

        Returns
        -------
        train data loader
        """

        def _worker_init_fn(worker_id):
            # Otherwise workers stay on the same physical core.
            os.sched_setaffinity(0, range(os.cpu_count()))

        return DataLoader(
            dataset=WormDataset(
                zarr_file=self.hparams.train_data_zarr,
                patch_size=self.hparams.patch_size,
                overlap=True,
                augment=self.hparams.augment,
                shuffle=True,
                zero_pad_z=self.hparams.zero_pad_z,
            ),
            num_workers=24,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            batch_size=self.hparams.batch_size,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self):
        """
        Creates a pytorch data loader which consumes a `SynapseDataset`.

        Returns
        -------
        validation data loader
        """

        def _worker_init_fn(worker_id):
            # Otherwise workers stay on the same physical core.
            os.sched_setaffinity(0, range(os.cpu_count()))

        return DataLoader(
            dataset=WormDataset(
                zarr_file=self.hparams.val_data_zarr,
                patch_size=(1024, 1024),
                overlap=False,
                augment=False,
                shuffle=True,
            ),
            num_workers=4,
            batch_size=self.hparams.batch_size,
            worker_init_fn=_worker_init_fn,
        )


class LogPredictionSamplesCallback(Callback):
    def __init__(self, save_dir: str):
        super(LogPredictionSamplesCallback, self).__init__()
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir

    cmap_outline = ListedColormap([[0, 0, 0, 0], [1, 0, 1, 1]])
    cmap_centerline = ListedColormap([[0, 0, 0, 0], [0, 1, 1, 1]])

    def get_outline(self, fg):
        return dilation(dilation(fg, disk(2)) - fg, disk(2))

    def get_center_line(self, cl):
        return dilation(skeletonize(cl), disk(2))

    def _plot_summary(self, raw, gt, pred):
        fig = plt.figure(figsize=(15, 10.5))
        plt.subplot(2, 3, 1)
        plt.imshow(raw[8:15].max(0), cmap="gray")
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Raw MIP")

        plt.subplot(2, 3, 2)
        plt.imshow(raw[8:15].max(0), cmap="gray")
        plt.imshow(
            self.get_outline((gt[0] > 0.5).astype(np.uint8)), cmap=self.cmap_outline
        )
        plt.imshow(
            self.get_center_line((gt[2] > 0.5).astype(np.uint8)),
            cmap=self.cmap_centerline,
        )
        plt.text(20, 50, "Worm", fontdict={"color": "magenta"})
        plt.text(20, 100, "Center Line", fontdict={"color": "cyan"})
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Ground Truth")

        plt.subplot(2, 3, 3)
        plt.imshow(raw[8:15].max(0), cmap="gray")
        plt.imshow(
            self.get_outline((pred[0] > 0.5).astype(np.uint8)), cmap=self.cmap_outline
        )
        plt.imshow(
            self.get_center_line((pred[2] > 0.5).astype(np.uint8)),
            cmap=self.cmap_centerline,
        )
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Prediction")

        plt.subplot(2, 3, 4)
        plt.imshow(pred[0], cmap="gray", vmin=0, vmax=1)
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Predicted - Foreground")

        plt.subplot(2, 3, 5)
        plt.imshow(pred[1], cmap="gray", vmin=0, vmax=1)
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Predicted - Background")

        plt.subplot(2, 3, 6)
        plt.imshow(pred[2], cmap="gray", vmin=0, vmax=1)
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.title("Predicted - Center Line")

        plt.tight_layout()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        plt.close()

        return image_from_plot

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            images = []
            x, (target, weights) = batch
            pred = pl_module(x)

            for i in range(min(3, x.shape[0])):
                raw = x[i].detach().cpu().numpy()
                gt = target[i].detach().cpu().numpy()
                p = pred[i].detach().cpu().numpy()

                images.append(
                    self._plot_summary(
                        raw,
                        gt,
                        p,
                    )
                )

            plt.figure(figsize=(15, 10.5 * len(images)))
            plt.imshow(np.concatenate(images, axis=0))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                f"{self._save_dir}/epoch-{str(trainer.current_epoch).zfill(5)}.png"
            )
            plt.close()
