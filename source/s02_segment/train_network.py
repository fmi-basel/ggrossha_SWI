import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from faim_ipa.utils import create_logger, get_git_root
from rich.pretty import pretty_repr

import sys

sys.path.append(str(get_git_root()))
from source.s02_segment.train_config import TrainConfig
from source.s02_segment.network import (
    LogPredictionSamplesCallback,
    WormSegmentationModule,
)


def main(config: TrainConfig):
    logger = create_logger("train_network")
    logger.info("Config:")
    logger.info(f"{pretty_repr(config)}")

    torch.set_float32_matmul_precision("high")

    if config.checkpoint and config.checkpoint.exists():
        logger.info("Loading model from checkpoint...")
        model = WormSegmentationModule.load_from_checkpoint(
            config.checkpoint,
            data_zarr=config.data_zarr,
            val_split=config.val_split,
            random_seed=config.random_seed,
            zero_pad_z=config.zero_pad_z,
            batch_size=config.batch_size,
            augment=config.augment,
            patch_size=config.patch_size,
            depth=config.depth,
            lr=config.lr
        )
    else:
        logger.info("Creating new model...")
        model = WormSegmentationModule(
            batch_size=config.batch_size,
            augment=config.augment,
            data_zarr=config.data_zarr,
            patch_size=config.patch_size,
            depth=config.depth,
            lr=config.lr,
            zero_pad_z=config.zero_pad_z,
            val_split=config.val_split,
            random_seed=config.random_seed
        )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config.max_epochs,
        callbacks=[
            LearningRateMonitor(),
            LogPredictionSamplesCallback(
                save_dir=str(config.output_dir / "validation-preview"),
            ),
            ModelCheckpoint(
                dirpath=config.output_dir,
                save_top_k=1,
                save_last=True,
                monitor="val_score_avg_epoch",
                mode="max",
            ),
        ],
        logger=[
            TensorBoardLogger(
                save_dir=config.output_dir,
            ),
        ],
    )

    logger.info("Starting training...")
    trainer.fit(model=model)

    logger.info("Done.")


if __name__ == "__main__":
    config = TrainConfig.load()

    main(config)
