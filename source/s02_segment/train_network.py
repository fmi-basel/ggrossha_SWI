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
            train_data_zarr=config.train_data_zarr,
            val_data_zarr=config.val_data_zarr,
        )
        if "zero_pad_z" not in model.hparams:
            model.hparams["zero_pad_z"] = config.zero_pad_z
    else:
        logger.info("Creating new model...")
        model = WormSegmentationModule(
            batch_size=config.batch_size,
            augment=config.augment,
            train_data_zarr=config.train_data_zarr,
            val_data_zarr=config.val_data_zarr,
            patch_size=config.patch_size,
            depth=config.depth,
            lr=config.lr,
            zero_pad_z=config.zero_pad_z,
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

    main(**config["train_network"])
