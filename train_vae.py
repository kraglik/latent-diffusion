import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torchvision.transforms import InterpolationMode

from diffusion.datasets import ImagesDataset
from diffusion.modules.autoencoder import Autoencoder, Encoder, Decoder


def main() -> None:
    # Setting "high" precision to utilize Tensor Cores
    torch.set_float32_matmul_precision("high")
    torch.set_default_device('cuda')
    torch.multiprocessing.set_start_method('spawn')

    train_dataset = ImagesDataset(
        "data/images",
        transform=T.Compose([
            T.Resize(size=256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(size=(256, 256)),
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        prefetch_factor=16,
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator(device='cuda'),
    )

    embedding_channels = 8

    encoder = Encoder(
        channels=32,
        channel_multipliers=[1, 2, 2, 3],
        in_channels=3,
        z_channels=embedding_channels,
        residual_blocks_count=2,
    )
    decoder = Decoder(
        channels=32,
        channel_multipliers=[1, 2, 2, 3],
        out_channels=3,
        z_channels=embedding_channels,
        residual_blocks_count=2
    )

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        embedding_channels=embedding_channels,
        z_channels=embedding_channels,
        learning_rate=1e-3,
        kullback_leibler_weight=1e-2,
        restoration_weight=1.0,
        perceptual_weight=2.0,
    )

    trainer = pl.Trainer(
        precision=32,
        accelerator="cuda",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=50000,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                save_top_k=1,
                every_n_epochs=1,
                monitor="loss",
                mode="min",
                dirpath="checkpoints",
                filename="vae",
            ),
        ],
        logger=TensorBoardLogger(
            "logs",
            name="vae",
        ),
        log_every_n_steps=1,
        # accumulate_grad_batches=4,
        # enable_checkpointing=False,
        # gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    main()
