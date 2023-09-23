import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torchvision.transforms import InterpolationMode

from diffusion.datasets import ImagesDataset
from diffusion.models import LatentDiffusion
from diffusion.modules import Autoencoder, Encoder, Decoder, UNet
from diffusion.samplers.ddim import DDIMSampler


def load_autoencoder(path: str) -> Autoencoder:
    encoder = Encoder(
        channels=32,
        channel_multipliers=[1, 2, 2, 3],
        in_channels=3,
        z_channels=8,
        residual_blocks_count=2,
    )
    decoder = Decoder(
        channels=32,
        channel_multipliers=[1, 2, 2, 3],
        out_channels=3,
        z_channels=8,
        residual_blocks_count=2
    )

    model = Autoencoder.load_from_checkpoint(
        path,
        encoder=encoder,
        decoder=decoder,
        embedding_channels=8,
        z_channels=8,
        learning_rate=1e-3,
        kullback_leibler_weight=1e-2,
        restoration_weight=1.0,
        perceptual_weight=2.0,
    )

    return model


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
        batch_size=16,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        prefetch_factor=16,
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator(device='cuda'),
    )

    unet = UNet(
        in_channels=8,
        out_channels=8,
        channels=64,
        channel_multipliers=[1, 1, 2, 4],
        attention_levels=[0, 1, 2, 3],
        num_unet_blocks=2,
        transformer_heads=2,
    )
    autoencoder = load_autoencoder("checkpoints/vae.ckpt")

    model = LatentDiffusion(
        unet=unet,
        autoencoder=autoencoder,
        learning_rate=5e-4,
        linear_start=0.00085,
        linear_end=0.02,
        n_steps=150,
        latent_scaling_factor=1.0,
    )
    model.set_sampler(DDIMSampler(model, n_steps=20, ddim_eta=0.0))

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
                filename="ldm",
            ),
        ],
        logger=TensorBoardLogger(
            "logs",
            name="ldm",
        ),
        log_every_n_steps=1,
        # accumulate_grad_batches=4,
        # enable_checkpointing=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    main()
