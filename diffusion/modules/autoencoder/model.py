import torch
import torch_optimizer
from torch import nn
from torch.nn import functional as fun

import pytorch_lightning as pl
from torch.nn.init import trunc_normal_

from .encoder import Encoder
from .decoder import Decoder
from .modules import GaussianSampler

from diffusion.modules.discriminator import NLayerDiscriminator
from diffusion.modules.perceptual import VGG19


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        embedding_channels: int,
        z_channels: int,
        learning_rate: float = 1e-3,
        kullback_leibler_weight: float = 1.0,
        restoration_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.kullback_leibler_weight = kullback_leibler_weight
        self.restoration_weight = restoration_weight
        self.perceptual_weight = perceptual_weight

        self.q_conv = nn.Conv2d(2 * z_channels, 2 * embedding_channels, 1)
        self.post_q_conv = nn.Conv2d(embedding_channels, z_channels, 1)
        self.log_var = nn.Parameter(torch.ones(1))

        self.perceptual = VGG19().eval()

        self.learning_rate = learning_rate

        # self.init_std: float = 0.02
        # self.apply(self._init_weights)

    def encode(self, img: torch.Tensor) -> GaussianSampler:
        z = self.encoder(img)

        return GaussianSampler(self.q_conv(z))

    def encode_sample(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode(img).sample()

    def encode_sample_with_mu_sigma(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(img)

        return GaussianSampler(self.q_conv(z)).sample()

    def decode(self, z: torch.Tensor):
        z = self.post_q_conv(z)

        return self.decoder(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        img = batch

        distribution = self.encode(img)
        z = distribution.sample()

        restored_img = self.decode(z)

        kl_loss = distribution.kl().mean()
        reconstruction_loss = fun.mse_loss(restored_img, img, reduction="sum")
        perceptual_loss = self.perceptual(torch.cat([img, restored_img], dim=0), reduction="sum")

        loss = (
            kl_loss * self.kullback_leibler_weight
            + perceptual_loss * self.perceptual_weight
            + reconstruction_loss * self.restoration_weight
        )

        self.log("kl_loss", kl_loss, on_step=True, logger=True)
        self.log("reconstruction_loss", reconstruction_loss, on_step=True, logger=True)
        self.log("perceptual_loss", perceptual_loss, on_step=True, logger=True)
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)

        if (batch_idx + 1) % 100 == 0 and batch_idx > 0:
            self.logger.experiment.add_images("actual/restored", torch.cat([batch[:1], restored_img[:1]], dim=0))

        return loss

    def configure_optimizers(self):
        opt_vae = torch.optim.AdamW(
            (
                list(self.encoder.parameters())
                + list(self.decoder.parameters())
                + list(self.q_conv.parameters())
                + list(self.post_q_conv.parameters())
                + [self.log_var]
            ),
            lr=self.learning_rate,
        )

        return opt_vae

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(fun.relu(1. - logits_real))
    loss_fake = torch.mean(fun.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)

    return d_loss


def hinge_g_loss(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def vanilla_g_loss(dis_fake):
    loss = torch.mean(fun.softplus(-dis_fake))
    return loss
