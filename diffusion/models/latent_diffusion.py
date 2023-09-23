import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as fun

from diffusion.modules.autoencoder import Autoencoder
from diffusion.modules.unet import UNet


class LatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        unet: UNet,
        autoencoder: Autoencoder,
        latent_scaling_factor: float,
        n_steps: int,
        linear_start: float,
        linear_end: float,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.unet = unet
        self.autoencoder = autoencoder.eval()

        self.n_steps = n_steps

        self.latent_scaling_factor = latent_scaling_factor
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

        self.learning_rate = learning_rate

        self.sampler = None

    def set_sampler(self, sampler):
        self.sampler = sampler

    def encode_sample(self, image: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(image).sample() * self.latent_scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, cond: torch.Tensor):
        return self.unet(x, time_embedding, cond)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if batch_idx % 100 == 0:
            self.restore_image(batch[:1])
            self.generate_image(size=(32, 32))

        return self.calculate_loss(batch)

    def calculate_loss(self, img):
        times = torch.randint(0, self.sampler.denoising_steps, (img.shape[0],), device=self.device, dtype=torch.long)

        z = self.encode_sample(img)
        noise = torch.randn_like(z)

        x = self.sampler.q_sample(z, t=times, eps=noise)
        eps_theta = self(x, times, cond=None)

        loss = fun.mse_loss(eps_theta, noise, reduction='sum')

        self.log('loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    @torch.inference_mode()
    def restore_image(self, img) -> None:
        times = torch.randint(
            self.sampler.denoising_steps // 2,
            self.sampler.denoising_steps,
            (img.shape[0],),
            device=self.device,
            dtype=torch.long,
        )

        z = self.encode_sample(img)
        noise = torch.randn_like(z)
        x = self.sampler.q_sample(z, eps=noise, t=times)

        images, steps, noises = self.sampler.paint(x, None, self.sampler.denoising_steps- 1)

        images = self.decode(images)
        steps = self.decode(steps.squeeze(1))
        noises = self.decode(noises.squeeze(1))

        images = images.clamp(0, 1)
        steps = steps.clamp(0, 1)
        noises = noises.clamp(0, 1)

        self.logger.experiment.add_images("restoration_denoised/original", torch.cat([images, img], dim=0))
        self.logger.experiment.add_images("restoration_steps", steps)
        self.logger.experiment.add_images("restoration_noises", noises)

    @torch.inference_mode()
    def generate_image(self, size: (int, int)) -> None:
        z = torch.randn(1, 8, *size, device=self.device)

        images, steps, noises = self.sampler.paint(z, None, self.sampler.denoising_steps - 1)

        images = self.decode(images)
        steps = self.decode(steps.squeeze(1))
        noises = self.decode(noises.squeeze(1))

        images = images.clamp(0, 1)
        steps = steps.clamp(0, 1)
        noises = noises.clamp(0, 1)

        self.logger.experiment.add_images("generation_denoised", images)
        self.logger.experiment.add_images("generation_steps", steps)
        self.logger.experiment.add_images("generation_noises", noises)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
