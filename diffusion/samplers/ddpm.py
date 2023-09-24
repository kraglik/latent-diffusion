from typing import Optional

import numpy as np
import torch

from diffusion.models import LatentDiffusion
from diffusion.samplers.sampler import Sampler


class DDPMSampler(Sampler):
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int):
        super().__init__(model)

        self.denoising_steps = n_steps
        self.time_steps = np.asarray(list(range(0, n_steps))) + 1

        with torch.no_grad():
            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(torch.float32)
            self.alpha = 1. - self.beta
            self.alpha_sqrt = torch.sqrt(self.alpha)
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.sigma = self.beta.sqrt()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        alpha = self.alpha_bar[t].reshape(-1, 1, 1, 1)

        return x0 * (alpha ** 0.5) + eps * ((1 - alpha) ** 0.5)

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        index: int | torch.Tensor,
    ):
        eps_theta = self.get_theta_eps(xt, t, c, uncond_cond=None, uncond_scale=1.0)

        alpha = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar[t].reshape(-1, 1, 1, 1)

        beta = self.beta[t].reshape(-1, 1, 1, 1)
        sigma = beta.sqrt()

        x_prev = (1 / alpha.sqrt()) * (xt - (1 - alpha) / (1 - alpha_bar).sqrt() * eps_theta)

        eps = torch.randn_like(xt) * (index != 0)
        x_prev = x_prev + sigma * eps

        return x_prev, eps_theta

    @torch.inference_mode()
    def paint(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_start: int,
        *,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
    ):
        bs = x.shape[0]

        time_steps = np.flip(self.time_steps[:t_start])

        imgs = []
        noises = []

        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1
            ts = x.new_full((bs,), step, dtype=torch.long)
            x, eps = self.p_sample(
                x,
                ts,
                cond,
                index=index,
            )
            imgs.append(x)
            noises.append(eps)

        return x, torch.stack(imgs[:2] + imgs[-3:], 0), torch.stack(noises[:2] + noises[-3:], 0)


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
