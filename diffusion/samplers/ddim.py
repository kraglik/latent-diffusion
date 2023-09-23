from typing import Optional, Tuple

import numpy as np
import torch

from diffusion.models import LatentDiffusion
from diffusion.samplers.sampler import Sampler


class DDIMSampler(Sampler):
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_eta: float = 0.):
        super().__init__(model)

        self.n_steps = model.n_steps

        self.denoising_steps = n_steps
        c = self.n_steps // n_steps
        self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1

        with torch.no_grad():
            alpha_bar = self.model.alpha_bar

            self.beta = torch.linspace(0.0001, 0.02, n_steps).to(torch.float32)
            self.alpha = 1. - self.beta
            self.alpha_sqrt = torch.sqrt(self.alpha)
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.sigma2 = self.beta

            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            self.ddim_sigma = (
                ddim_eta * (((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha)) ** 0.5)
                * ((1 - self.ddim_alpha / self.ddim_alpha_prev) ** .5)
            )

            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_x_prev_and_pred_x0(
        self,
        e_t: torch.Tensor,
        index: int,
        x: torch.Tensor,
        *,
        temperature: float,
        repeat_noise: bool,
    ):

        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]

        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        if sigma == 0.:
            noise = 0.

        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * (pred_x0 + dir_xt) + sigma * noise

        return x_prev, pred_x0

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0

        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        alpha = self.ddim_alpha[t].reshape(-1, 1, 1, 1)

        return x0 * (alpha ** 0.5) + eps * ((1 - alpha) ** 0.5)

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        index: int,
    ):
        eps_theta = self.get_theta_eps(xt, t, c, uncond_cond=None, uncond_scale=1.0)

        alpha = self.ddim_alpha[index].reshape(-1, 1, 1, 1)
        alpha_prev = self.ddim_alpha_prev[index].reshape(-1, 1, 1, 1)
        sigma = self.ddim_sigma[index].reshape(-1, 1, 1, 1)

        x0 = (xt - eps_theta * (1 - alpha).sqrt()) * (1 / alpha).sqrt()
        xt_dir = ((1 - alpha_prev - sigma ** 2) ** 0.5) * eps_theta
        eps = torch.randn_like(xt) * (index != 0)

        xt_prev = alpha_prev.sqrt() * x0 + xt_dir + sigma * eps

        return xt_prev, x0, eps_theta

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

        x0 = x

        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1
            ts = x.new_full((bs,), step, dtype=torch.long)
            x, x0, eps = self.p_sample(
                x,
                ts,
                cond,
                index=index,
            )
            imgs.append(x)
            noises.append(eps)

        return x0, torch.stack(imgs[:2] + imgs[-3:], 0), torch.stack(noises[:2] + noises[-3:], 0)


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
