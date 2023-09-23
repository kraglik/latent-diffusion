from typing import Optional, List

import torch

from diffusion.models import LatentDiffusion


class Sampler:
    model: LatentDiffusion
    n_steps: int

    def __init__(self, model: LatentDiffusion):
        super().__init__()

        self.model = model

    def get_theta_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        uncond_scale: float,
        uncond_cond: Optional[torch.Tensor],
    ):
        if uncond_cond is None or uncond_scale == 1.:
            return self.model(x, t, c)

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([uncond_cond, c])

        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)

        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        return e_t

    def paint(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_start: int,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        raise NotImplementedError()
