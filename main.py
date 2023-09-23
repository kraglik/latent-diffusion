import numpy as np
import torch

from diffusion.modules.unet import UNet


with torch.no_grad():
    torch.set_default_device('cuda')
    x = torch.randn(32, 8, 32, 32).cuda()

    unet = UNet(
        in_channels=8,
        out_channels=8,
        channels=64,
        channel_multipliers=[1, 1, 2, 2],
        attention_levels=[0, 1, 2, 3],
        num_unet_blocks=2,
        transformer_heads=4,
    ).cuda()

    model_parameters = filter(lambda p: p.requires_grad, unet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(params)
    for i in range(100):
        print(i)
        y = unet(x, time_steps=torch.ones(1).cuda(), cond=None)
