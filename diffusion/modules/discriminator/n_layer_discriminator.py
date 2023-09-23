import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, channels=64, n_layers=3) -> None:
        super(NLayerDiscriminator, self).__init__()
        kernel_width = 4
        pad_width = 1

        sequence = [
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=kernel_width,
                stride=2,
                padding=pad_width,
            ),
            nn.LeakyReLU(0.2, True)
        ]
        nf_multiplier = 1
        nf_multiplier_prev = 1

        for i in range(1, n_layers):
            nf_multiplier_prev = nf_multiplier
            nf_multiplier = min(2 ** i, 8)
            sequence += [
                nn.Conv2d(
                    channels * nf_multiplier_prev,
                    channels * nf_multiplier,
                    kernel_size=kernel_width,
                    stride=2,
                    padding=pad_width,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * nf_multiplier),
                nn.LeakyReLU(0.2, True)
            ]

        nf_multiplier_prev = nf_multiplier
        nf_multiplier = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                channels * nf_multiplier_prev,
                channels * nf_multiplier,
                kernel_size=kernel_width,
                stride=1,
                padding=pad_width,
                bias=False,
            ),
            nn.BatchNorm2d(channels * nf_multiplier),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(
                in_channels=channels * nf_multiplier,
                out_channels=1,
                kernel_size=kernel_width,
                stride=1,
                padding=pad_width
            ),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, xs):
        return self.model(xs)
