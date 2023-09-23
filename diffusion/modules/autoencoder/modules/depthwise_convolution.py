from torch import nn


class DepthWiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthWiseSeparableConvolution, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         in_channels,
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         groups=in_channels,
        #     ),
        #     nn.BatchNorm2d(
        #         in_channels,
        #     ),
        #     nn.SiLU(),
        #     nn.Conv2d(
        #         in_channels,
        #         out_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),
        # )

    def forward(self, x):
        return self.conv(x)
