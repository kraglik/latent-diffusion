import torch.nn as nn
import torch


class VGG19(nn.Module):
    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

    def load_params_(self):
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x, reduction='mean'):
        loss = (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2)

        if reduction == 'mean':
            loss = loss.mean()

        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def forward(self, x, reduction='mean'):
        x = self.conv1(x)
        loss = self.feature_loss(x, reduction=reduction)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x, reduction=reduction)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x, reduction=reduction)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x, reduction=reduction)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x, reduction=reduction)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x, reduction=reduction)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x, reduction=reduction)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x, reduction=reduction)
        # x = self.mp(self.relu(x))  # 16x16
        #
        # x = self.conv9(x)
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv10(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv11(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv12(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.mp(self.relu(x))  # 8x8
        #
        # x = self.conv13(x)
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv14(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv15(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)
        # x = self.conv16(self.relu(x))
        # loss += self.feature_loss(x, reduction=reduction)

        return loss
