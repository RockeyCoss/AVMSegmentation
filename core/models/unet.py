import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import MODELS


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels,
                      kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.PReLU(),
            nn.Conv3d(mid_channels, out_channels,
                      kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.blocks(x)


class Downsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_rate=2):
        super(Downsample, self).__init__()
        self.max_pool = nn.MaxPool3d(down_rate)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.max_pool(x))


class Upsample(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 up_rate=2.0,
                 mode='trilinear',
                 align_corners=True):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=up_rate,
                              mode=mode,
                              align_corners=align_corners)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, x_skip):
        # input of shape [B, C, X, Y, Z]
        x = self.up(x)
        size_diff = [x_skip.size()[i] - x.size()[i]
                     for i in range(-1, -4, -1)]
        pad_size = []
        for diff in size_diff:
            pad_size.extend([diff // 2, diff - diff//2])
        x = F.pad(x, pad_size)
        assert x.shape[2:] == x_skip.shape[2:]
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)


class Segmentor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Segmentor, self).__init__()
        self.conv = nn.Conv3d(in_channels, num_classes,
                              kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@MODELS.registry_module()
class UNet(nn.Module):
    def __init__(self,
                 img_channels=1,
                 in_channels=16,
                 num_classes=2,
                 stage = 3):
        super(UNet, self).__init__()
        self.in_conv = ConvBlock(img_channels,
                                 in_channels)
        self.down = nn.ModuleList()
        current_channels = in_channels
        for i in range(stage):
            self.down.append(Downsample(current_channels,
                                        2 * current_channels))
            current_channels *= 2

        self.up = nn.ModuleList()
        for i in range(stage):
            self.up.append(Upsample(current_channels,
                                    current_channels // 2,
                                    current_channels // 2))
            current_channels //= 2

        self.out_conv = Segmentor(current_channels, num_classes)

    def forward(self, x):
        x = self.in_conv(x)
        downsample_features = [x, ]
        for index, block in enumerate(self.down):
            x = block(x)
            if index < len(self.down) - 1:
                downsample_features.append(x)
        feature_num = len(downsample_features)
        for index, block in enumerate(self.up, 1):
            x = block(x, downsample_features[feature_num - index])
        logit = self.out_conv(x)
        return logit


if __name__ == '__main__':
    unet = UNet()
    from torchsummary import summary
    summary(unet, (1, 96, 96, 96), device='cpu')
