import torch
import torch.nn as nn

# GroupNorm Group Size is a Hyperparameter


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels=8, num_groups=8):
        super().__init__()

        self.num_groups = num_groups
        self.pool = nn.MaxPool3d((2, 2, 2))

        # Encoder
        self.conv1 = self.double_conv(in_channels=in_channels, out_channels=64)
        self.conv2 = self.double_conv(in_channels=64, out_channels=128)
        self.conv3 = self.double_conv(in_channels=128, out_channels=256)
        self.conv4 = self.double_conv(in_channels=256, out_channels=512)

        # Decoder
        self.conv5 = self.double_conv(in_channels=512, out_channels=256, encoder=False)
        self.up5 = self.transpose_conv(in_channels=512, out_channels=256)
        self.conv6 = self.double_conv(in_channels=256, out_channels=128, encoder=False)
        self.up6 = self.transpose_conv(in_channels=256, out_channels=128)
        self.conv7 = self.double_conv(in_channels=128, out_channels=64, encoder=False)
        self.up7 = self.transpose_conv(in_channels=128, out_channels=64)

        self.conv8 = nn.Conv3d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0)

        # self.out_scale = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv1(x)
        # print(f'\nShape of x1: {x1.shape}')
        x2 = self.pool(x1)
        # print(f'Shape of x2 - 1: {x2.shape}')

        x2 = self.conv2(x2)
        # print(f'Shape of x2 - 2: {x2.shape}')
        x3 = self.pool(x2)
        # print(f'Shape of x3 - 1: {x3.shape}')

        x3 = self.conv3(x3)
        # print(f'Shape of x3 - 2: {x3.shape}')
        x4 = self.pool(x3)
        # print(f'Shape of x4 - 1: {x4.shape}')

        x4 = self.conv4(x4)
        # print(f'Shape of x4 - 2: {x4.shape}')

        x5 = torch.cat([self.up5(x4), x3], dim=1)
        # print(f'Shape of x5 - 1: {x5.shape}')
        x5 = self.conv5(x5)
        # print(f'Shape of x5 - 2: {x5.shape}')

        x6 = torch.cat([self.up6(x5), x2], dim=1)
        # print(f'Shape of x6 - 1: {x6.shape}')
        x6 = self.conv6(x6)
        # print(f'Shape of x6 - 2: {x6.shape}')

        x7 = torch.cat([self.up7(x6), x1], dim=1)
        # print(f'Shape of x7 - 1: {x7.shape}')
        x7 = self.conv7(x7)
        # print(f'Shape of x7 - 2: {x7.shape}')

        out = self.conv8(x7)
        # print(f'Shape of x8: {x8.shape}')
        # out = self.out_scale(out)
        # print(f'Shape of output: {out.shape}\n')

        return out

    def double_conv(self, in_channels, out_channels, mid_channels=None,
                    kernel_size=(3, 3, 3), padding=(1, 1, 1),
                    encoder=True):

        if not mid_channels:
            if encoder:
                mid_channels = out_channels // 2
                if mid_channels < in_channels:
                    mid_channels = in_channels
            else:
                mid_channels = out_channels

        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=mid_channels),
            nn.ReLU(),

            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=out_channels),
            nn.ReLU()
        )

    def transpose_conv(self, in_channels, out_channels, kernel_size=(2, 2, 2), padding=0, stride=(2, 2, 2)):
        return nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)
