import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, num_layers):
    return nn.Sequential(*[block() for _ in range(num_layers)])


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDN, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        # Residual Dense Blocks
        self.rrdb_trunk = make_layer(lambda: RRDB(nf, gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling layers
        self.up_layers = nn.Sequential()
        for _ in range(int(scale // 2)):  # Upsample 2x for each step
            self.up_layers.add_module(
                'upsample_conv',
                nn.Conv2d(nf, nf * 4, 3, 1, 1)
            )
            self.up_layers.add_module('pixel_shuffle', nn.PixelShuffle(2))
            self.up_layers.add_module('lrelu', nn.LeakyReLU(0.2, inplace=True))

        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        fea = fea + trunk

        out = self.up_layers(fea)
        out = self.hr_conv(out)
        out = self.conv_last(out)

        return out
