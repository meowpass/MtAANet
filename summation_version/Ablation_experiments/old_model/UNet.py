import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in * 2, ch_in * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_first = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv_block(1)
        self.conv2 = conv_block(4)
        self.conv3 = conv_block(16)
        self.conv4 = conv_block(64)
        self.conv5 = conv_block(256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_middle = nn.Conv2d(1024, out_channels=1024, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.att_block4 = Attention_block(1024, 256, 256)
        self.att_block3 = Attention_block(256, 64, 64)
        self.att_block2 = Attention_block(64, 16, 16)
        self.att_block1 = Attention_block(16, 4, 4)

        self.last_layer = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        x = self.conv_first(src)
        x1 = self.conv1(x)
        x_ = self.maxpool(x1)
        x2 = self.conv2(x_)
        x_ = self.maxpool(x2)
        x3 = self.conv3(x_)
        x_ = self.maxpool(x3)
        x4 = self.conv4(x_)
        x_ = self.maxpool(x4)
        x5 = self.conv5(x_)

        up5 = self.conv_middle(x5)
        up4 = self.att_block4(up5, x4)
        up3 = self.att_block3(up4, x3)
        up2 = self.att_block2(up3, x2)
        up1 = self.att_block1(up2, x1)

        last_feature = self.last_layer(up1)
        result = self.sigmoid(last_feature)
        return result


if __name__ == '__main__':
    x = torch.randn(2, 1, 512, 512)
    model = UNet()
    x = model(x)
    print(x.shape)
