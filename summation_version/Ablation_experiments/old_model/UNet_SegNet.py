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


class channel_match(nn.Module):
    def __init__(self, in_channel):
        super(channel_match, self).__init__()
        self.channel_match = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.channel_match(x)


class shared_encoder(nn.Module):
    def __init__(self):
        super(shared_encoder, self).__init__()
        self.conv_first = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1 = conv_block(1)
        self.conv2 = conv_block(4)
        self.conv3 = conv_block(16)
        self.conv4 = conv_block(64)
        self.conv5 = conv_block(256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_first(x)
        x1 = self.conv1(x)
        x_ = self.maxpool(x1)
        x2 = self.conv2(x_)
        x_ = self.maxpool(x2)
        x3 = self.conv3(x_)
        x_ = self.maxpool(x3)
        x4 = self.conv4(x_)
        x_ = self.maxpool(x4)
        x5 = self.conv5(x_)
        return x5, x4, x3, x2, x1


class seg_decoder(nn.Module):
    def __init__(self, in_channel=1024):
        super(seg_decoder, self).__init__()
        self.conv_middle = nn.Conv2d(in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.up4 = up_conv(in_channel, in_channel // 4)
        self.CM4 = channel_match(in_channel // 2)
        self.up3 = up_conv(in_channel // 4, in_channel // 16)
        self.CM3 = channel_match(in_channel // 8)
        self.up2 = up_conv(in_channel // 16, in_channel // 64)
        self.CM2 = channel_match(in_channel // 32)
        self.up1 = up_conv(in_channel // 64, in_channel // 256)
        self.CM1 = channel_match(in_channel // 128)
        self.last_layer = nn.Conv2d(in_channel // 256, 2, kernel_size=1, stride=1)

    def forward(self, x5, x4, x3, x2, x1):
        d5 = self.conv_middle(x5)
        d4 = self.up4(d5)
        d_4 = torch.cat((x4, d4), dim=1)
        d4 = self.CM4(d_4)

        d3 = self.up3(d4)
        d_3 = torch.cat((x3, d3), dim=1)
        d3 = self.CM3(d_3)

        d2 = self.up2(d3)
        d_2 = torch.cat((x2, d2), dim=1)
        d2 = self.CM2(d_2)

        d1 = self.up1(d2)
        d_1 = torch.cat((x1, d1), dim=1)
        d1 = self.CM1(d_1)

        result = self.last_layer(d1)
        return result


class dose_decoder(nn.Module):
    def __init__(self, in_channel=1024):
        super(dose_decoder, self).__init__()
        self.conv_middle = nn.Conv2d(in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.channel_match1 = channel_match(512)
        self.channel_match2 = channel_match(128)
        self.channel_match3 = channel_match(32)
        self.channel_match4 = channel_match(8)
        self.att_block4 = Attention_block(in_channel, in_channel // 4, in_channel // 4)
        self.att_block3 = Attention_block(in_channel // 4, in_channel // 16, in_channel // 16)
        self.att_block2 = Attention_block(in_channel // 16, in_channel // 64, in_channel // 64)
        self.att_block1 = Attention_block(in_channel // 64, in_channel // 256, in_channel // 256)

        self.last_layer = nn.Conv2d(in_channel // 256, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x5, x4, x3, x2, x1):
        x5 = self.conv_middle(x5)
        up4 = self.att_block4(x5, x4)

        up3 = self.att_block3(up4, x3)

        up2 = self.att_block2(up3, x2)

        up1 = self.att_block1(up2, x1)

        result = self.last_layer(up1)
        return self.sigmoid(result)


class UNet_SegNet(nn.Module):
    def __init__(self):
        super(UNet_SegNet, self).__init__()
        self.shared_enc = shared_encoder()
        self.seg_dec = seg_decoder()
        self.dose_dec = dose_decoder()

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.shared_enc(x)
        seg_result = self.seg_dec(x5, x4, x3, x2, x1)
        dose_result = self.dose_dec(x5, x4, x3, x2, x1)
        return seg_result, dose_result


if __name__ == '__main__':
    x = torch.randn(2, 1, 512, 512)
    model = MT_architecture()
    y, z = model(x)
    print(y.shape)
    print(z.shape)