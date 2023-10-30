import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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


class shared_encoder(nn.Module):
    def __init__(self, img_ch=1):
        super(shared_encoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        return x5, x4, x3, x2, x1


class seg_decoder(nn.Module):
    def __init__(self):
        super(seg_decoder, self).__init__()
        self.conv5 = conv_block(128, 128)

        self.up4 = up_conv(ch_in=128, ch_out=64)
        self.conv4 = conv_block(ch_in=128, ch_out=64)

        self.up3 = up_conv(ch_in=64, ch_out=32)
        self.conv3 = conv_block(ch_in=64, ch_out=32)

        self.up2 = up_conv(ch_in=32, ch_out=16)
        self.conv2 = conv_block(ch_in=32, ch_out=16)

        self.up1 = up_conv(ch_in=16, ch_out=8)
        self.conv1 = conv_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x5, x4, x3, x2, x1):
        s5 = self.conv5(x5)  # [1,128,32,32]

        s_4_ = self.up4(s5)  # [1,64,64,64]
        s_4 = torch.cat((s_4_, x4), dim=1)  # [1,128,64,64]
        s4 = self.conv4(s_4)  # [1,64,64,64]

        s_3_ = self.up3(s4)  # [1,32,128,128]
        s_3 = torch.cat((s_3_, x3), dim=1)  # [1,64,128,128]
        s3 = self.conv3(s_3)  # [1,32,128,128]

        s_2_ = self.up2(s3)  # [1,16,256,256]
        s_2 = torch.cat((s_2_, x2), dim=1)  # [1,32,256,256]
        s2 = self.conv2(s_2)  # [1,16,256,256]

        s_1_ = self.up1(s2)  # [1,32,128,128]
        s_1 = torch.cat((s_1_, x1), dim=1)  # [1,64,128,128]
        s1 = self.conv1(s_1)  # [1,32,128,128]

        seg_result = self.Conv_1x1(s1)
        return seg_result, s4, s3, s2, s1


class dose_decoder(nn.Module):
    def __init__(self):
        super(dose_decoder, self).__init__()
        self.conv5 = conv_block(128, 128)
        self.conv4 = conv_block(128, 64)
        self.conv3 = conv_block(64, 32)
        self.conv2 = conv_block(32, 16)
        self.conv1 = conv_block(16, 8)
        self.att_block1 = Attention_block(128, 64, 64)
        self.att_block2 = Attention_block(64, 32, 32)
        self.att_block3 = Attention_block(32, 16, 16)
        self.att_block4 = Attention_block(16, 8, 8)

        self.Conv1x1 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x5, x4, x3, x2, x1, s4, s3, s2, s1):
        d5 = self.conv5(x5)

        d_4_ = self.att_block1(d5, s4)
        d_4 = torch.cat((d_4_, x4), dim=1)
        d4 = self.conv4(d_4)

        d_3_ = self.att_block2(d4, s3)
        d_3 = torch.cat((d_3_, x3), dim=1)
        d3 = self.conv3(d_3)

        d_2_ = self.att_block3(d3, s2)
        d_2 = torch.cat((d_2_, x2), dim=1)
        d2 = self.conv2(d_2)

        d_1_ = self.att_block4(d2, s1)
        d_1 = torch.cat((d_1_, x1), dim=1)
        d1 = self.conv1(d_1)

        d = self.Conv1x1(d1)
        dose_result = self.sigmoid(d)
        return dose_result


class UNet_SegNet_CFF(nn.Module):
    def __init__(self):
        super(UNet_SegNet_CFF, self).__init__()
        self.Enc = shared_encoder()
        self.segDec = seg_decoder()
        self.doseDec = dose_decoder()

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.Enc(x)
        seg_result, s4, s3, s2, s1 = self.segDec(x5, x4, x3, x2, x1)
        dose_result = self.doseDec(x5, x4, x3, x2, x1, s4, s3, s2, s1)
        return seg_result, dose_result


def main():
    x = torch.randn(1, 1, 512, 512)
    model = UNet_SegNet_CFF()
    seg_out, dose_out = model(x)
    print(seg_out.shape)
    print(dose_out.shape)


if __name__ == '__main__':
    main()
