import torch
import torch.nn as nn


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv7x7(nn.Module):
    def __init__(self, ch_in):
        super(Conv7x7, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)


class downsample7x7(nn.Module):
    def __init__(self):
        super(downsample7x7, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)


class Maxpooling3x3(nn.Module):
    def __init__(self):
        super(Maxpooling3x3, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.maxpool(x)


class Double_conv3x3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Double_conv3x3, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class upsample3x3(nn.Module):
    def __init__(self, ch_in):
        super(upsample3x3, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


class ConvTrans3x3(nn.Module):
    def __init__(self, ch_out):
        super(ConvTrans3x3, self).__init__()
        self.convTrans = nn.Sequential(
            nn.ConvTranspose2d(16, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convTrans(x)


class Identity_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Identity_Block, self).__init__()
        self.con1x1 = Conv1x1(in_channels, in_channels)
        self.con3x3 = Conv3x3(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.conv_ = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        pre_layer = x
        x1 = self.con1x1(x)
        x2 = self.con3x3(x1)
        x3 = self.conv(x2)
        mid_layer = torch.cat((pre_layer, x3), dim=1)
        mid_layer_ = self.conv_(mid_layer)
        out = self.relu(mid_layer_)
        return out


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.con1x1 = Conv1x1(in_channels, in_channels)
        self.con3x3 = Conv3x3(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.conv_ = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        pre_layer = x
        x1 = self.con1x1(x)
        x2 = self.con3x3(x1)
        x3 = self.conv(x2)
        pre_layer_ = self.conv(pre_layer)
        mid_layer = torch.cat((pre_layer_, x3), dim=1)
        mid_layer_ = self.conv_(mid_layer)
        out = self.relu(mid_layer_)
        return out


class U_ResNet_D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(U_ResNet_D, self).__init__()
        self.conv7x7 = Conv7x7(ch_in)
        self.downsample7x7 = downsample7x7()
        self.maxpool1 = Maxpooling3x3()
        self.conv_block1 = Conv_Block(32, 64)
        self.id_block1 = Identity_Block(64, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = Conv_Block(64, 128)
        self.id_block2 = Identity_Block(128, 128)
        self.conv_block3 = Conv_Block(128, 256)
        self.id_block3 = Identity_Block(256, 256)
        self.conv_block4 = Conv_Block(256, 512)
        self.id_block4 = Identity_Block(512, 512)

        self.middle_conv = Identity_Block(512, 512)

        self.double_conv1 = Double_conv3x3(512, 512)
        self.upsample1 = upsample3x3(512)
        self.double_conv2 = Double_conv3x3(512, 256)
        self.upsample2 = upsample3x3(256)
        self.double_conv3 = Double_conv3x3(256, 128)
        self.upsample3 = upsample3x3(128)
        self.double_conv4 = Double_conv3x3(128, 64)
        self.upsample4 = upsample3x3(64)
        self.double_conv5 = Double_conv3x3(64, 32)
        self.upsample5 = upsample3x3(32)
        self.double_conv6 = Double_conv3x3(32, 16)
        self.ConvTrans3x3 = ConvTrans3x3(1)

    def forward(self, x):
        L1 = self.conv7x7(x)
        L2 = self.downsample7x7(L1)
        L3_1 = self.maxpool1(L2)
        L3_2 = self.conv_block1(L3_1)
        L3_3 = self.id_block1(L3_2)

        L4_1_ = self.conv_block2(L3_3)
        L4_1 = self.maxpool(L4_1_)
        L4_2 = self.id_block2(L4_1)

        L5_1_ = self.conv_block3(L4_2)
        L5_1 = self.maxpool(L5_1_)
        L5_2 = self.id_block3(L5_1)

        L6_1_ = self.conv_block4(L5_2)
        L6_1 = self.maxpool(L6_1_)
        L6_2 = self.id_block4(L6_1)

        middle_layer = self.middle_conv(L6_2)

        D6 = self.double_conv1(middle_layer)
        D5_2_ = self.upsample1(D6)
        D5_2 = torch.cat((L5_2, D5_2_), dim=1)
        D5_1 = self.double_conv2(D5_2)

        D4_2_ = self.upsample2(D5_1)
        D4_2 = torch.cat((L4_2, D4_2_), dim=1)
        D4_1 = self.double_conv3(D4_2)
        D3_2_ = self.upsample3(D4_1)
        D3_2 = torch.cat((L3_2, D3_2_), dim=1)
        D3_1 = self.double_conv4(D3_2)

        D2_2_ = self.upsample4(D3_1)
        D2_2 = torch.cat((L2, D2_2_), dim=1)
        D2_1 = self.double_conv5(D2_2)

        D1_2_ = self.upsample5(D2_1)
        D1_2 = torch.cat((L1, D1_2_), dim=1)
        D1_1 = self.double_conv6(D1_2)

        out = self.ConvTrans3x3(D1_1)
        return out


if __name__ == '__main__':
    in_put = torch.randn(1, 1, 512, 512)
    model = U_ResNet_D(1, 1)
    out = model(in_put)
    print(out.shape)
