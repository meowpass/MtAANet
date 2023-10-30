import torch
import torch.nn as nn


class Con_block1(nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv = nn.Sequential(

        )

class Conv_block2(nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 2, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv(x)
        out = torch.cat(x, x1)
        return out



