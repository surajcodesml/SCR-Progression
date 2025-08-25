# unet_skeleton.py
# PyTorch ≥1.12
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base = base

    def forward(self, x):
        return x



















import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNetBasic(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base=64):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2,2)
        # bottleneck
        self.bottleneck = DoubleConv(base*8, base*16)
        # decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base*2, base)
        # head
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        # enforce multiples of 16 to avoid mismatches
        h, w = x.shape[-2:]
        assert h % 16 == 0 and w % 16 == 0, "H and W must be multiples of 16"
        e1 = self.enc1(x)           # H,   W
        e2 = self.enc2(self.pool(e1))   # H/2, W/2
        e3 = self.enc3(self.pool(e2))   # H/4, W/4
        e4 = self.enc4(self.pool(e3))   # H/8, W/8
        b  = self.bottleneck(self.pool(e4))  # H/16, W/16

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
