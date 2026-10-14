# PyroSegNet: lightweight dual-encoder network for burned-area segmentation.
#
# Design:
#   - Two independent encoders (pre-fire, post-fire)
#   - Multi-scale temporal fusion (concat + difference)
#   - U-Net style decoder with skip connections
#   - ~2M parameters (far lighter than Swin-based approaches)
#
# Inputs:  pre (B, 12, H, W), post (B, 12, H, W)
# Output:  logits (B, 1, H, W)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    # Two 3x3 convolutions with BatchNorm + ReLU.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class TemporalFusion(nn.Module):
    # Fuse pre and post features: concat(pre, post, post-pre) -> 1x1 conv.
    def __init__(self, in_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )
    def forward(self, pre_feat, post_feat):
        diff = post_feat - pre_feat
        return self.fuse(torch.cat([pre_feat, post_feat, diff], dim=1))


class PyroSegNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, base_ch=32):
        super().__init__()

        # Encoders (independent weights)
        self.pre_e1 = ConvBlock(in_channels, base_ch)
        self.pre_e2 = ConvBlock(base_ch, base_ch * 2)
        self.pre_e3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pre_e4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.post_e1 = ConvBlock(in_channels, base_ch)
        self.post_e2 = ConvBlock(base_ch, base_ch * 2)
        self.post_e3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.post_e4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Temporal fusion at 4 scales
        self.fuse1 = TemporalFusion(base_ch)
        self.fuse2 = TemporalFusion(base_ch * 2)
        self.fuse3 = TemporalFusion(base_ch * 4)
        self.fuse4 = TemporalFusion(base_ch * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, pre, post):
        p1 = self.pre_e1(pre);      q1 = self.post_e1(post)
        p2 = self.pre_e2(self.pool(p1)); q2 = self.post_e2(self.pool(q1))
        p3 = self.pre_e3(self.pool(p2)); q3 = self.post_e3(self.pool(q2))
        p4 = self.pre_e4(self.pool(p3)); q4 = self.post_e4(self.pool(q3))

        f1 = self.fuse1(p1, q1)
        f2 = self.fuse2(p2, q2)
        f3 = self.fuse3(p3, q3)
        f4 = self.fuse4(p4, q4)

        d4 = self.dec4(torch.cat([self.up4(f4), f3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), f2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f1], dim=1))
        d1 = self.dec1(torch.cat([d2, p1], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    model = PyroSegNet(in_channels=12, out_channels=1)
    pre = torch.randn(2, 12, 256, 256)
    post = torch.randn(2, 12, 256, 256)
    out = model(pre, post)
    print(f"Input:  {tuple(pre.shape)}")
    print(f"Output: {tuple(out.shape)}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    assert out.shape == (2, 1, 256, 256)
    print("Shape check passed.")
