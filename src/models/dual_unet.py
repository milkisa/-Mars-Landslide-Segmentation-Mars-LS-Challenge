import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DualEncoderUNet(nn.Module):
    """
    RGB encoder: pretrained (3ch)
    AUX encoder: scratch (4ch)
    Fusion at each stage -> UNet decoder -> 1ch logits
    """
    def __init__(self, rgb_backbone="resnet34", aux_backbone="resnet18"):
        super().__init__()
        self.rgb_encoder = timm.create_model(
            rgb_backbone, pretrained=True, features_only=True, out_indices=(0,1,2,3,4), in_chans=3
        )
        self.aux_encoder = timm.create_model(
            aux_backbone, pretrained=False, features_only=True, out_indices=(0,1,2,3,4), in_chans=4
        )

        rgb_chs = self.rgb_encoder.feature_info.channels()  # list len=5
        aux_chs = self.aux_encoder.feature_info.channels()

        # fuse convs -> map concat(rgb,aux) back to rgb_ch at each stage
        self.fuse = nn.ModuleList([
            nn.Conv2d(rgb_chs[i] + aux_chs[i], rgb_chs[i], kernel_size=1, bias=False)
            for i in range(5)
        ])

        # decoder
        # stage4 (smallest) -> stage3 -> stage2 -> stage1 -> stage0
        self.dec4 = ConvBNReLU(rgb_chs[4], 512)
        self.dec3 = DecoderBlock(512, rgb_chs[3], 256)
        self.dec2 = DecoderBlock(256, rgb_chs[2], 128)
        self.dec1 = DecoderBlock(128, rgb_chs[1], 64)
        self.dec0 = DecoderBlock(64,  rgb_chs[0], 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x7):
        # x7: (B,7,H,W) -> rgb (B,3,H,W), aux (B,4,H,W)
        rgb = x7[:, 4:7, :, :]  # bands 5,6,7
        aux = x7[:, 0:4, :, :]  # bands 1..4

        rgb_feats = self.rgb_encoder(rgb)  # list [f0..f4]
        aux_feats = self.aux_encoder(aux)

        fused = []
        for i in range(5):
            f = torch.cat([rgb_feats[i], aux_feats[i]], dim=1)
            f = self.fuse[i](f)
            fused.append(f)

        f0, f1, f2, f3, f4 = fused  # f4 smallest

        x = self.dec4(f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        x = self.dec0(x, f0)

        # upsample to input size (in case)
        x = F.interpolate(x, size=x7.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(x)
        return logits
