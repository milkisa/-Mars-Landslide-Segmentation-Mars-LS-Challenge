import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.dual_unet import DualEncoderUNet

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, p=1, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(c // r, 4), 1)
        self.fc2 = nn.Conv2d(max(c // r, 4), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class TwoStreamStem(nn.Module):

    def __init__(self, inA=6, inB=1, midA=32, midB=48, midC=48, out=32):
        super().__init__()

        # stronger per-stream feature extraction
        self.stemA = nn.Sequential(
            ConvBNReLU(inA, midA, 3, 1),
            ConvBNReLU(midA, midA, 3, 1),
            SEBlock(midA, r=8),
        )
        self.stemB = nn.Sequential(
            ConvBNReLU(inB, midB, 3, 1),
            ConvBNReLU(midB, midB, 3, 1),
            SEBlock(midB, r=8),
        )

        # cross-stream gating (content-dependent fusion)
        self.gateA_to_B = nn.Sequential(nn.Conv2d(midA, midB, 1, bias=False), nn.Sigmoid())
        self.gateB_to_A = nn.Sequential(nn.Conv2d(midB, midA, 1, bias=False), nn.Sigmoid())

        # fuse + refine
        self.fuse = nn.Sequential(
            nn.Conv2d(midA + midB, out, kernel_size=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            ConvBNReLU(out, out, 3, 1),
        )

    def forward(self, xA, xB, xC=None):
        fA = self.stemA(xA)
        fB = self.stemB(xB)

        # gate each stream using the other stream
        fB = fB * self.gateA_to_B(fA)
        fA = fA * self.gateB_to_A(fB)

        return self.fuse(torch.cat([fA, fB], dim=1))





class TwoStreamSegformer(nn.Module):
    def __init__(
        self,
        encoder_name="resnet50",
        #encoder_name="resnext101_32x4d",
        encoder_weights="swsl",
        classes=1,
        stem_out=8,
        inA=5,
        inB=1,
        inC=1,
    ):
        super().__init__()
        self.stem = TwoStreamStem(inA=inA, inB=inB, midA=32, midB=48, out=stem_out)

        # IMPORTANT: Segformer now expects stem_out channels (NOT 7)
        #self.seg= DualEncoderUNet(rgb_backbone="resnet34", aux_backbone="resnet18")
        self.seg = smp.Segformer( encoder_name=encoder_name,  encoder_weights=encoder_weights,  in_channels=stem_out, classes=classes,  )

    def forward(self, xA, xB):
        x = self.stem(xA, xB)  # (B, stem_out, H, W)
        return self.seg(x)
