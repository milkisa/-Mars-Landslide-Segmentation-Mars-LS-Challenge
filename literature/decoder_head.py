import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPPHead(nn.Module):
        def __init__(self, in_ch=256, out_ch=1, rates=(1, 12, 24, 36), use_groupnorm=True):
            super().__init__()
            def Norm(c): return nn.GroupNorm(32 if c % 32 == 0 else 16, c) if use_groupnorm else nn.BatchNorm2d(c)

            def aspp_branch(cin, cout, k=3, d=1):
                pad = d
                return nn.Sequential(
                    nn.Conv2d(cin, cout, k, padding=pad, dilation=d, bias=False),
                    Norm(cout), nn.ReLU(inplace=True)
                )

            self.b1 = nn.Sequential(nn.Conv2d(in_ch, 128, 1, bias=False), Norm(128), nn.ReLU(inplace=True))
            self.b2 = aspp_branch(in_ch, 128, 3, rates[1])
            self.b3 = aspp_branch(in_ch, 128, 3, rates[2])
            self.b4 = aspp_branch(in_ch, 128, 3, rates[3])
            self.img_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, 128, 1, bias=False),
                Norm(128), nn.ReLU(inplace=True)
            )
            self.project = nn.Sequential(
                nn.Conv2d(128*5, 256, 1, bias=False), Norm(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1, bias=False), Norm(128), nn.ReLU(inplace=True)
            )
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.head = nn.Conv2d(128, out_ch, 1)

        def forward(self, x):                         # [B,256,64,64]
            y1 = self.b1(x)
            y2 = self.b2(x)
            y3 = self.b3(x)
            y4 = self.b4(x)
            y5 = self.img_pool(x)
            y5 = F.interpolate(y5, size=x.shape[-2:], mode='bilinear', align_corners=False)
            y = torch.cat([y1, y2, y3, y4, y5], dim=1)
            y = self.project(y)                       # [B,128,64,64]
            y = self.up1(y); y = self.up2(y)          # [B,128,256,256]
            return self.head(y)                       # [B,1,256,256]



import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, d=1, groups=1, use_groupnorm=True):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(32 if c % 32 == 0 else 16, c)) if use_groupnorm else nn.BatchNorm2d
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, dilation=d, groups=groups, bias=False)
        self.norm = Norm(cout)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DWSeparableAtrous(nn.Module):
    """Depthwise separable conv with dilation (DW -> PW)."""
    def __init__(self, cin, cout, k=3, d=1, use_groupnorm=True):
        super().__init__()
        p = d
        self.dw = ConvGNReLU(cin, cin, k=k, p=p, d=d, groups=cin, use_groupnorm=use_groupnorm)
        self.pw = ConvGNReLU(cin, cout, k=1, p=0, d=1, use_groupnorm=use_groupnorm)

    def forward(self, x): return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention (no reduction, 1D conv over channel descriptor)."""
    def __init__(self, channels, k_size=5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        y = self.avg(x)                  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(1,2))  # [B,1,C]
        y = self.sig(y).transpose(1,2).unsqueeze(-1) # [B,C,1,1]
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Channel-pool (avg+max) -> 7x7 conv -> sigmoid."""
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.sig(self.conv(y))
        return x * y


class ASPPHeadStrong(nn.Module):
    """
    Stronger ASPP head with attention, dropout, and optional low-level fusion (DeepLabV3+ style).

    Args:
        in_ch: channels of high-level feature (e.g., 256)
        out_ch: number of output classes (logits)
        rates: dilation rates for ASPP branches (expects len>=4; we use [1x1, r1, r2, r3])
        low_ch: channels of low-level feature (if provided, will fuse for sharper edges)
        use_groupnorm: GroupNorm vs BatchNorm
        dropout: dropout after ASPP projection
        return_aux: if True, also returns an aux logit at 1/4 scale for deep supervision
    """
    def __init__(self, in_ch=256, out_ch=1, rates=(1, 6, 12, 18),
                 low_ch=None, use_groupnorm=True, dropout=0.1, return_aux=False):
        super().__init__()
        assert len(rates) >= 4, "rates should have at least 4 entries (e.g., (1,6,12,18))"
        r1, r2, r3 = rates[1], rates[2], rates[3]
        Norm = (lambda c: nn.GroupNorm(32 if c % 32 == 0 else 16, c)) if use_groupnorm else nn.BatchNorm2d

        # ASPP branches
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, 128, kernel_size=1, bias=False), Norm(128), nn.ReLU(inplace=True))
        self.b2 = DWSeparableAtrous(in_ch, 128, k=3, d=r1, use_groupnorm=use_groupnorm)
        self.b3 = DWSeparableAtrous(in_ch, 128, k=3, d=r2, use_groupnorm=use_groupnorm)
        self.b4 = DWSeparableAtrous(in_ch, 128, k=3, d=r3, use_groupnorm=use_groupnorm)
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 128, kernel_size=1, bias=False),
            Norm(128), nn.ReLU(inplace=True)
        )

        # Projection after concat (+ attention, + dropout)
        proj_in = 128 * 5
        self.project = nn.Sequential(
            nn.Conv2d(proj_in, 256, kernel_size=1, bias=False), Norm(256), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False), Norm(128), nn.ReLU(inplace=True)
        )
        self.eca = ECA(128)
        self.spatial_attn = SpatialAttention(k=7)

        # Optional low-level fusion (DeepLabV3+ decoder)
        self.low_ch = low_ch
        if low_ch is not None:
            self.low_proj = nn.Sequential(
                nn.Conv2d(low_ch, 48, kernel_size=1, bias=False), Norm(48), nn.ReLU(inplace=True)
            )
            self.refine = nn.Sequential(
                ConvGNReLU(48 + 128, 128, k=3, p=1, use_groupnorm=use_groupnorm),
                ConvGNReLU(128, 128, k=3, p=1, use_groupnorm=use_groupnorm)
            )

        # Heads
        self.head = nn.Conv2d(128, out_ch, kernel_size=1)
        self.return_aux = return_aux
        if return_aux:
            self.aux_head = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x, low=None, target_size=None, upsample_scale=4):
        """
        x:   [B, C=in_ch, H, W] high-level feature (e.g., 1/4 or 1/8 input size)
        low: [B, C=low_ch, H_low, W_low] low-level feature (optional, larger spatial size)
        target_size: (H_out, W_out) to upsample logits to. If None, uses scale.
        upsample_scale: factor to upsample if target_size is None.
        """
        # ASPP
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y5 = self.img_pool(x)
        y5 = F.interpolate(y5, size=x.shape[-2:], mode='bilinear', align_corners=False)

        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        y = self.project(y)
        y = self.eca(y)
        y = self.spatial_attn(y)  # [B,128,H,W]

        aux = None
        if self.return_aux:
            aux = self.aux_head(y)  # logits at current stride

        # Optional low-level fusion
        if self.low_ch is not None and low is not None:
            low_proj = self.low_proj(low)
            y_up = F.interpolate(y, size=low_proj.shape[-2:], mode='bilinear', align_corners=False)
            y = self.refine(torch.cat([y_up, low_proj], dim=1))

        logits = self.head(y)

        # Final upsample
        if target_size is not None:
            logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
            if aux is not None:
                aux = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=False)
        elif upsample_scale is not None and upsample_scale != 1:
            logits = F.interpolate(logits, scale_factor=upsample_scale, mode='bilinear', align_corners=False)
            if aux is not None:
                aux = F.interpolate(aux, scale_factor=upsample_scale, mode='bilinear', align_corners=False)

        return (logits, aux) if self.return_aux else logits
