import os, glob
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
def _to_hwc7(x: np.ndarray) -> np.ndarray:
    """
    Accepts common TIFF layouts:
      - (H,W,7)
      - (7,H,W)
    Returns:
      - (H,W,7) float32
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array for image, got shape={x.shape}")
    # (7,H,W)
    if x.shape[0] == 7 and x.shape[-1] != 7:
        x = np.transpose(x, (1,2,0))
    # (H,W,7)
    if x.shape[-1] != 7:
        raise ValueError(f"Cannot interpret image shape as 7-band: shape={x.shape}")
    return x.astype(np.float32)

def _to_hw_mask(y: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (H,W)
      - (1,H,W)
      - (H,W,1)
      - (H,W,C) where C>1 (use first channel)
    Returns (H,W) float32 in {0,1}
    """
    if y.ndim == 2:
        pass
    elif y.ndim == 3:
        if y.shape[0] == 1 and y.shape[1] > 1 and y.shape[2] > 1:
            y = y[0]  # (1,H,W) -> (H,W)
        elif y.shape[-1] == 1:
            y = y[..., 0]  # (H,W,1) -> (H,W)
        else:
            y = y[..., 0]
    else:
        raise ValueError(f"Unexpected mask shape={y.shape}")

    y = (y > 0).astype(np.float32)
    return y
import numpy as np
import torch
import torch.nn.functional as F

class SelectResizeToTensor:
    def __init__(self, bands=(0, 3, 4), out_size=(1028, 1028), image_mode="bilinear"):
        self.bands = tuple(bands) if bands is not None else None
        self.out_size = out_size
        self.image_mode = image_mode

    def __call__(self, image, mask=None):


        # ---- ensure numpy ----
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        # ---- fix layout to HWC ----
        # If it's CHW (C,H,W), convert to HWC
        if image.ndim == 3 and image.shape[0] in (1, 3, 4, 7) and image.shape[-1] not in (1, 3, 4, 7):
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

        # now we assume HWC
        if self.bands is not None:
            image = image[..., list(self.bands)]  # select channels on last dim

        # ---- to torch BCHW ----
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # 1,C,H,W

        if self.out_size is not None:
            x = F.interpolate(x, size=self.out_size, mode=self.image_mode, align_corners=False)

        x = x.squeeze(0).contiguous()  # C,H,W

        if mask is None:
            return {"image": x, "mask": None}

        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()

        y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # 1,1,H,W
       
        y = y.squeeze(0).contiguous()  # 1,H,W


        return {"image": x, "mask": y}

class MarsLSDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transforms=None, stats=None, probe=False, extra_features=False,streamA_idx=( 0, 3, 4,5,6), streamB_idx=(1,2,), streamC_idx=(1,), return_streams=False,postprocess=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        assert len(self.img_paths) > 0, f"No tif found in {img_dir}"
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.stats = stats  # dict with mean/std/p1/p99 (each len=7)
        self._did_probe = False
        self.probe = probe
        self.extra_features = extra_features
        self.streamA_idx = tuple(streamA_idx)
        self.streamB_idx = tuple(streamB_idx)
        self.streamC_idx = tuple(streamC_idx)
        self.return_streams = return_streams
        self.postprocess = postprocess

    def _read_img(self, path):
        x = tiff.imread(path)
        x = _to_hwc7(x)
        return x

    def _read_mask(self, path):
        y = tiff.imread(path)
        y = _to_hw_mask(y)
        return y
    def _normalize_new(self, x):
        """
        Robust per-band normalization to [0,1] using training p1/p99.
        Assumes x is (H, W, 7).
        """
        x = x.astype(np.float32)

        if self.stats is None:
            return x

        # handle NaN / inf safely
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # load stats and reshape for broadcasting
        p1  = np.array(self.stats["p1"],  dtype=np.float32).reshape(1, 1, -1)
        p99 = np.array(self.stats["p99"], dtype=np.float32).reshape(1, 1, -1)

        # robust percentile min–max normalization
        x = np.clip(x, p1, p99)
        x = (x - p1) / (p99 - p1 + 1e-6)

        # final safety clamp
        return np.clip(x, 0.0, 1.0)

    def _normalize(self, x):
        if self.stats is None:
            return x
        mean = np.array(self.stats["mean"], dtype=np.float32)
        std  = np.array(self.stats["std"], dtype=np.float32)
        p1   = np.array(self.stats["p1"], dtype=np.float32)
        p99  = np.array(self.stats["p99"], dtype=np.float32)
        x = np.clip(x, p1, p99)
        x = (x - mean) / (std + 1e-6)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(x, -10.0, 10.0)

    def _expand_features(self, x):
        # x: (H,W,7) normalized
        eps = 1e-6
        b1, b2, b3, b4, b5, b6, b7 = [x[..., i] for i in range(7)]
        ndi_5_7 = (b5 - b7) / (b5 + b7 + eps)
        ndi_4_5 = (b4 - b5) / (b4 + b5 + eps)
        ndi_2_3 = (b2 - b3) / (b2 + b3 + eps)
        ratio_5_7 = b5 / (b7 + eps)
        ratio_2_3 = b2 / (b3 + eps)
        diff_5_7 = b5 - b7
        extra = np.stack([ndi_5_7, ndi_4_5, ndi_2_3, ratio_5_7, ratio_2_3, diff_5_7], axis=-1)
        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
        extra = np.clip(extra, -10.0, 10.0)
        return np.concatenate([x, extra.astype(np.float32)], axis=-1)
    def _select_bands(self, x, bands=(0, 3, 4, 5, 6)):
        """
        Select a subset of bands.
        x: (H, W, C)
        returns: (H, W, len(bands))
        """
        return x[..., list(bands)]
    
    def _split_streams(self, x):
        xA = x[..., list(self.streamA_idx)]  # (H,W,5)
        xB = x[..., list(self.streamB_idx)]  # (H,W,2)
        XC = x[..., list(self.streamC_idx)]  # (H,W,2)
        return xA, xB, XC

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)  # 👈 get name
        x = self._read_img(img_path)

        if self.probe and (not self._did_probe):
            self._did_probe = True
            print(f"[PROBE] image file: {os.path.basename(img_path)}")
            print(f"[PROBE] raw image shape (after to_hwc7): {x.shape}, dtype={x.dtype}")
            print(f"[PROBE] per-band min/max:")
            for b in range(7):
                print(f"  band{b+1}: min={x[...,b].min():.4f}, max={x[...,b].max():.4f}")
        
        x = self._normalize(x)
        #x = self._select_bands(x, bands=(0,1,2,4,5,6))  # select 6 bands for now
        if self.extra_features:
            x = self._expand_features(x)

        if self.mask_dir is not None:
            base = os.path.basename(img_path)
            mask_path = os.path.join(self.mask_dir, base)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            y = self._read_mask(mask_path)

            if self.probe and self._did_probe:
                # 只打印一次mask信息
                self.probe = False
                print(f"[PROBE] mask file: {os.path.basename(mask_path)}")
                print(f"[PROBE] raw mask shape: {y.shape}, unique={np.unique(y)[:10]}")

        else:
            y = None

        # albumentations expects HWC for image, HW for mask
        if self.transforms is not None:
            if y is None:
                aug = self.transforms(image=x)
                x = aug["image"]
            else:
                aug = self.transforms(image=x, mask=y)
                x, y = aug["image"], aug["mask"]



        if self.return_streams:
            xA, xB, XC = self._split_streams(x)
            xA = torch.tensor(xA).permute(2,0,1).contiguous()  # (5,H,W)
            xB = torch.tensor(xB).permute(2,0,1).contiguous()  # (2,H,W)
            XC = torch.tensor(XC).permute(2,0,1).contiguous()  # (2,H,W)
            if y is None:
                return (xA, xB, XC), os.path.basename(img_path)
            y = torch.tensor(y).unsqueeze(0).contiguous()
            return (xA, xB, XC), y,filename
        try:
            x = torch.tensor(x).permute(2, 0, 1).contiguous()  # (7,H,W)
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                x = torch.tensor(x.tolist()).permute(2, 0, 1).contiguous()
            else:
                raise RuntimeError(
                    "Failed to convert NumPy array to torch tensor. "
                    "This often happens when PyTorch was built against NumPy 1.x "
                    "but NumPy 2.x is installed. Please downgrade numpy<2 or "
                    "install a PyTorch build compatible with NumPy 2.x."
                ) from exc
        if y is None:
            return x, os.path.basename(img_path)
        try:
            y = torch.tensor(y).unsqueeze(0).contiguous()  # (1,H,W)
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                y = torch.tensor(y.tolist()).unsqueeze(0).contiguous()
            else:
                raise RuntimeError(
                    "Failed to convert NumPy mask to torch tensor. "
                    "This often happens when PyTorch was built against NumPy 1.x "
                    "but NumPy 2.x is installed. Please downgrade numpy<2 or "
                    "install a PyTorch build compatible with NumPy 2.x."
                ) from exc
        return x, y,filename
