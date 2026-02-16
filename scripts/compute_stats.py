
import os, argparse, sys
import numpy as np
import tifffile as tiff
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import save_json
from src.datasets import _to_hwc7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", type=str, default="/mnt/data/mars/train/images",
                    help="e.g. data/train/images")
    ap.add_argument("--out_json", type=str, default="/mnt/data/mars/stats.json")
    ap.add_argument("--sample_per_image", type=int, default=4096)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    img_paths = sorted([os.path.join(args.train_images, f) for f in os.listdir(args.train_images) if f.endswith(".tif")])
    assert len(img_paths) > 0, f"No tif in {args.train_images}"

    # PROBE: print first file shapes
    if args.probe:
        x0 = _to_hwc7(tiff.imread(img_paths[0]))
        print(f"[PROBE] first train image: {os.path.basename(img_paths[0])}")
        print(f"[PROBE] shape after to_hwc7: {x0.shape}, dtype={x0.dtype}")
        print("[PROBE] per-band min/max:")
        for b in range(7):
            print(f"  band{b+1}: min={x0[...,b].min():.4f}, max={x0[...,b].max():.4f}")

    # Pass1: collect samples for percentiles
    samples = [[] for _ in range(7)]
    rng = np.random.default_rng(42)
    for p in tqdm(img_paths, desc="Pass1 sample for percentiles"):
        x = _to_hwc7(tiff.imread(p)).astype(np.float32)  # (H,W,7)
        H, W, C = x.shape
        n = min(args.sample_per_image, H*W)
        idx = rng.choice(H*W, size=n, replace=False)
        flat = x.reshape(-1, C)
        s = flat[idx]  # (n,7)
        for b in range(7):
            samples[b].append(s[:, b])

    samples = [np.concatenate(s, axis=0) for s in samples]
    p1  = [float(np.percentile(samples[b], 1))  for b in range(7)]
    p99 = [float(np.percentile(samples[b], 99)) for b in range(7)]
    print("[STATS] p1:", p1)
    print("[STATS] p99:", p99)

    # Pass2: compute mean/std after clipping to [p1,p99] (more robust)
    count = 0
    sum_ = np.zeros(7, dtype=np.float64)
    sum_sq = np.zeros(7, dtype=np.float64)

    for p in tqdm(img_paths, desc="Pass2 mean/std with clipping"):
        x = _to_hwc7(tiff.imread(p)).astype(np.float32)
        x = np.clip(x, np.array(p1, dtype=np.float32), np.array(p99, dtype=np.float32))
        flat = x.reshape(-1, 7).astype(np.float64)
        count += flat.shape[0]
        sum_ += flat.sum(axis=0)
        sum_sq += (flat ** 2).sum(axis=0)

    mean = sum_ / max(1, count)
    var = (sum_sq / max(1, count)) - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    stats = {
        "p1":  [float(v) for v in p1],
        "p99": [float(v) for v in p99],
        "mean":[float(v) for v in mean],
        "std": [float(v) for v in std],
    }
    save_json(stats, args.out_json)
    print(f"[DONE] saved stats to {args.out_json}")

if __name__ == "__main__":
    main()

"""
import os, argparse, sys
import numpy as np
import tifffile as tiff
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import save_json
from src.datasets import _to_hwc7


def clean_nodata(x: np.ndarray) -> np.ndarray:

    nodata = np.float32(-3.4028235e+38)

    x = x.astype(np.float32, copy=False)

    # Replace inf with NaN
    x = np.where(np.isfinite(x), x, np.nan)

    # Replace nodata-like extreme negatives with NaN
    x = np.where(x <= nodata / 2, np.nan, x)

    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images", type=str, default="/mnt/data/mars/train/images",
                    help="e.g. data/train/images")
    ap.add_argument("--out_json", type=str, default="/mnt/data/mars/stats.json")
    ap.add_argument("--sample_per_image", type=int, default=4096)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    img_paths = sorted(
        [os.path.join(args.train_images, f) for f in os.listdir(args.train_images) if f.endswith(".tif")]
    )
    assert len(img_paths) > 0, f"No tif in {args.train_images}"

    # PROBE: print first file shapes
    if args.probe:
        x0 = clean_nodata(_to_hwc7(tiff.imread(img_paths[0])))
        print(f"[PROBE] first train image: {os.path.basename(img_paths[0])}")
        print(f"[PROBE] shape after to_hwc7: {x0.shape}, dtype={x0.dtype}")
        print("[PROBE] per-band min/max (ignoring NaNs):")
        for b in range(7):
            xb = x0[..., b]
            xb = xb[np.isfinite(xb)]
            if xb.size == 0:
                print(f"  band{b+1}: all NaN")
            else:
                print(f"  band{b+1}: min={xb.min():.4f}, max={xb.max():.4f}")

    # Pass1: collect samples for percentiles (ignore NaNs)
    samples = [[] for _ in range(7)]
    rng = np.random.default_rng(42)

    for p in tqdm(img_paths, desc="Pass1 sample for percentiles"):
        x = clean_nodata(_to_hwc7(tiff.imread(p)))  # (H,W,7) float32 with NaNs
        H, W, C = x.shape
        n = min(args.sample_per_image, H * W)

        flat = x.reshape(-1, C)
        idx = rng.choice(H * W, size=n, replace=False)
        s = flat[idx]  # (n,7)

        for b in range(7):
            sb = s[:, b]
            sb = sb[np.isfinite(sb)]
            if sb.size > 0:
                samples[b].append(sb)

    samples = [np.concatenate(s, axis=0) if len(s) > 0 else np.array([], dtype=np.float32) for s in samples]
    assert all(s.size > 0 for s in samples), "Some bands have no valid samples after removing NoData."

    p1 = [float(np.percentile(samples[b], 1)) for b in range(7)]
    p99 = [float(np.percentile(samples[b], 99)) for b in range(7)]
    print("[STATS] p1:", p1)
    print("[STATS] p99:", p99)

    # Pass2: compute mean/std after clipping to [p1,p99], ignoring NaNs
    count = np.zeros(7, dtype=np.float64)   # per-band counts
    sum_ = np.zeros(7, dtype=np.float64)
    sum_sq = np.zeros(7, dtype=np.float64)

    p1_arr = np.array(p1, dtype=np.float32)
    p99_arr = np.array(p99, dtype=np.float32)

    for p in tqdm(img_paths, desc="Pass2 mean/std with clipping"):
        x = clean_nodata(_to_hwc7(tiff.imread(p)))  # float32 with NaNs
        x = np.clip(x, p1_arr, p99_arr)             # NaNs remain NaNs

        flat = x.reshape(-1, 7)                     # (N,7)
        mask = np.isfinite(flat)                    # (N,7)

        count += mask.sum(axis=0).astype(np.float64)
        sum_ += np.nansum(flat, axis=0).astype(np.float64)
        sum_sq += np.nansum(flat * flat, axis=0).astype(np.float64)

    count = np.maximum(count, 1.0)
    mean = sum_ / count
    var = (sum_sq / count) - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    stats = {
        "p1": [float(v) for v in p1],
        "p99": [float(v) for v in p99],
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }
    save_json(stats, args.out_json)
    print(f"[DONE] saved stats to {args.out_json}")


if __name__ == "__main__":
    main()
"""