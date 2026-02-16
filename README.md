# Mars Landslide Segmentation (Two-Stream SegFormer)

Train + validate + inference pipeline for **Mars landslide semantic
segmentation** using a **two-stream SegFormer** model.

------------------------------------------------------------------------

## 📂 Repository Structure

    Mars/
      scripts/
        mars_train.py
        mars_test.py
        inference_save.py
        validate_metrics.py
        compute_stats.py
        optuna_2.py
        optuna_lr_thr.py
      src/
        datasets.py
        augmentation.py
        losses.py
        metrics.py
        utils.py
        models/
          twoStreamSMP.py
          dual_unet.py
      outputs/
      best_test_preds/

------------------------------------------------------------------------

## 📊 Expected Data Layout

Default `--data_root` is:

    /mnt/data/mars

Structure:

    /mnt/data/mars/
      train/
        images/
        masks/
      val/
        images/
        masks/
      test/
        images/
      stats.json   (optional but recommended)

✅ Train/Val masks must align with image filenames.\
✅ Test split contains images only.

------------------------------------------------------------------------

## 🔀 Two-Stream Input

Your model uses:

-   **Stream A:** 5 channels\
-   **Stream B:** 2 channels\
-   **Total:** 7-band radargram

Default model:

``` python
TwoStreamSegformer(classes=1, stem_out=32, inA=5, inB=2)
```

If the dataset returns `(xA, xB, xC)`, the current pipeline uses **only
xA and xB**.

------------------------------------------------------------------------

## 🧪 Augmentations

Implemented in `src/augmentation.py`:

-   Random rotation
-   Horizontal flip
-   Vertical flip
-   Optional Gaussian noise

Default probabilities:

-   rotation: 0.5\
-   hflip: 0.5\
-   vflip: 0.5\
-   noise: 0.2

Images are expected as **7-band HWC arrays**.

------------------------------------------------------------------------

## ⚙️ Installation

Create environment and install dependencies:

``` bash
pip install -r requirements.txt
```

Minimum packages:

-   torch
-   torchvision
-   numpy
-   tqdm
-   tifffile
-   segmentation-models-pytorch (optional)

------------------------------------------------------------------------

## 📈 Compute Normalization Stats (Recommended)

``` bash
python scripts/compute_stats.py \
  --data_root /mnt/data/mars \
  --out /mnt/data/mars/stats.json
```

Training automatically loads this file if present.

------------------------------------------------------------------------

## 🚀 Training

``` bash
python scripts/mars_train.py \
  --data_root /mnt/data/mars \
  --stats_json /mnt/data/mars/stats.json \
  --out_dir outputs/exp01 \
  --epochs 801 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --num_workers 8 \
  --seed 42 \
  --amp
```

**Optimizer:** AdamW\
**Scheduler:** CosineAnnealingLR

Checkpoints saved:

    outputs/exp01/checkpoints/best.pt
    outputs/exp01/checkpoints/last.pt

------------------------------------------------------------------------

## 🔍 Inference

### Option A --- after training

`mars_train.py` automatically runs test inference with the best model.

### Option B --- manual inference

``` bash
python scripts/mars_test.py \
  --data_root /mnt/data/mars \
  --stats_json /mnt/data/mars/stats.json \
  --out_dir outputs/exp01 \
  --ckpt outputs/exp01/checkpoints/best.pt
```

Predictions are saved to:

    best_test_preds/

------------------------------------------------------------------------

## 🎯 Hyperparameter Search (Optuna)

Available scripts:

-   `scripts/optuna_2.py`
-   `scripts/optuna_lr_thr.py`

Typical search space:

-   learning rate
-   weight decay
-   threshold

------------------------------------------------------------------------

## ⚠️ Common Issues

### ❌ src import errors

Run from repo root:

``` bash
cd Mars
python scripts/mars_train.py
```

------------------------------------------------------------------------

### ❌ Wrong band shape

Augmentation expects **7-channel images**.

Check TIFF loading if you see shape errors.

------------------------------------------------------------------------

### ❌ Train high / Val low (overfitting)

Try:

-   stronger weight decay
-   fewer epochs
-   more data
-   early stopping
-   dropout in decoder

------------------------------------------------------------------------

## 👨‍💻 Maintainer

**Milkisa T. Yebasse**\
PhD Researcher --- University of Trento\
Applied AI for Radar Sounder Analysis
