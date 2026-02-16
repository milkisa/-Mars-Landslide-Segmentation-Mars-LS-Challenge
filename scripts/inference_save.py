import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch


def run_test_inference(model, loader, device, tag="best", thresh=0.5):
    model.eval()

    out_png  = os.path.join("best_test_preds", "png")
    out_tiff = os.path.join("best_test_preds", "tiff")

    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_tiff, exist_ok=True)


    with torch.no_grad():
        for batch_idx, (x, filename) in enumerate(loader):

            # ----------------------------
            # Load inputs (streams or single)
            # ----------------------------
            if isinstance(x, (tuple, list)):
                xA, xB, xC = x
                xA = xA.to(device, non_blocking=True)
                xB = xB.to(device, non_blocking=True)
                xC = xC.to(device, non_blocking=True)

                logits = model(xA, xB)
                bs = xA.size(0)
            else:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                bs = x.size(0)

           

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # ----------------------------
            # Convert logits -> mask
            # ----------------------------
            # Binary segmentation (1-channel)
            prob = torch.sigmoid(logits)
            pred = (prob > thresh).float()  # [B,1,H,W] or [B,H,W]

            if pred.dim() == 3:
                pred = pred.unsqueeze(1)  # -> [B,1,H,W]

            pred = pred.detach().cpu().numpy()

            # ----------------------------
            # Save per-sample PNG + TIFF
            # ----------------------------
            # filename might be list of strings or tuple
            if isinstance(filename, (list, tuple)):
                names = filename
            else:
                names = [str(filename)] * bs

            for i in range(bs):
                name = os.path.splitext(os.path.basename(names[i]))[0]

                mask01 = pred[i, 0]  # 0/1 float
                mask_u8 = (mask01 * 255).astype(np.uint8)

                png_path = os.path.join(out_png, f"{name}.png")
                tif_path = os.path.join(out_tiff, f"{name}.tif")

                plt.imsave(png_path, mask_u8, cmap="gray")
                #tiff.imwrite(tif_path, mask_u8)
                tiff.imwrite(
                    tif_path,
                    mask_u8,
                    compression="lzw",
                    resolution=(96, 96),              # (X, Y) pixels per unit
                    resolutionunit="INCH",            # so it's DPI
                )

    print(f"✅ Saved test predictions to:\n- {out_png}\n- {out_tiff}")