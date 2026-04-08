"""
evaluate.py
-----------
Evaluation pipeline for Brain MRI segmentation.

Implements:
    1. Otsu thresholding baseline       (classical CV)
    2. Standard U-Net evaluation        (deep learning baseline)
    3. U-Net + PSO evaluation           (proposed method)

Metrics: Dice, IoU, Pixel Accuracy
Outputs: comparison table + qualitative grid plot

Usage:
    python evaluate.py \
        --data_dir    processed_dataset \
        --model_path  best_pso_model.pth \
        --threshold   0.42              # from ThresholdPSO
"""

import os
import argparse
import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import BrainMRIDataset, get_loaders
from model   import build_model
from utils   import compute_all_metrics, dice_coefficient

torch.backends.cudnn.benchmark = True


# ── Otsu Thresholding ─────────────────────────────────────────────────────────

def otsu_threshold(img_np: np.ndarray) -> float:
    img_uint8 = (img_np * 255).astype(np.uint8)
    counts, _ = np.histogram(img_uint8, bins=256, range=(0, 255))
    total     = img_uint8.size
    sum_total = np.dot(np.arange(256), counts)

    sum_bg, weight_bg = 0.0, 0.0
    best_var, best_thresh = 0.0, 0

    for t in range(256):
        weight_bg += counts[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg   += t * counts[t]
        mean_bg   = sum_bg / weight_bg
        mean_fg   = (sum_total - sum_bg) / weight_fg
        inter_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if inter_var > best_var:
            best_var    = inter_var
            best_thresh = t

    return best_thresh / 255.0


def evaluate_otsu(dataset: BrainMRIDataset) -> dict:
    all_dice, all_iou, all_acc = [], [], []

    for idx in range(len(dataset)):
        img_tensor, mask_tensor = dataset[idx]
        img_np  = img_tensor.squeeze().numpy()

        thresh  = otsu_threshold(img_np)
        pred_np = (img_np > thresh).astype(np.float32)

        pred_t = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0)
        mask_t = mask_tensor.unsqueeze(0)

        m = compute_all_metrics(pred_t, mask_t, threshold=0.5)
        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
        all_acc.append(m["accuracy"])

    return {
        "dice":     float(np.mean(all_dice)),
        "iou":      float(np.mean(all_iou)),
        "accuracy": float(np.mean(all_acc)),
    }


# ── Deep Learning Evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    val_loader,
    device:    torch.device,
    threshold: float = 0.5,
    dropout:   float = 0.0,
) -> dict:
    model = build_model(dropout=dropout, device=str(device))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_dice, all_iou, all_acc = [], [], []
    imgs_list, preds_list, masks_list = [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(images)

            # Step 1: sigmoid → probabilities
            probs = torch.sigmoid(logits).float()

            # Step 2: apply threshold → binary mask
            # ✅ metrics receive binary predictions, not raw probabilities
            preds_bin = (probs > threshold).float()

            # Step 3: compute metrics on binary predictions
            m = compute_all_metrics(preds_bin, masks, threshold=0.5)
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_acc.append(m["accuracy"])

            # Save first batch for visualization (use probs so viz can re-threshold)
            if len(imgs_list) == 0:
                imgs_list  = images.cpu()
                preds_list = probs.cpu()   # keep probs for viz — it applies threshold itself
                masks_list = masks.cpu()

    metrics = {
        "dice":     float(np.mean(all_dice)),
        "iou":      float(np.mean(all_iou)),
        "accuracy": float(np.mean(all_acc)),
    }
    return metrics, model, (imgs_list, preds_list, masks_list)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_predictions(
    imgs:      torch.Tensor,
    preds:     torch.Tensor,
    masks:     torch.Tensor,
    threshold: float = 0.5,
    n_samples: int   = 5,
    save_path: str   = "predictions.png",
    title:     str   = "U-Net Segmentation Results",
):
    n   = min(n_samples, imgs.shape[0])
    fig = plt.figure(figsize=(16, n * 3.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(n, 4, hspace=0.4, wspace=0.15)

    col_labels = ["Input MRI", "Ground Truth", "Predicted Mask", "Overlay"]
    cmaps      = ["gray",       "gray",          "gray",           None]

    for row in range(n):
        img      = imgs[row, 0].numpy()
        mask     = masks[row, 0].numpy()
        pred     = preds[row, 0].numpy()
        pred_bin = (pred > threshold).astype(np.float32)

        overlay = np.stack([img, img, img], axis=-1)
        overlay[..., 1] = np.where(pred_bin * mask > 0,           0.8, overlay[..., 1])
        overlay[..., 0] = np.where(pred_bin * (1 - mask) > 0,     0.8, overlay[..., 0])
        overlay[..., 2] = np.where((1 - pred_bin) * mask > 0,     0.8, overlay[..., 2])
        overlay = np.clip(overlay, 0, 1)

        data_list = [img, mask, pred_bin, overlay]

        for col, (data, label, cmap) in enumerate(zip(data_list, col_labels, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1) if cmap else ax.imshow(data)
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            ax.axis("off")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="lime", label="True Positive"),
            Patch(facecolor="red",  label="False Positive"),
            Patch(facecolor="blue", label="False Negative"),
        ],
        loc="lower center", ncol=3, fontsize=10,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02)
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Viz] Saved → {save_path}")


# ── Comparison Table ──────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    print("\n" + "=" * 65)
    print(f"{'Method':<30} {'Dice':>8} {'IoU':>8} {'Accuracy':>10}")
    print("=" * 65)
    for method, m in results.items():
        print(f"{method:<30} {m['dice']:>8.4f} {m['iou']:>8.4f} {m['accuracy']:>10.4f}")
    print("=" * 65 + "\n")


def plot_comparison_bar(results: dict, save_path: str = "comparison.png"):
    methods   = list(results.keys())
    dice_vals = [results[m]["dice"]     for m in methods]
    iou_vals  = [results[m]["iou"]      for m in methods]
    acc_vals  = [results[m]["accuracy"] for m in methods]

    x      = np.arange(len(methods))
    width  = 0.25
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, dice_vals, width, label="Dice",     color=colors[0], alpha=0.85)
    ax.bar(x,         iou_vals,  width, label="IoU",      color=colors[1], alpha=0.85)
    ax.bar(x + width, acc_vals,  width, label="Accuracy", color=colors[2], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Segmentation Method Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Plot] Saved → {save_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain MRI Segmentation Evaluation")
    parser.add_argument("--data_dir",       default="processed_dataset")
    parser.add_argument("--standard_model", default="best_model.pth")
    parser.add_argument("--pso_model",      default="best_pso_model.pth")
    parser.add_argument("--threshold",      type=float, default=0.5,
                        help="PSO-optimized threshold from ThresholdPSO")
    parser.add_argument("--pso_dropout",    type=float, default=0.0)
    parser.add_argument("--batch_size",     type=int,   default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    full_dataset = BrainMRIDataset(args.data_dir, augment=False)
    _, val_loader = get_loaders(
        args.data_dir, batch_size=args.batch_size, val_split=0.2, num_workers=0
    )

    results = {}

    # ── 1. Otsu Baseline ─────────────────────────────────────────────────────
    print("\n[Eval] Running Otsu baseline ...")
    results["Otsu Thresholding"] = evaluate_otsu(full_dataset)
    print(f"  Dice={results['Otsu Thresholding']['dice']:.4f} | "
          f"IoU={results['Otsu Thresholding']['iou']:.4f} | "
          f"Acc={results['Otsu Thresholding']['accuracy']:.4f}")

    # ── 2. Standard U-Net ────────────────────────────────────────────────────
    if os.path.exists(args.standard_model):
        print(f"\n[Eval] Standard U-Net ({args.standard_model}, threshold=0.50) ...")
        m, _, viz = evaluate_model(
            args.standard_model, val_loader, device, threshold=0.5, dropout=0.0
        )
        results["Standard U-Net"] = m
        print(f"  Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | Acc={m['accuracy']:.4f}")
        visualize_predictions(
            *viz, threshold=0.5, n_samples=5,
            save_path="standard_unet_predictions.png",
            title="Standard U-Net (threshold=0.50)"
        )
    else:
        print(f"[Warning] {args.standard_model} not found — skipping")

    # ── 3. U-Net + PSO ───────────────────────────────────────────────────────
    if os.path.exists(args.pso_model):
        print(f"\n[Eval] U-Net + PSO ({args.pso_model}, threshold={args.threshold:.4f}) ...")
        m, _, viz = evaluate_model(
            args.pso_model, val_loader, device,
            threshold=args.threshold, dropout=args.pso_dropout
        )
        results["U-Net + PSO (Proposed)"] = m
        print(f"  Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | Acc={m['accuracy']:.4f}")
        visualize_predictions(
            *viz, threshold=args.threshold, n_samples=5,
            save_path="pso_unet_predictions.png",
            title=f"U-Net + PSO (threshold={args.threshold:.4f})"
        )
    else:
        print(f"[Warning] {args.pso_model} not found — skipping")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_comparison_table(results)
    plot_comparison_bar(results, save_path="comparison_chart.png")


if __name__ == "__main__":
    main()