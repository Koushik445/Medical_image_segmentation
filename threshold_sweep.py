"""
threshold_sweep.py
------------------
Evaluate Dice score across thresholds 0.1 → 0.9 (step 0.05)
on the PSO-trained U-Net, then plot the curve with two markers:
  - theta = 0.50  (conventional fixed threshold)
  - theta = 0.4385 (PSO-optimised threshold)

Usage:
    python threshold_sweep.py --data_dir processed_dataset --model best_pso_model.pth

Output:
    threshold_vs_dice.png  — publication-ready figure
    threshold_vs_dice.csv  — raw numbers for the paper table
"""

import argparse
import csv
import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import get_loaders
from model   import build_model
from utils   import dice_coefficient

torch.backends.cudnn.benchmark = True

# ── Config ────────────────────────────────────────────────────────────────────

THRESHOLDS   = np.round(np.arange(0.10, 0.91, 0.05), 4).tolist()
THRESHOLD_FIXED = 0.50
THRESHOLD_PSO   = 0.4385

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Threshold vs Dice sweep")
    parser.add_argument("--data_dir",   default="processed_dataset")
    parser.add_argument("--model",      default="best_pso_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = build_model(device=str(device))
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model]  Loaded {args.model}")

    # Validation loader only
    _, val_loader = get_loaders(args.data_dir, batch_size=args.batch_size, val_split=0.2)
    print(f"[Data]   Val batches: {len(val_loader)}\n")

    # ── Sweep ────────────────────────────────────────────────────────────────
    results = []   # list of (threshold, dice)

    for theta in THRESHOLDS:
        total_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device,  non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast(device_type="cuda"):
                    logits = model(imgs)
                total_dice += dice_coefficient(logits.float(), masks.float(), threshold=theta)
        avg_dice = total_dice / len(val_loader)
        results.append((theta, round(avg_dice, 6)))
        marker = ""
        if abs(theta - THRESHOLD_FIXED) < 0.001:
            marker = "  ← fixed 0.50"
        if abs(theta - THRESHOLD_PSO) < 0.03:
            marker = "  ← PSO optimum"
        print(f"  theta={theta:.4f}  Dice={avg_dice:.4f}{marker}")

    thresholds = [r[0] for r in results]
    dices      = [r[1] for r in results]
    best_theta, best_dice = max(results, key=lambda x: x[1])
    print(f"\n[Result] Best: theta={best_theta:.4f} → Dice={best_dice:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open("threshold_vs_dice.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "dice"])
        writer.writerows(results)
    print("[CSV]    Saved → threshold_vs_dice.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    # Main curve
    ax.plot(thresholds, dices, color="#2A9D8F", linewidth=2.2,
            marker="o", markersize=5, label="Validation Dice", zorder=3)

    # Shaded region under curve
    ax.fill_between(thresholds, dices, min(dices) - 0.005,
                    alpha=0.08, color="#2A9D8F")

    # ── Marker: fixed 0.50 ──────────────────────────────────────────────────
    dice_at_fixed = dict(results).get(THRESHOLD_FIXED)
    if dice_at_fixed is None:
        # interpolate if 0.50 not in list
        dice_at_fixed = float(np.interp(THRESHOLD_FIXED, thresholds, dices))

    ax.axvline(x=THRESHOLD_FIXED, color="#E63946", linewidth=1.5,
               linestyle="--", zorder=2, label=f"Fixed θ = 0.50  (Dice = {dice_at_fixed:.4f})")
    ax.scatter([THRESHOLD_FIXED], [dice_at_fixed],
               color="#E63946", s=80, zorder=5, edgecolors="white", linewidths=1)
    ax.annotate(f"θ = 0.50\nDice = {dice_at_fixed:.4f}",
                xy=(THRESHOLD_FIXED, dice_at_fixed),
                xytext=(THRESHOLD_FIXED + 0.07, dice_at_fixed - 0.006),
                fontsize=9, color="#E63946",
                arrowprops=dict(arrowstyle="-", color="#E63946", lw=1))

    # ── Marker: PSO optimum ──────────────────────────────────────────────────
    dice_at_pso = float(np.interp(THRESHOLD_PSO, thresholds, dices))

    ax.axvline(x=THRESHOLD_PSO, color="#457B9D", linewidth=1.5,
               linestyle="--", zorder=2, label=f"PSO θ* = {THRESHOLD_PSO}  (Dice = {dice_at_pso:.4f})")
    ax.scatter([THRESHOLD_PSO], [dice_at_pso],
               color="#457B9D", s=100, zorder=5, marker="*",
               edgecolors="white", linewidths=0.8)
    ax.annotate(f"PSO θ* = {THRESHOLD_PSO}\nDice = {dice_at_pso:.4f}",
                xy=(THRESHOLD_PSO, dice_at_pso),
                xytext=(THRESHOLD_PSO - 0.19, dice_at_pso - 0.007),
                fontsize=9, color="#457B9D",
                arrowprops=dict(arrowstyle="-", color="#457B9D", lw=1))

    # ── Axis labels & formatting ──────────────────────────────────────────────
    ax.set_xlabel("Binarisation Threshold (θ)", fontsize=12)
    ax.set_ylabel("Validation Dice Coefficient", fontsize=12)
    ax.set_title("Effect of Binarisation Threshold on Segmentation Dice\n(U-Net + PSO Model, 274-sample Validation Set)",
                 fontsize=12, fontweight="bold", pad=12)

    ax.set_xlim(0.08, 0.92)
    ax.set_xticks(thresholds)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    y_min = min(dices) - 0.01
    y_max = max(dices) + 0.01
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("threshold_vs_dice.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("[Plot]   Saved → threshold_vs_dice.png")


if __name__ == "__main__":
    main()