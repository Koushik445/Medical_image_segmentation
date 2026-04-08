"""
multi_run_eval.py
-----------------
Statistical significance evaluation for Brain MRI U-Net segmentation.
Trains the model multiple times with different seeds and reports mean ± std.

Compares:
    1. Standard U-Net   (fixed lr=1e-3, bs=16)
    2. U-Net + PSO      (best params from PSO search)

Usage:
    # After PSO gives you best params:
    python multi_run_eval.py \
        --data_dir processed_dataset \
        --pso_lr 0.00381 \
        --pso_bs 16

    # Compare only standard U-Net:
    python multi_run_eval.py --data_dir processed_dataset --skip_pso
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import random

from dataset import get_loaders
from model   import build_model
from utils   import DiceBCELoss, dice_coefficient

torch.backends.cudnn.benchmark = True

SEEDS = [42, 52]


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops for this seed — slight speed cost, worth it for
    # reproducibility across runs
    torch.backends.cudnn.deterministic =False
    torch.backends.cudnn.benchmark     =True


def restore_benchmark():
    """Re-enable benchmark mode after a seeded run."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ── Full Training Run ─────────────────────────────────────────────────────────

def full_train(
    data_dir:   str,
    lr:         float,
    batch_size: int,
    epochs:     int,
    device:     torch.device,
    seed:       int,
    label:      str = "",
) -> float:
    """
    One complete training run with a fixed seed.
    Returns best validation Dice achieved.
    """
    set_seed(seed)

    train_loader, val_loader = get_loaders(
        data_dir,
        batch_size  = batch_size,
        val_split   = 0.2,
        num_workers = 4,
        seed        = seed,
    )

    model     = build_model(dropout=0.0, device=str(device))
    criterion = DiceBCELoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler    = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                preds = model(images)
                loss  = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        total_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with autocast(device_type="cuda"):
                    preds = model(images)
                # Binarize before metric — probabilities after sigmoid
                probs     = torch.sigmoid(preds).float()
                preds_bin = (probs > 0.5).float()
                total_dice += dice_coefficient(preds_bin, masks.float(), threshold=0.5)

        val_dice = total_dice / len(val_loader)

        if val_dice > best_dice:
            best_dice = val_dice

        print(f"    [{label} | seed={seed}] "
              f"Epoch [{epoch:3d}/{epochs}] "
              f"train={train_loss:.4f} | dice={val_dice:.4f} | "
              f"best={best_dice:.4f} | "
              f"VRAM={torch.cuda.memory_allocated(device)/1024**2:.0f}MB",
              flush=True)

    restore_benchmark()

    # Clean up to free VRAM for next run
    del model, optimizer, scaler, scheduler
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return best_dice


# ── Multi-Run Evaluation ──────────────────────────────────────────────────────

def multi_run(
    data_dir:   str,
    lr:         float,
    batch_size: int,
    epochs:     int,
    device:     torch.device,
    label:      str,
    seeds:      list = None,
) -> dict:
    """
    Train `len(seeds)` times with different seeds.
    Returns dict with per-seed scores, mean, and std.
    """
    if seeds is None:
        seeds = SEEDS

    print(f"\n{'='*62}")
    print(f"[MultiRun] {label}")
    print(f"  lr={lr:.5f} | batch_size={batch_size} | epochs={epochs}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*62}", flush=True)

    scores = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n  ── Run {i}/{len(seeds)} (seed={seed}) ──────────────────────",
              flush=True)
        dice = full_train(
            data_dir   = data_dir,
            lr         = lr,
            batch_size = batch_size,
            epochs     = epochs,
            device     = device,
            seed       = seed,
            label      = label,
        )
        scores.append(dice)
        print(f"  Run {i} final Dice = {dice:.4f}", flush=True)

    mean = float(np.mean(scores))
    std  = float(np.std(scores))

    print(f"\n[MultiRun] {label} COMPLETE")
    print(f"  Per-seed scores : {[f'{s:.4f}' for s in scores]}")
    print(f"  Dice = {mean:.4f} ± {std:.4f}", flush=True)

    return {
        "label":      label,
        "lr":         lr,
        "batch_size": batch_size,
        "scores":     scores,
        "mean":       mean,
        "std":        std,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-run statistical evaluation — Brain MRI U-Net"
    )
    parser.add_argument("--data_dir",  default="processed_dataset")
    parser.add_argument("--epochs",    type=int,   default=40,
                        help="Training epochs per run (default 40)")
    parser.add_argument("--pso_lr",    type=float, default=None,
                        help="Best lr found by PSO (from train.py --use_pso output)")
    parser.add_argument("--pso_bs",    type=int,   default=None,
                        help="Best batch_size found by PSO")
    parser.add_argument("--skip_pso",  action="store_true",
                        help="Skip PSO model evaluation (if PSO not yet run)")
    parser.add_argument("--seeds",     type=int, nargs="+", default=SEEDS,
                        help="Seeds to use (default: 42 52 62)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA not available.")

    device = torch.device("cuda")
    print(f"\n[Device] {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"[MultiRun] Seeds: {args.seeds} | Epochs per run: {args.epochs}")

    all_results = []

    # ── Standard U-Net ────────────────────────────────────────────────────────
    std_result = multi_run(
        data_dir   = args.data_dir,
        lr         = 1e-3,
        batch_size = 16,
        epochs     = args.epochs,
        device     = device,
        label      = "Standard U-Net",
        seeds      = args.seeds,
    )
    all_results.append(std_result)

    # ── U-Net + PSO ───────────────────────────────────────────────────────────
    if not args.skip_pso:
        if args.pso_lr is None or args.pso_bs is None:
            print("\n[MultiRun] PSO params not provided via --pso_lr / --pso_bs")
            print("[MultiRun] Run train.py --use_pso first, then pass the best params here.")
            print("[MultiRun] Skipping PSO evaluation.\n")
        else:
            pso_result = multi_run(
                data_dir   = args.data_dir,
                lr         = args.pso_lr,
                batch_size = args.pso_bs,
                epochs     = args.epochs,
                device     = device,
                label      = "U-Net + PSO",
                seeds      = args.seeds,
            )
            all_results.append(pso_result)

    # ── Final Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'Method':<25} {'Dice Mean':>10} {'Dice Std':>10}  Scores")
    print("=" * 62)
    for r in all_results:
        scores_str = "  ".join([f"{s:.4f}" for s in r["scores"]])
        print(f"  {r['label']:<25} {r['mean']:>10.4f} {r['std']:>10.4f}  [{scores_str}]")
    print("=" * 62)

    # ── Publication-ready one-liners ──────────────────────────────────────────
    print("\n[MultiRun] Publication-ready results:")
    for r in all_results:
        print(f"  {r['label']}: Dice = {r['mean']:.4f} ± {r['std']:.4f}")

    # ── Save to text file ─────────────────────────────────────────────────────
    with open("multi_run_results.txt", "w") as f:
        f.write("Multi-Run Evaluation Results\n")
        f.write("=" * 62 + "\n")
        for r in all_results:
            f.write(f"{r['label']}\n")
            f.write(f"  lr={r['lr']:.5f} | batch_size={r['batch_size']}\n")
            f.write(f"  Seeds: {args.seeds}\n")
            f.write(f"  Per-seed Dice: {[round(s,4) for s in r['scores']]}\n")
            f.write(f"  Dice = {r['mean']:.4f} ± {r['std']:.4f}\n\n")
    print("\n[MultiRun] Results saved → multi_run_results.txt")


if __name__ == "__main__":
    main()