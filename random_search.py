"""
random_search.py
----------------
Random Search hyperparameter optimization for Brain MRI U-Net segmentation.
Fair comparison baseline against PSO (same budget: 54 trials × 15 proxy epochs).

Search space:
    lr         : uniform in [1e-4, 1e-2]
    batch_size : choice from [8, 16]   ← 32 removed (OOMs on 8GB RTX 4060)

Usage:
    python random_search.py --data_dir processed_dataset
    python random_search.py --data_dir processed_dataset --trials 54 --epochs 15
"""

import os
import csv
import random
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

from dataset import get_loaders
from model   import build_model
from utils   import DiceBCELoss, dice_coefficient

torch.backends.cudnn.benchmark = True


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Single Proxy Training Run ─────────────────────────────────────────────────

def probe_train(
    data_dir:   str,
    lr:         float,
    batch_size: int,
    epochs:     int,
    device:     torch.device,
    trial_idx:  int,
) -> float:
    """
    Train for a fixed number of proxy epochs and return validation Dice.
    Mirrors the exact training setup used in PSO fitness_fn.
    """
    train_loader, val_loader = get_loaders(
        data_dir,
        batch_size  = batch_size,
        val_split   = 0.2,
        num_workers = 0,        # Windows-safe: no subprocess spawning
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
                total_dice += dice_coefficient(
                    preds.float(), masks.float(), threshold=0.5
                )
        val_dice = total_dice / len(val_loader)

        print(f"    Epoch [{epoch:2d}/{epochs}] dice={val_dice:.4f}", flush=True)

        if val_dice > best_dice:
            best_dice = val_dice

    # Free VRAM before next trial
    del model, optimizer, scaler, scheduler
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return best_dice


# ── Random Search ─────────────────────────────────────────────────────────────

def random_search(
    data_dir:     str,
    n_trials:     int,
    proxy_epochs: int,
    device:       torch.device,
    lr_low:       float = 1e-4,
    lr_high:      float = 1e-2,
    batch_choices: list = None,
    seed:         int   = 42,
) -> tuple:
    """
    Randomly sample hyperparameters, evaluate each with proxy training,
    and return the best configuration.

    Returns:
        best_params : dict {"lr": ..., "batch_size": ...}
        best_dice   : float
        all_results : list of dicts (all trials)
    """
    if batch_choices is None:
        batch_choices = [8, 16]

    set_seed(seed)

    all_results = []
    best_dice   = 0.0
    best_params = {}

    print(f"\n[RandomSearch] {n_trials} trials × {proxy_epochs} proxy epochs")
    print(f"[RandomSearch] lr ∈ [{lr_low:.0e}, {lr_high:.0e}] | "
          f"batch ∈ {batch_choices}")
    print("=" * 62, flush=True)

    for trial in range(1, n_trials + 1):
        # Sample hyperparameters uniformly
        lr         = float(np.random.uniform(lr_low, lr_high))
        batch_size = int(np.random.choice(batch_choices))

        print(f"\n[Trial {trial:03d}/{n_trials}] "
              f"lr={lr:.5f}  bs={batch_size}", flush=True)

        t_start = time.time()
        dice    = probe_train(
            data_dir   = data_dir,
            lr         = lr,
            batch_size = batch_size,
            epochs     = proxy_epochs,
            device     = device,
            trial_idx  = trial,
        )
        elapsed = time.time() - t_start

        result = {
            "trial":      trial,
            "lr":         lr,
            "batch_size": batch_size,
            "dice":       dice,
            "elapsed_s":  round(elapsed, 1),
        }
        all_results.append(result)

        print(f"  → Trial {trial:03d} DONE | dice={dice:.4f} | {elapsed:.0f}s",
              flush=True)

        if dice > best_dice:
            best_dice   = dice
            best_params = {"lr": lr, "batch_size": batch_size}
            print(f"  ★ New best! dice={best_dice:.4f}", flush=True)

    return best_params, best_dice, all_results


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random Search — Brain MRI U-Net")
    parser.add_argument("--data_dir",   default="processed_dataset")
    parser.add_argument("--trials",     type=int,   default=54,
                        help="Number of random trials (default 54 = same budget as PSO)")
    parser.add_argument("--epochs",     type=int,   default=15,
                        help="Proxy training epochs per trial")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save_csv",   default="random_search_results.csv",
                        help="Path to save per-trial results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA not available.")

    device = torch.device("cuda")
    print(f"\n[Device] {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    best_params, best_dice, all_results = random_search(
        data_dir     = args.data_dir,
        n_trials     = args.trials,
        proxy_epochs = args.epochs,
        device       = device,
        seed         = args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("[RandomSearch] COMPLETE")
    print(f"  Best lr         : {best_params['lr']:.5f}")
    print(f"  Best batch_size : {best_params['batch_size']}")
    print(f"  Best Dice       : {best_dice:.4f}")
    print("=" * 62)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(args.save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial","lr","batch_size","dice","elapsed_s"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[RandomSearch] Results saved → {args.save_csv}")

    # ── Top 5 ─────────────────────────────────────────────────────────────────
    top5 = sorted(all_results, key=lambda x: x["dice"], reverse=True)[:5]
    print("\n[RandomSearch] Top 5 trials:")
    print(f"  {'Trial':>6} {'LR':>10} {'BS':>4} {'Dice':>8}")
    print("  " + "-" * 34)
    for r in top5:
        print(f"  {r['trial']:>6} {r['lr']:>10.5f} {r['batch_size']:>4} {r['dice']:>8.4f}")


if __name__ == "__main__":
    main()