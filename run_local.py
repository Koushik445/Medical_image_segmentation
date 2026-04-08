"""
run_local.py
------------
End-to-end runner for RTX 4060 (8 GB VRAM).
Runs all three stages in sequence:
    1. Standard U-Net (baseline)
    2. PSO hyperparameter search + full retrain (proposed method)
    3. PSO threshold optimization
    4. Full evaluation + comparison

Usage:
    python run_local.py --data_dir processed_dataset
    python run_local.py --data_dir processed_dataset --epochs 80
    python run_local.py --data_dir processed_dataset --skip_baseline
"""

import argparse
import torch
from train    import train, plot_history, DEFAULT_CFG
from dataset  import get_loaders
from pso      import HyperparamPSO, ThresholdPSO
from evaluate import (evaluate_otsu, evaluate_model, print_comparison_table,
                      plot_comparison_bar, visualize_predictions)
from dataset  import BrainMRIDataset
from utils    import dice_coefficient
from torch.amp import autocast


def check_gpu():
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[ERROR] PyTorch cannot see your GPU.\n"
            "Make sure you installed the CUDA build:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "Then verify with:\n"
            "  python -c \"import torch; print(torch.cuda.get_device_name(0))\""
        )
    device    = torch.device("cuda")
    gpu       = torch.cuda.get_device_name(0)
    vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n{'='*60}")
    print(f"  GPU   : {gpu}")
    print(f"  VRAM  : {vram_gb:.1f} GB")
    print(f"  AMP   : autocast + GradScaler enabled")
    print(f"  cuDNN : benchmark=True")
    print(f"{'='*60}\n")
    return device


def main():
    parser = argparse.ArgumentParser(description="RTX 4060 Brain MRI Segmentation Pipeline")
    parser.add_argument("--data_dir",      default="processed_dataset")
    parser.add_argument("--epochs",        type=int, default=50,
                        help="Epochs for final training runs (default: 50)")
    parser.add_argument("--pso_epochs",    type=int, default=15,
                        help="Probe epochs per PSO particle (default: 15)")
    parser.add_argument("--pso_particles", type=int, default=10)
    parser.add_argument("--pso_iters",     type=int, default=15)
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip standard U-Net training (if already done)")
    args = parser.parse_args()

    device = check_gpu()

    results       = {}
    best_thresh   = 0.5
    best_dropout  = 0.0
    best_bs       = 16

    # ── STAGE 1: Standard U-Net Baseline ──────────────────────────────────────
    if not args.skip_baseline:
        print("=" * 60)
        print("  STAGE 1: Standard U-Net (Baseline)")
        print("=" * 60)
        _, history_std, dice_std = train(
            data_dir=args.data_dir, epochs=args.epochs,
            lr=1e-3, batch_size=16, dropout=0.0,
            val_split=0.2, save_path="best_model.pth", device=device,
        )
        print(f"\n  ✓ Standard U-Net best Dice: {dice_std:.4f}")
        plot_history(history_std, title="Standard U-Net Training",
                     save="standard_training_history.png")
    else:
        print("[skip] Standard U-Net training skipped.")

    # ── STAGE 2: PSO Hyperparameter Search ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2: PSO Hyperparameter Search")
    print(f"           {args.pso_particles} particles × {args.pso_iters} iterations")
    print(f"           {args.pso_epochs} probe epochs per particle")
    print("=" * 60)

    def fitness_fn(lr, batch_size, dropout):
        print(f"    [probe] lr={lr:.5f} bs={batch_size} dropout={dropout:.3f}", end=" ")
        _, _, dice = train(
            data_dir=args.data_dir, epochs=args.pso_epochs,
            lr=lr, batch_size=batch_size, dropout=dropout,
            val_split=0.2, save_path="pso_probe.pth",
            device=device, verbose=False,
        )
        print(f"→ dice={dice:.4f}")
        return dice

    pso_hyper = HyperparamPSO(
        fitness_fn=fitness_fn,
        n_particles=args.pso_particles,
        n_iters=args.pso_iters,
        batch_choices=[8, 16, 32],   # RTX 4060 can handle 32 with AMP
    )
    best_params, probe_dice, _ = pso_hyper.optimize()
    best_bs      = best_params["batch_size"]
    best_dropout = best_params["dropout"]
    print(f"\n  ✓ Best params: {best_params} | probe dice: {probe_dice:.4f}")

    # ── STAGE 3: Full Retrain with PSO Hyperparams ────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 3: Full Retrain with PSO Hyperparams")
    print("=" * 60)
    pso_model, history_pso, dice_pso = train(
        data_dir=args.data_dir, epochs=args.epochs,
        lr=best_params["lr"], batch_size=best_bs, dropout=best_dropout,
        val_split=0.2, save_path="best_pso_model.pth", device=device,
    )
    print(f"\n  ✓ PSO U-Net best Dice: {dice_pso:.4f}")
    plot_history(history_pso, title="U-Net + PSO Training",
                 save="pso_training_history.png")

    # ── STAGE 4: PSO Threshold Optimization ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4: PSO Threshold Optimization")
    print("=" * 60)
    _, val_loader = get_loaders(args.data_dir, batch_size=best_bs, val_split=0.2)

    pso_model.eval()

    def thresh_fitness(threshold):
        total = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                with autocast(device_type="cuda"):
                    logits = pso_model(imgs)
                # dice_coefficient applies sigmoid internally
                total += dice_coefficient(logits.float(), msks.float(), threshold)
        return total / len(val_loader)

    thresh_pso = ThresholdPSO(fitness_fn=thresh_fitness, n_particles=20, n_iters=30)
    best_thresh, thresh_dice = thresh_pso.optimize()
    print(f"\n  ✓ Optimal threshold: {best_thresh:.4f} → Dice: {thresh_dice:.4f}")

    # ── STAGE 5: Full Evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 5: Full Evaluation & Comparison")
    print("=" * 60)

    full_ds = BrainMRIDataset(args.data_dir, augment=False)
    _, val_loader_std = get_loaders(args.data_dir, batch_size=16, val_split=0.2)
    _, val_loader_pso = get_loaders(args.data_dir, batch_size=best_bs, val_split=0.2)

    # Otsu
    print("\n  Running Otsu baseline ...")
    results["Otsu Thresholding"] = evaluate_otsu(full_ds)

    # Standard U-Net
    if not args.skip_baseline:
        print("  Evaluating Standard U-Net ...")
        m_std, _, viz_std = evaluate_model(
            "best_model.pth", val_loader_std, device, threshold=0.5, dropout=0.0
        )
        results["Standard U-Net"] = m_std
        visualize_predictions(*viz_std, threshold=0.5, n_samples=5,
                              save_path="standard_unet_predictions.png",
                              title="Standard U-Net (threshold=0.50)")

    # PSO U-Net
    print("  Evaluating U-Net + PSO ...")
    m_pso, _, viz_pso = evaluate_model(
        "best_pso_model.pth", val_loader_pso, device,
        threshold=best_thresh, dropout=best_dropout
    )
    results["U-Net + PSO (Proposed)"] = m_pso
    visualize_predictions(*viz_pso, threshold=best_thresh, n_samples=5,
                          save_path="pso_unet_predictions.png",
                          title=f"U-Net + PSO (threshold={best_thresh:.4f})")

    # ── Print Summary ──────────────────────────────────────────────────────────
    print_comparison_table(results)
    plot_comparison_bar(results, save_path="comparison_chart.png")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print(f"  Best threshold for paper: {best_thresh:.4f}")
    print(f"  Saved models: best_model.pth, best_pso_model.pth")
    print(f"  Plots: standard_training_history.png, pso_training_history.png,")
    print(f"         comparison_chart.png, *_predictions.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()