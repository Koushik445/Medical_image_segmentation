import os
import argparse
import time
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from dataset import get_loaders
from model   import build_model
from utils   import DiceBCELoss, dice_coefficient

torch.backends.cudnn.benchmark = True

DEFAULT_CFG = {
    "data_dir":   "processed_dataset",
    "epochs":     50,
    "lr":         1e-3,
    "batch_size": 16,
    "dropout":    0.0,
    "val_split":  0.2,
    "save_path":  "best_model.pth",
    "seed":       42,
}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            preds = model(images)
            loss  = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = total_dice = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            with autocast(device_type="cuda"):
                preds = model(images)
                loss  = criterion(preds, masks)
            total_loss += loss.item()
            total_dice += dice_coefficient(preds.float(), masks.float(), threshold)
    return total_loss / len(loader), total_dice / len(loader)


def train(
    data_dir, epochs, lr, batch_size, dropout,
    val_split, save_path, device,
    verbose=True, threshold=0.5,
    num_workers=4,# ← 0 for PSO probes, 4 for final train
):
    torch.manual_seed(DEFAULT_CFG["seed"])

    train_loader, val_loader = get_loaders(
        data_dir, batch_size=batch_size,
        val_split=val_split, num_workers=num_workers,
    )

    model     = build_model(dropout=dropout, device=str(device))
    criterion = DiceBCELoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler    = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history   = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0
    start     = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device, threshold)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)

        if verbose:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"dice={val_dice:.4f} | best={best_dice:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"VRAM={torch.cuda.memory_allocated(device)/1024**2:.0f}MB | "
                  f"{time.time()-start:.0f}s")

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model, history, best_dice


def plot_history(history, title="Training History", save=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss", color="#E63946")
    ax1.plot(history["val_loss"],   label="Val Loss",   color="#457B9D")
    ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("DiceBCE Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["val_dice"], label="Val Dice", color="#2A9D8F")
    ax2.set_title("Validation Dice"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = save or f"{title.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150); plt.show()
    print(f"[Plot] Saved → {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=DEFAULT_CFG["data_dir"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--dropout",    type=float, default=DEFAULT_CFG["dropout"])
    parser.add_argument("--save_path",  default=DEFAULT_CFG["save_path"])
    parser.add_argument("--use_pso",    action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA not available.")

    device = torch.device("cuda")
    print(f"\n[Device] {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"[AMP]    autocast + GradScaler enabled\n")

    if args.use_pso:
        from pso import HyperparamPSO, ThresholdPSO

        PSO_EPOCHS = 8

        def fitness_fn(lr, batch_size):
            print(f"\n  [probe] lr={lr:.5f}, bs={batch_size}", end=" ", flush=True)
            _, _, dice = train(
                data_dir=args.data_dir,
                epochs=PSO_EPOCHS,
                lr=lr,
                batch_size=batch_size,
                dropout=0.0,
                val_split=DEFAULT_CFG["val_split"],
                save_path="pso_probe.pth",
                device=device,
                verbose=True,
                num_workers=0,          # ← THE fix: no subprocesses on Windows
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"→ dice={dice:.4f} | "
                  f"VRAM={torch.cuda.memory_allocated()/1024**2:.0f}MB")
            return dice

        pso_hyper = HyperparamPSO(
            fitness_fn=fitness_fn,
            n_particles=6,
            n_iters=8,
            batch_choices=[8, 16],
        )
        best_params, best_probe_dice, _ = pso_hyper.optimize()
        print(f"\n[PSO] Best hyperparams : {best_params}")
        print(f"[PSO] Best probe Dice  : {best_probe_dice:.4f}")

        print(f"\n[Train] Retraining for {args.epochs} epochs ...")
        model, history, best_dice = train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=best_params["lr"],
            batch_size=best_params["batch_size"],
            dropout=0.0,
            val_split=DEFAULT_CFG["val_split"],
            save_path="best_pso_model.pth",
            device=device,
            num_workers=4,              # ← back to 4 for the single long run
        )
        print(f"\n[PSO Train] Best Dice = {best_dice:.4f}")

        print("\n[PSO-Thresh] Optimizing threshold ...")
        _, val_loader = get_loaders(
            args.data_dir,
            batch_size=best_params["batch_size"],
            val_split=DEFAULT_CFG["val_split"],
            num_workers=0,              # short-lived val-only loader
        )

        def thresh_fitness(threshold):
            model.eval()
            total_dice = 0.0
            with torch.no_grad():
                for imgs, msks in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    msks = msks.to(device, non_blocking=True)
                    with autocast(device_type="cuda"):
                        preds = model(imgs)
                    total_dice += dice_coefficient(preds.float(), msks.float(), threshold)
            return total_dice / len(val_loader)

        from pso import ThresholdPSO
        thresh_pso = ThresholdPSO(fitness_fn=thresh_fitness)
        best_thresh, thresh_dice = thresh_pso.optimize()
        print(f"\n[PSO] Optimal threshold = {best_thresh:.4f} → Dice = {thresh_dice:.4f}")
        print(f"[IMPORTANT] Pass --threshold {best_thresh:.4f} to evaluate.py\n")
        plot_history(history, title="U-Net + PSO Training", save="pso_training_history.png")

    else:
        print(f"[Train] Standard U-Net | epochs={args.epochs} | "
              f"batch={args.batch_size} | lr={args.lr}")
        model, history, best_dice = train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            val_split=DEFAULT_CFG["val_split"],
            save_path=args.save_path,
            device=device,
            num_workers=4,
        )
        print(f"\n[Done] Best Dice = {best_dice:.4f}")
        plot_history(history, title="Standard U-Net Training",
                     save="standard_training_history.png")


if __name__ == "__main__":
    main()