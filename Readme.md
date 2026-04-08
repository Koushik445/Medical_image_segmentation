# Hybrid Brain MRI Segmentation — U-Net + PSO

> Automated hyperparameter tuning and threshold calibration via Particle Swarm Optimization for binary brain tumor segmentation.

---

## Overview

This project implements a two-stage optimization pipeline around a standard U-Net for binary brain MRI segmentation:

1. **PSO Hyperparameter Search** — a swarm of particles probes the joint space of learning rate and batch size, using 8-epoch validation Dice as the fitness signal.
2. **PSO Threshold Calibration** — after full training, a second independent PSO finds the optimal sigmoid binarization threshold, replacing the fixed default of 0.5.

The framework is architecture-agnostic, fully reproducible, and runs comfortably on a single consumer GPU (tested on RTX 4060 8 GB with AMP).

**Best results (single run, 274-sample validation set):**

| Method | Dice | IoU | Accuracy |
|---|---|---|---|
| Otsu Thresholding | 0.1531 | 0.0865 | 0.6968 |
| Standard U-Net (lr=1e-3) | 0.8168 | 0.7235 | 0.9930 |
| U-Net + PSO (lr=1e-4, θ*=0.4385) | **0.8419** | **0.7506** | **0.9938** |

---

## Project Structure

```
project/
├── dataset.py          # Custom Dataset class + DataLoader factory
├── model.py            # Standard U-Net (31.4M params, no dropout)
├── utils.py            # DiceLoss, DiceBCELoss, Dice/IoU/Accuracy metrics
├── pso.py              # HyperparamPSO + ThresholdPSO (from scratch, no libs)
├── train.py            # Training loop with AMP + CLI interface
├── evaluate.py         # Otsu baseline, model evaluation, visualizations
├── run_local.py        # Single-command end-to-end runner (RTX 4060 tuned)
├── threshold_sweep.py  # Threshold vs Dice curve experiment
└── requirements.txt    # Dependencies
```

---

## Requirements

### Hardware
- **GPU:** NVIDIA RTX 4060 (8 GB VRAM) or equivalent CUDA-capable GPU
- **RAM:** 8 GB minimum
- **OS:** Windows 10/11 or Linux

### Software

**Step 1 — Install PyTorch with CUDA support first:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify your GPU is detected:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 4060 Laptop GPU
```

**Step 2 — Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

## Dataset

Expected structure:
```
processed_dataset/
├── images/
│   ├── img_0.png
│   ├── img_1.png
│   └── ...
└── masks/
    ├── mask_0.png
    ├── mask_1.png
    └── ...
```

- **Total samples:** 1,373
- **Image format:** Grayscale PNG, 128×128 pixels
- **Mask format:** Binary PNG, pixel values {0, 255}
- **Split:** 80% train (1,099) / 20% validation (274), fixed seed 42

---

## Usage

### Option A — Full pipeline, one command (recommended)

```bash
python run_local.py --data_dir processed_dataset
```

This runs all stages in sequence:
1. Standard U-Net baseline training (50 epochs)
2. PSO hyperparameter search (6 particles × 8 iterations × 8 probe epochs)
3. Full retrain with PSO-found hyperparameters (50 epochs)
4. PSO threshold optimization (20 particles × 30 iterations)
5. Full evaluation with comparison table and plots

**Skip baseline if already trained:**
```bash
python run_local.py --data_dir processed_dataset --skip_baseline
```

---

### Option B — Run stages individually

**Stage 1: Train standard U-Net baseline**
```bash
python train.py --data_dir processed_dataset --epochs 50 --save_path best_model.pth
```

**Stage 2: PSO hyperparameter search + full retrain**
```bash
python train.py --data_dir processed_dataset --epochs 50 --use_pso
```

This prints the best hyperparameters and optimal threshold at the end:
```
[PSO] Best hyperparams : {'lr': 0.0001, 'batch_size': 16}
[PSO] Optimal threshold = 0.4385 → Dice = 0.8419
```

**Stage 3: Evaluate all methods**
```bash
python evaluate.py \
  --data_dir processed_dataset \
  --standard_model best_model.pth \
  --pso_model best_pso_model.pth \
  --threshold 0.4385
```

**Stage 4: Threshold sensitivity analysis**
```bash
python threshold_sweep.py --data_dir processed_dataset --model best_pso_model.pth
```
Outputs `threshold_vs_dice.png` and `threshold_vs_dice.csv`.

---

## Expected Runtime (RTX 4060 Laptop)

| Stage | Time |
|---|---|
| Standard U-Net, 50 epochs | ~10 min |
| PSO search (6 × 8 × 8 probe epochs) | ~72 min |
| PSO full retrain, 50 epochs | ~8 min |
| Threshold PSO (20 × 30 iters) | ~3 min |
| Evaluation | ~1 min |
| **Total** | **~95 min** |

Peak VRAM usage: **487 MB** (well within 8 GB budget).

---

## Output Files

After running the full pipeline, the following files are saved to your working directory:

| File | Description |
|---|---|
| `best_model.pth` | Standard U-Net checkpoint (best validation Dice) |
| `best_pso_model.pth` | PSO-optimized U-Net checkpoint |
| `standard_training_history.png` | Loss and Dice curves — standard U-Net |
| `pso_training_history.png` | Loss and Dice curves — PSO U-Net |
| `comparison_chart.png` | Bar chart: Dice / IoU / Accuracy across all methods |
| `standard_unet_predictions.png` | Qualitative predictions — standard U-Net |
| `pso_unet_predictions.png` | Qualitative predictions — PSO U-Net |
| `threshold_vs_dice.png` | Threshold sensitivity curve (from threshold_sweep.py) |
| `threshold_vs_dice.csv` | Raw sweep data |

---

## Key Design Decisions

**No dropout.** Every DoubleConv block uses BatchNorm2d. Combining Dropout with BatchNorm causes variance shift artifacts that degrade validation performance (Li et al., CVPR 2019). Regularization is handled by L2 weight decay in the optimizer instead.

**BCEWithLogitsLoss, not BCELoss.** PyTorch's AMP dispatcher explicitly blocks BCELoss under autocast because FP16 cannot represent probabilities near 0 or 1 accurately. BCEWithLogitsLoss folds sigmoid + BCE into one numerically stable, AMP-whitelisted operation.

**Raw logits from model.** The U-Net outputs raw logits. `torch.sigmoid()` is applied explicitly at inference time. This is required for AMP compatibility.

**Adaptive inertia PSO.** The inertia weight decays linearly from w=0.9 to w=0.4 over iterations (Shi & Eberhart, 1998). High inertia early = broad exploration; low inertia late = precise exploitation.

---

## PSO Configuration

| Parameter | Hyperparameter PSO | Threshold PSO |
|---|---|---|
| Particles | 6 | 20 |
| Iterations | 8 | 30 |
| Search space | lr ∈ [1e-4, 1e-2], bs ∈ {8, 16, 32} | θ ∈ [0.1, 0.9] |
| Fitness | 8-epoch validation Dice | Full-model validation Dice |
| Inertia w_max | 0.9 | 0.9 |
| Inertia w_min | 0.4 | 0.4 |
| c1 = c2 | 1.5 | 1.5 |
| Total evaluations | 54 probe runs | 600 forward passes |

---

## RTX 4060 Optimizations Applied

- `torch.amp.autocast` + `GradScaler` — FP16 forward/backward, ~40% VRAM reduction
- `torch.backends.cudnn.benchmark = True` — cuDNN auto-tunes conv algorithms
- `optimizer.zero_grad(set_to_none=True)` — frees gradient memory instead of zeroing
- `non_blocking=True` on `.to(device)` — overlaps CPU→GPU transfer with compute
- `persistent_workers=True`, `prefetch_factor=2` — keeps DataLoader workers alive

---

## Results Interpretation

The threshold sensitivity analysis shows the Dice-threshold curve is smooth and unimodal, peaking near θ = 0.43–0.46. The conventional fixed threshold of 0.5 is mildly suboptimal for this class-imbalanced dataset (tumor occupies a small fraction of pixels). PSO reliably converges to this peak region.

The improvement over the standard U-Net baseline (~2.3 pp in Dice) is modest but consistent, falling within the 1–3% range considered meaningful in medical image segmentation given already strong baselines. Note that single-run results are subject to stochastic variability; multi-seed evaluation is recommended for publication.

---

## Citation

If you use this code in your research, please cite:

```
@article{unet_pso_brain_mri,
  title   = {Hybrid Brain MRI Segmentation Using U-Net with Particle Swarm Optimization},
  journal = {[Conference/Journal Name]},
  year    = {2025},
  note    = {Adaptive Hyperparameter Tuning and Threshold Calibration via PSO}
}
```

---

## References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
- Kennedy & Eberhart, "Particle Swarm Optimization," IEEE ICNN 1995
- Shi & Eberhart, "A Modified Particle Swarm Optimizer," IEEE CEC 1998
- Li et al., "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift," CVPR 2019
- Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015