import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torch


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir  = os.path.join(root_dir, "masks")
        self.augment   = augment

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        self.mask_files = sorted(
            [f for f in os.listdir(self.mask_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        assert len(self.image_files) == len(self.mask_files), (
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("L")
        mask  = Image.open(os.path.join(self.mask_dir,  self.mask_files[idx])).convert("L")

        if self.augment and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask  = (TF.to_tensor(mask) > 0.5).float()
        return image, mask


def get_loaders(
    root_dir:    str,
    batch_size:  int   = 16,
    val_split:   float = 0.2,
    num_workers: int   = 4,     # pass 0 for PSO probes on Windows
    seed:        int   = 42,
):
    _tmp    = BrainMRIDataset(root_dir)
    n_total = len(_tmp)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset = BrainMRIDataset(root_dir, augment=True)
    val_dataset   = BrainMRIDataset(root_dir, augment=False)

    train_ds, _ = random_split(train_dataset, [n_train, n_val],
                               generator=torch.Generator().manual_seed(seed))
    _, val_ds   = random_split(val_dataset,   [n_train, n_val],
                               generator=torch.Generator().manual_seed(seed))

    # num_workers=0 → single-threaded, no subprocess spawning
    # Required for PSO on Windows; fine for short probe runs
    # num_workers>0 only when persistent_workers can stay alive (final train)
    persistent = num_workers > 0
    pf         = 2 if persistent else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=pf,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=pf,
    )

    print(f"[Dataset] Total: {n_total} | Train: {n_train} | Val: {n_val} "
          f"| batch={batch_size} | workers={num_workers}")
    return train_loader, val_loader