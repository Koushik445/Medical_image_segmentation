import os
import cv2
import numpy as np

img_dir = r"D:\processed_dataset\images"
mask_dir = r"D:\processed_dataset\masks"

images = sorted(os.listdir(img_dir))
masks = sorted(os.listdir(mask_dir))

print("\n🔍 DATASET VERIFICATION STARTED\n")

# 1. Count check
print(f"Total Images: {len(images)}")
print(f"Total Masks : {len(masks)}")

assert len(images) == len(masks), "❌ Mismatch in image-mask count"

issues = 0

for i in range(len(images)):

    img_path = os.path.join(img_dir, images[i])
    mask_path = os.path.join(mask_dir, masks[i])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"❌ Corrupt file at index {i}")
        issues += 1
        continue

    # 2. Shape check
    if img.shape != mask.shape:
        print(f"❌ Shape mismatch at {i}")
        issues += 1

    # 3. Binary mask check
    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, [0, 255])):
        print(f"❌ Non-binary mask at {i}: {unique_vals}")
        issues += 1

    # 4. Empty mask check (should not happen now)
    if np.sum(mask) == 0:
        print(f"⚠️ Empty mask at {i}")
        issues += 1

print("\n📊 VERIFICATION COMPLETE")
print(f"Issues found: {issues}")