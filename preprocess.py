import os
import cv2
import numpy as np

input_dir = r"D:\Brain_Segmentation\brain_segmentation_data\lgg-mri-segmentation\kaggle_3m"
output_dir = r"D:\processed_dataset"

img_dir = os.path.join(output_dir, "images")
mask_dir = os.path.join(output_dir, "masks")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

count = 0
skipped = 0

print("\n🚀 PREPROCESSING STARTED...\n")

for patient in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient)

    if not os.path.isdir(patient_path):
        continue

    for file in os.listdir(patient_path):

        if file.endswith(".tif") and "_mask" in file:

            mask_path = os.path.join(patient_path, file)
            img_path = os.path.join(patient_path, file.replace("_mask", ""))

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Resize
            img = cv2.resize(img, (128, 128))
            mask = cv2.resize(mask, (128, 128))

            # Normalize image
            img = img / 255.0

            # Binary mask
            mask = (mask > 127).astype(np.uint8)

            # Skip empty masks (important)
            if np.sum(mask) == 0:
                skipped += 1
                continue

            # Save
            cv2.imwrite(os.path.join(img_dir, f"img_{count}.png"), (img * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(mask_dir, f"mask_{count}.png"), mask * 255)

            count += 1

print("\n✅ DONE")
print(f"Final dataset size: {count}")
print(f"Skipped empty masks: {skipped}")