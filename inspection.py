import os

root = r"D:\Brain_Segmentation\brain_segmentation_data\lgg-mri-segmentation\kaggle_3m"

total_images = 0
total_masks = 0
total_pairs = 0

print("\n🔍 FINAL DATASET INSPECTION\n")

for patient in os.listdir(root):
    patient_path = os.path.join(root, patient)

    if not os.path.isdir(patient_path):
        continue

    files = os.listdir(patient_path)

    images = []
    masks = []

    for f in files:
        f_lower = f.lower()

        if f_lower.endswith(".tif"):  # ✅ FIXED
            if "mask" in f_lower:
                masks.append(f)
            else:
                images.append(f)

    # Pair check
    pairs = 0
    for m in masks:
        img_name = m.replace("_mask", "")
        if img_name in images:
            pairs += 1

    total_images += len(images)
    total_masks += len(masks)
    total_pairs += pairs

    print(f"🧠 {patient}")
    print(f"   Images: {len(images)}, Masks: {len(masks)}, Pairs: {pairs}")
    print(f"   Sample Image: {images[0] if images else 'None'}")
    print(f"   Sample Mask : {masks[0] if masks else 'None'}\n")

print("📊 SUMMARY")
print(f"Total Images: {total_images}")
print(f"Total Masks: {total_masks}")
print(f"Total Pairs: {total_pairs}")