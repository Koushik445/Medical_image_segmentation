import matplotlib.pyplot as plt
import cv2
import random
import os

img_dir = r"D:\processed_dataset\images"
mask_dir = r"D:\processed_dataset\masks"

files = os.listdir(img_dir)

for _ in range(3):
    idx = random.randint(0, len(files)-1)

    img = cv2.imread(os.path.join(img_dir, files[idx]), 0)
    mask = cv2.imread(os.path.join(mask_dir, files[idx].replace("img", "mask")), 0)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("MRI")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")

    plt.show()