import cv2
import os
import numpy as np
from pathlib import Path
import random

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

INPUT_DIR = project_root / "data" / "calibration"
OUTPUT_DIR = project_root / "data" / "processed"

TARGET_COUNT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

img_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

if len(img_files) == 0:
    print(f"Error: no images '{INPUT_DIR}'")
    exit()

print(f"Found {len(img_files)} original images. Proceeding with duplication and augmentation...")

current_count = 0
for f in img_files:
    img = cv2.imread(os.path.join(INPUT_DIR, f))
    out_path = os.path.join(OUTPUT_DIR, f"orig_{current_count:03d}.jpg")
    cv2.imwrite(out_path, img)
    current_count += 1


while current_count < TARGET_COUNT:
    random_file = random.choice(img_files)
    img = cv2.imread(os.path.join(INPUT_DIR, random_file))
    
    aug_type = random.randint(1, 3)
    
    if aug_type == 1:
        img_aug = cv2.flip(img, 1)
    elif aug_type == 2:
        value = random.randint(-40, 40)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img_aug = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    else:
        angle = random.randint(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_aug = cv2.warpAffine(img, M, (w, h))
        
    out_path = os.path.join(OUTPUT_DIR, f"aug_{current_count:03d}.jpg")
    cv2.imwrite(out_path, img_aug)
    current_count += 1

print(f"Complete! {TARGET_COUNT} images prepared in '{OUTPUT_DIR}'.")
