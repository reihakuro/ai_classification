import cv2
import os
import numpy as np
import random

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))

INPUT_DIR = os.path.join(project_root, "calibration")
OUTPUT_DIR = os.path.join(project_root, "calibration_dupe")

TARGET_COUNT = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tạo thư mục đích nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Lấy danh sách ảnh gốc
img_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

if len(img_files) == 0:
    print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục '{INPUT_DIR}'")
    exit()

print(f"Tìm thấy {len(img_files)} ảnh gốc. Đang tiến hành nhân bản và biến đổi...")

# 1. Copy toàn bộ ảnh gốc sang thư mục đích trước
current_count = 0
for f in img_files:
    img = cv2.imread(os.path.join(INPUT_DIR, f))
    out_path = os.path.join(OUTPUT_DIR, f"orig_{current_count:03d}.jpg")
    cv2.imwrite(out_path, img)
    current_count += 1

# 2. Tự động tạo thêm ảnh bằng cách biến đổi ngẫu nhiên cho đến khi đủ 200
while current_count < TARGET_COUNT:
    # Chọn ngẫu nhiên 1 ảnh gốc
    random_file = random.choice(img_files)
    img = cv2.imread(os.path.join(INPUT_DIR, random_file))
    
    # Chọn ngẫu nhiên 1 trong 3 phương pháp biến đổi
    aug_type = random.randint(1, 3)
    
    if aug_type == 1:
        # Lật ngang ảnh
        img_aug = cv2.flip(img, 1)
    elif aug_type == 2:
        # Thay đổi độ sáng ngẫu nhiên
        value = random.randint(-40, 40)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img_aug = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    else:
        # Xoay nhẹ ảnh ngẫu nhiên (từ -15 đến 15 độ)
        angle = random.randint(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_aug = cv2.warpAffine(img, M, (w, h))
        
    # Lưu ảnh mới vào thư mục đích
    out_path = os.path.join(OUTPUT_DIR, f"aug_{current_count:03d}.jpg")
    cv2.imwrite(out_path, img_aug)
    current_count += 1

print(f"Hoàn tất! Đã chuẩn bị xong {TARGET_COUNT} ảnh trong thư mục '{OUTPUT_DIR}'.")
