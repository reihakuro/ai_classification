import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

DATA_DIR = project_root / "data" / "raw"

TARGET_IMAGES_PER_CLASS = 500

datagen = ImageDataGenerator(
    rotation_range=15,        # Xoay ngẫu nhiên +- 15 độ
    width_shift_range=0.1,    # Dịch qua trái/phải 10%
    height_shift_range=0.1,   # Dịch lên/xuống 10%
    brightness_range=[0.7, 1.3], # Tối đi 30% hoặc sáng lên 30%
    zoom_range=0.1,           # Phóng to/thu nhỏ 10%
    horizontal_flip=True,     # Lật gương (rất quan trọng)
    fill_mode='nearest'       # Lấp đầy các điểm ảnh bị trống khi xoay
)

def augment_dataset():
    if not DATA_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {DATA_DIR}")
        return

    for person_name in os.listdir(DATA_DIR):
        person_dir = DATA_DIR / person_name
        if not person_dir.is_dir():
            continue

        # Đếm số ảnh hiện tại của người này
        existing_images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(existing_images)

        if current_count >= TARGET_IMAGES_PER_CLASS:
            print(f"[*] {person_name}: Đã có {current_count} ảnh. Bỏ qua.")
            continue

        needed_images = TARGET_IMAGES_PER_CLASS - current_count
        print(f"[*] {person_name}: Hiện có {current_count} ảnh. Đang sinh thêm {needed_images} ảnh...")

        # Load toàn bộ ảnh hiện tại lên RAM để làm "con giống"
        images_data = []
        for img_name in existing_images:
            img_path = person_dir / img_name
            img = load_img(img_path, target_size=(160, 160))
            x = img_to_array(img)
            images_data.append(x)
        
        images_data = np.array(images_data)
        
        # Bắt đầu vòng lặp sinh ảnh
        generated_count = 0
        
        # Hàm flow() sẽ ngẫu nhiên chọn ảnh gốc, biến tấu nó và lưu thẳng xuống ổ cứng
        for batch in datagen.flow(
            images_data, 
            batch_size=1, 
            save_to_dir=person_dir, 
            save_prefix="aug", 
            save_format="jpg"
        ):
            generated_count += 1
            if generated_count >= needed_images:
                break # Dừng lại khi đã đạt đủ Target
                
        print(f" -> Đã hoàn thành thư mục của {person_name}. Tổng: {TARGET_IMAGES_PER_CLASS} ảnh.")

    print("\n[V] ĐÃ HOÀN TẤT NHÂN BẢN DỮ LIỆU!")

if __name__ == "__main__":
    augment_dataset()