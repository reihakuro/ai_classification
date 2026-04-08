import os
import shutil
import random
from pathlib import Path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & TỈ LỆ
# ==========================================
current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

# Nguồn: Thư mục chứa 500 ảnh đã Crop & Augment
SRC_DIR = project_root / "data" / "raw"

# Đích: Thư mục mới chứa dữ liệu đã chia gọn gàng
DEST_DIR = project_root / "data" / "dataset_split" 

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def split_data():
    if not SRC_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {SRC_DIR}")
        return

    # Tạo trước các thư mục đích (train, val, test)
    for split_name in ["train", "val", "test"]:
        (DEST_DIR / split_name).mkdir(parents=True, exist_ok=True)

    print(f"[*] Bắt đầu chia dữ liệu từ {SRC_DIR} sang {DEST_DIR}...")
    
    # Đặt Seed để đảm bảo nếu bạn có lỡ chạy lại file này 2 lần, 
    # nó vẫn chia cùng một danh sách ảnh giống hệt nhau, không bị lộn xộn.
    random.seed(SEED) 

    for person_name in os.listdir(SRC_DIR):
        person_dir = SRC_DIR / person_name
        if not person_dir.is_dir():
            continue

        # Lấy danh sách ảnh và trộn ngẫu nhiên
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images) 
        
        total_images = len(images)
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        
        # Cắt mảng (Slicing) theo đúng tỉ lệ
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:] # Gom phần lẻ còn lại cho test

        # Tạo thư mục con và Copy file
        for split_name, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_person_dir = DEST_DIR / split_name / person_name
            split_person_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name in img_list:
                src_path = person_dir / img_name
                dest_path = split_person_dir / img_name
                shutil.copy2(src_path, dest_path)

        print(f"[+] {person_name}: {total_images} ảnh -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    print("\n[V] ĐÃ HOÀN TẤT CHIA DỮ LIỆU!")

if __name__ == "__main__":
    split_data()