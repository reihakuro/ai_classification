import os
import shutil
import random
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default="data/raw", help='Path to the raw data directory')
parser.add_argument('--dest', type=str, default="data/dataset_split", help='Path to the destination directory')
args = parser.parse_args()

current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

SRC_DIR = project_root / args.src if not Path(args.src).is_absolute() else Path(args.src)
DEST_DIR = project_root / args.dest if not Path(args.dest).is_absolute() else Path(args.dest)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def split_data():
    if not SRC_DIR.exists():
        print(f"Cant find {SRC_DIR}")
        return

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
        
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:] 

        for split_name, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_person_dir = DEST_DIR / split_name / person_name
            split_person_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name in img_list:
                src_path = person_dir / img_name
                dest_path = split_person_dir / img_name
                shutil.copy2(src_path, dest_path)

        print(f"[+] {person_name}: {total_images} images -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    print("\nSplit data completed successfully!")

if __name__ == "__main__":
    split_data()