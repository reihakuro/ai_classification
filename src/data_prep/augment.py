import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default="data/dataset_split/train", help='Path to the raw data directory')
args = parser.parse_args()

current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

DATA_DIR = project_root / args.src if not Path(args.src).is_absolute() else Path(args.src)

TARGET_IMAGES_PER_CLASS = 1000

datagen = ImageDataGenerator(
    rotation_range=15,        
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    brightness_range=[0.7, 1.3], 
    zoom_range=0.1,          
    horizontal_flip=True,
    vertical_flip=True,     
    fill_mode='nearest'       
)

def augment_dataset():
    if not DATA_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {DATA_DIR}")
        return

    for person_name in os.listdir(DATA_DIR):
        person_dir = DATA_DIR / person_name
        if not person_dir.is_dir():
            continue

        existing_images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(existing_images)

        if current_count >= TARGET_IMAGES_PER_CLASS:
            print(f"[*] {person_name}: Đã có {current_count} ảnh. Bỏ qua.")
            continue

        needed_images = TARGET_IMAGES_PER_CLASS - current_count
        print(f"[*] {person_name}: Hiện có {current_count} ảnh. Đang sinh thêm {needed_images} ảnh...")

        images_data = []
        for img_name in existing_images:
            img_path = person_dir / img_name
            img = load_img(img_path, target_size=(160, 160))
            x = img_to_array(img)
            images_data.append(x)
        
        images_data = np.array(images_data)
        
        generated_count = 0
        
        for batch in datagen.flow(
            images_data, 
            batch_size=1, 
            save_to_dir=person_dir, 
            save_prefix="aug", 
            save_format="jpg"
        ):
            generated_count += 1
            if generated_count >= needed_images:
                break 
                
        print(f" -> Đã hoàn thành thư mục của {person_name}. Tổng: {TARGET_IMAGES_PER_CLASS} ảnh.")

    print("\n[V] ĐÃ HOÀN TẤT NHÂN BẢN DỮ LIỆU!")

if __name__ == "__main__":
    augment_dataset()