import os
import cv2
from pathlib import Path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
project_root = Path(__file__).resolve().parent

# Thư mục chứa ảnh gốc (chưa cắt)INPUT_DIR = project_root / "raw"
# Thư mục sẽ chứa ảnh ĐÃ CẮT (AI sẽ train bằng thư mục này)
OUTPUT_DIR = project_root / "cropped"

# Khởi tạo AI tìm mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# PADDING: Nới lỏng viền cắt thêm vài pixel để không bị lẹm cằm hay lẹm tóc
PADDING = 20  

def process_images():
    if not INPUT_DIR.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    for person_name in os.listdir(INPUT_DIR):
        person_dir = INPUT_DIR / person_name
        if not person_dir.is_dir():
            continue

        # Tạo thư mục tương ứng bên folder cropped
        out_person_dir = OUTPUT_DIR / person_name
        out_person_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[*] Đang quét thư mục: {person_name}...")
        valid_count = 0

        for img_name in os.listdir(person_dir):
            img_path = person_dir / img_name
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            # Haar Cascade cần ảnh xám để tìm mặt
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # minNeighbors=6: Ép điều kiện khắt khe để tránh nhận diện nhầm tường/áo thành mặt
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            # CHỈ XỬ LÝ NHỮNG ẢNH CÓ ĐÚNG 1 KHUÔN MẶT
            if len(faces) == 1:
                x, y, w, h = faces[0]

                # Tính toán vùng cắt có cộng thêm Padding (đảm bảo không bị lố ra khỏi viền ảnh)
                x_pad = max(0, x - PADDING)
                y_pad = max(0, y - PADDING)
                w_pad = min(img.shape[1] - x_pad, w + 2 * PADDING)
                h_pad = min(img.shape[0] - y_pad, h + 2 * PADDING)

                # Thực hiện cắt ảnh
                face_crop = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

                # Lưu ảnh sang thư mục mới
                out_path = out_person_dir / f"cropped_{valid_count}_{img_name}"
                cv2.imwrite(str(out_path), face_crop)
                valid_count += 1
                total_processed += 1

        print(f" -> Đã cắt thành công {valid_count} khuôn mặt.")

    print(f"\n[V] HOÀN TẤT! Đã thu hoạch tổng cộng {total_processed} khuôn mặt.")

if __name__ == "__main__":
    process_images()