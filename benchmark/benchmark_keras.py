
import os
import json
import time
import numpy as np
import cv2
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR   = os.path.join(ROOT, "tf_cnn_face_model_v1")
MODEL_PATH = os.path.join(SAVE_DIR, "best.keras")
META_PATH  = os.path.join(SAVE_DIR, "class_names.json")

IMG_PATH   = os.path.join(ROOT, "test.jpg")
def main():
    # 1. Tải metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = tuple(meta["img_size"])

    # 2. Tải mô hình FP32 gốc
    print("[*] Đang tải mô hình Keras FP32...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Chuẩn bị ảnh đầu vào
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"LỖI: Không tìm thấy ảnh '{IMG_PATH}'.")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # 4. Warm-up (Chạy mồi)
    # RẤT QUAN TRỌNG: Lần chạy đầu tiên của TF luôn mất nhiều thời gian để khởi tạo đồ thị.
    # Ta cần chạy mồi vài lần để không làm sai lệch chỉ số Mean và P99.
    print("[*] Đang chạy warm-up...")
    for _ in range(5):
        _ = model(x, training=False)

    # 5. Tiến hành Benchmark 200 lần
    print("[*] Bắt đầu chạy benchmark 200 lần...")
    latencies = []
    
    for i in range(200):
        t0 = time.perf_counter()
        # Dùng model(x) thay vì model.predict(x) để đo tốc độ suy luận cốt lõi, bỏ qua overhead của keras
        preds = model(x, training=False) 
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    # 6. Xử lý và in kết quả
    preds_np = preds.numpy()
    pred_idx = int(np.argmax(preds_np[0]))
    
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)

    print("\nPredict:", class_names[pred_idx])
    print("--- Kết quả Benchmark best.keras (FP32) ---")
    print(f"Mean Latency : {mean_latency:.2f} ms")
    print(f"P50 Latency  : {p50_latency:.2f} ms")
    print(f"P90 Latency  : {p90_latency:.2f} ms")
    print(f"P99 Latency  : {p99_latency:.2f} ms")

if __name__ == "__main__":
    main()
