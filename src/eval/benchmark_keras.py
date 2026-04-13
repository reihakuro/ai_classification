import os
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='1.0', help='Version tag for the model')
args = parser.parse_args()
version = args.version

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

SAVE_DIR   = project_root / "models" / f"tf_cnn_face_model_v{version}"
MODEL_PATH = SAVE_DIR / "best.keras"
META_PATH  = SAVE_DIR / "class_names.json"
TEST_DIR   = project_root / "data" / "test"

def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = tuple(meta["img_size"])

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(TEST_DIR.rglob(ext))
        
    if not image_paths:
        print(f"Error: No images found in '{TEST_DIR}'.")
        return

    print(f"Found {len(image_paths)} images for testing.")

    print("[*] Running warm-up...")
    dummy_x = np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32)
    for _ in range(5):
        _ = model(dummy_x, training=False)

    print("[*] Starting benchmark on dataset...")
    latencies = []
    correct_predictions = 0
    valid_images = 0
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Tiền xử lý ảnh
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        t0 = time.perf_counter()
        preds = model(x, training=False) 
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000)
        
        true_label = img_path.parent.name 
        preds_np = preds.numpy()
        pred_idx = int(np.argmax(preds_np[0]))
        pred_label = class_names[pred_idx]
        
        if true_label == pred_label:
            correct_predictions += 1
        valid_images += 1

    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    
    accuracy = (correct_predictions / valid_images) * 100 if valid_images > 0 else 0

    print("\n--- Benchmark Results best.keras (FP32) ---")
    print(f"Total Images : {valid_images}")
    if set(p.parent.name for p in image_paths).issubset(set(class_names)):
        print(f"Accuracy     : {accuracy:.2f}% ({correct_predictions}/{valid_images})")
    else:
        print("Accuracy     : No directory labels found")
        
    print(f"Mean Latency : {mean_latency:.2f} ms")
    print(f"P50 Latency  : {p50_latency:.2f} ms")
    print(f"P90 Latency  : {p90_latency:.2f} ms")
    print(f"P99 Latency  : {p99_latency:.2f} ms")

if __name__ == "__main__":
    main()
