
import os
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

version = input("Checking what version?: ")
SAVE_DIR   = project_root / "models" / f"tf_cnn_face_model_v{version}"
MODEL_PATH = SAVE_DIR / "best.keras"
META_PATH  = SAVE_DIR / "class_names.json"
IMG_PATH   = project_root / "test.jpg"

def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = tuple(meta["img_size"])

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Error: Image not found '{IMG_PATH}'.")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    print("[*] Running warm-up...")
    for _ in range(5):
        _ = model(x, training=False)

    print("[*] Starting benchmark 200 times...")
    latencies = []
    
    for i in range(200):
        t0 = time.perf_counter()
        
        preds = model(x, training=False) 
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    preds_np = preds.numpy()
    pred_idx = int(np.argmax(preds_np[0]))
    
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)

    print("\nPredict:", class_names[pred_idx])
    print("--- Benchmark Results best.keras (FP32) ---")
    print(f"Mean Latency : {mean_latency:.2f} ms")
    print(f"P50 Latency  : {p50_latency:.2f} ms")
    print(f"P90 Latency  : {p90_latency:.2f} ms")
    print(f"P99 Latency  : {p99_latency:.2f} ms")

if __name__ == "__main__":
    main()
