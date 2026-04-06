import os
import json
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent.parent 

SAVE_DIR = ROOT / "models" / "tf_cnn_face_model_v2"
MODEL_PATH = SAVE_DIR / "best.keras"
META_PATH = SAVE_DIR / "class_names.json"

REP_DIR = ROOT / "data" / "processed"
REP_SAMPLES = 500

OUT_FP32 = SAVE_DIR / "model_fp32.tflite"
OUT_INT8 = SAVE_DIR / "model_int8.tflite"

def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return tuple(meta["img_size"])

def rep_data_gen(img_size):
    h, w = img_size
    
    if not os.path.exists(REP_DIR):
        print(f"'{REP_DIR}' does not exist")
        return

    paths = [os.path.join(REP_DIR, f) for f in os.listdir(REP_DIR)]
    paths = paths[:REP_SAMPLES]
    
    valid_count = 0 
    print(f"[*] Searching for images in directory: {REP_DIR}")
    print(f"[*] Total files scanned: {len(paths)}")

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        
        valid_count += 1
        yield [tf.convert_to_tensor(x, dtype=tf.float32)]
        
    print(f"[*] {valid_count} valid images found.")

def main():
    img_size = load_meta()
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nConverting to FP32 format...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()
    
    with open(OUT_FP32, "wb") as f:
        f.write(tflite_fp32)
    print("Successfully saved FP32:", OUT_FP32)

    
    print("\n[*] Converting to INT8 format...")
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = lambda: rep_data_gen(img_size)
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.uint8

    tflite_int8 = converter_int8.convert()
    with open(OUT_INT8, "wb") as f:
        f.write(tflite_int8)

    print("Successfully saved INT8:", OUT_INT8)

if __name__ == "__main__":
    main()