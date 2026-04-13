import json
import cv2
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='1.0', help='Version tag cho model')
args = parser.parse_args()

current_file = Path(__file__).resolve().parent
ROOT = current_file.parent.parent 

SAVE_DIR = ROOT / "models" / f"tf_cnn_face_model_v{version}"
MODEL_PATH = SAVE_DIR / "best.keras"
META_PATH = SAVE_DIR / "class_names.json"
REP_DIR = ROOT / "data" / "processed" 
REP_SAMPLES = 500
OUT_INT8 = SAVE_DIR / "model_int8.tflite"

def load_meta():
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return tuple(meta["img_size"])

def rep_data_gen(img_size):
    h, w = img_size

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = [p for p in REP_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    paths = paths[:REP_SAMPLES]
    
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        
 
        x = img.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        yield [x]

def main():
    img_size = load_meta()
    print(f"Input size loaded: {img_size}")
    
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_data_gen(img_size)
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Quantization to INT8...")
    tflite_int8 = converter.convert()
    
    OUT_INT8.write_bytes(tflite_int8)

    print("[+] Successfully saved INT8 TFLite model at:", OUT_INT8)

if __name__ == "__main__":
    main()