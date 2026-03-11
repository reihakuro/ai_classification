import os
import json
import cv2
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "tf_cnn_face_model_v1")
MODEL_PATH = os.path.join(SAVE_DIR, "best.keras")
META_PATH = os.path.join(SAVE_DIR, "class_names.json")

REP_DIR = os.path.join(ROOT, "calibration_dupe")
REP_SAMPLES = 500

OUT_INT8 = os.path.join(SAVE_DIR, "model_int8.tflite")

def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return tuple(meta["img_size"])

def rep_data_gen(img_size):
    h, w = img_size
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(REP_DIR):
        print(f"LỖI NGHIÊM TRỌNG: Thư mục '{REP_DIR}' không tồn tại!")
        return

    paths = [os.path.join(REP_DIR, f) for f in os.listdir(REP_DIR)]
    paths = paths[:REP_SAMPLES]
    
    valid_count = 0 
    print(f"[*] Đang tìm ảnh trong thư mục: {REP_DIR}")
    print(f"[*] Tổng số file quét được: {len(paths)}")

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        
        valid_count += 1
        # Ép x thành tensor trước khi yield để TFLite Converter nhận diện đúng
        yield [tf.convert_to_tensor(x, dtype=tf.float32)]
        
    print(f"[*] Hoàn tất nạp dữ liệu: Có {valid_count} ảnh hợp lệ được đưa vào lượng tử hóa.")

def main():
    img_size = load_meta()
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_data_gen(img_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_int8 = converter.convert()
    with open(OUT_INT8, "wb") as f:
        f.write(tflite_int8)

    print("Saved:", OUT_INT8)

if __name__ == "__main__":
    main()
