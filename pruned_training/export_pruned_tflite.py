import os
import tensorflow as tf

# =========================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cập nhật đường dẫn (thay đổi linh hoạt nếu bạn đã chia thư mục)
DATA_DIR = os.path.join(CURRENT_DIR, "face_dataset")
MODEL_DIR = os.path.join(CURRENT_DIR, "tf_cnn_face_model_v1pruned") # Thư mục lưu file pruned.keras

MODEL_PATH = os.path.join(MODEL_DIR, "pruned.keras")
OUT_FP32 = os.path.join(MODEL_DIR, "pruned_fp32.tflite")
OUT_INT8 = os.path.join(MODEL_DIR, "pruned_int8.tflite")

IMG_SIZE = (96, 96) # Giữ nguyên 96x96 theo file train
BATCH_SIZE = 32

# =========================
# 2. LOAD MÔ HÌNH
# =========================
print(f"Đang tải mô hình từ: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# 3. EXPORT ĐỊNH DẠNG FP32
# =========================
print("\nĐang xuất mô hình TFLite định dạng FP32...")
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter_fp32.convert()

with open(OUT_FP32, "wb") as f:
    f.write(tflite_fp32)
print(f"Đã lưu thành công FP32 tại: {OUT_FP32}")

# =========================
# 4. EXPORT ĐỊNH DẠNG INT8
# =========================
print("\nĐang load dữ liệu mẫu để lượng tử hoá INT8...")

# Khởi tạo train_ds ngay trong script này
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None, # Quá trình calibrate chỉ cần ảnh, không cần quan tâm nhãn
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

def representative_dataset():
    # Lấy khoảng vài batch (tương đương 200 mẫu) để căn chỉnh
    for x in train_ds.take(10): 
        yield [x]

print("Đang xuất mô hình TFLite định dạng INT8 (quá trình này có thể mất vài phút)...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_int8 = converter_int8.convert()

with open(OUT_INT8, "wb") as f:
    f.write(tflite_int8)
print(f"Đã lưu thành công INT8 tại: {OUT_INT8}")
