import os
from pathlib import Path
import tensorflow as tf

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent.parent 

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "tf_cnn_face_model_v2pruned"

MODEL_PATH = MODEL_DIR / "best_pruned.keras"
OUT_FP32 = MODEL_DIR / "pruned_fp32.tflite"
OUT_INT8 = MODEL_DIR / "pruned_int8.tflite"

IMG_SIZE = (96, 96)
BATCH_SIZE = 32

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
print("\nExporting TFLite model in FP32 format...")
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter_fp32.convert()

with open(OUT_FP32, "wb") as f:
    f.write(tflite_fp32)
print(f"Successfully saved FP32 at: {OUT_FP32}")


print("\nLoading representative data for INT8 quantization  ...")


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None, 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

def representative_dataset():
    for x in train_ds.take(10): 
        yield [x]

print("Exporting TFLite model in INT8 format...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_int8 = converter_int8.convert()

with open(OUT_INT8, "wb") as f:
    f.write(tflite_int8)
print(f"Successfully saved INT8 at: {OUT_INT8}")
