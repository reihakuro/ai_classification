import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

version = input("Checking what version?: ")
SAVE_DIR = project_root / "models" / f"tf_cnn_face_model_v{version}"
MODEL_PATH = SAVE_DIR / "best.keras"
META_PATH = SAVE_DIR / "class_names.json"

TEST_IMAGE_PATH = project_root / "test.jpg"

def main():
   
    if not META_PATH.exists():
        print(f"Cant find {META_PATH}")
        return
        
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    class_names = meta["class_names"]
    img_size = tuple(meta["img_size"]) 

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    if not TEST_IMAGE_PATH.exists():
        print(f"[!] Error: Image not found at {TEST_IMAGE_PATH}")
        return

    img = cv2.imread(str(TEST_IMAGE_PATH))
    if img is None:
        print("[!] Error: Unable to read image.")
        return

    print(f"Processing image: {TEST_IMAGE_PATH.name}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

    print("Processing...")
    predictions = model.predict(input_data, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = class_names[predicted_idx]

    print("\n" + "="*30)
    print(" RESULT")
    print("="*30)
    print(f"- Person in image : {predicted_class}")
    print(f"- Confidence      : {confidence * 100:.2f}%")
    print("\n[Detailed probabilities for all people]:")
    
    for i, name in enumerate(class_names):
        print(f"  + {name:<10}: {predictions[0][i] * 100:.2f}%")

if __name__ == "__main__":
    main()