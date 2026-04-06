import os
import sys
import json
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(ROOT, "tf_cnn_face_model_v1")
MODEL_PATH = os.path.join(SAVE_DIR, "best.keras")
LABELS_JSON = os.path.join(SAVE_DIR, "class_names.json")


def load_meta():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = tuple(meta["img_size"])
    return class_names, img_size


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_tf_image.py path/to/image.jpg")
        return

    img_path = sys.argv[1]
    class_names, img_size = load_meta()

    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    print("Pred:", class_names[pred])
    print("Conf:", conf)


if __name__ == "__main__":
    main()
