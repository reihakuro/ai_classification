#==================================================================
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

DATA_FILE = project_root / "data"
DATA_DIR = DATA_FILE / "raw"

# Hyperparameters
IMG_SIZE = (96, 96)		
BATCH_SIZE = 32			
EPOCHS = 15			
LR = 1e-3			
SEED = 42			

# DATA SPLIT RATIOS
TRAIN_RATIO = 0.70 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15

SAVE_DIR = project_root / "models" / "tf_cnn_face_model"
LABELS_JSON = SAVE_DIR / "class_names.json"


def ensure_data_dir():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Can't find data directory: {DATA_DIR}")


def list_class_names():
    class_names = sorted([
        d.name for d in DATA_DIR.iterdir() if d.is_dir()
    ])
    if len(class_names) < 2:
        raise ValueError("Can't have less than 2 classes (2 people) for classification training.")
    return class_names

def make_datasets():
    """
    Create train, validation, and test datasets from the image directory.
    The function uses Keras's image_dataset_from_directory to load images and split them into training
    """
    val_split = VAL_RATIO

    train_full = tf.keras.utils.image_dataset_from_directory( 
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=val_split,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=val_split,
        subset="validation",
    )

    class_names = train_full.class_names
    num_classes = len(class_names)

    test_in_train_full_ratio = TEST_RATIO / (1.0 - VAL_RATIO) #LẤY 15% TRONG 85% ĐỂ TÁCH TEST 

    n_batches = tf.data.experimental.cardinality(train_full).numpy()
    n_test_batches = max(1, int(round(n_batches * test_in_train_full_ratio)))
#ĐẾM SỐ BATCH ĐỂ CHIA
    test_ds = train_full.take(n_test_batches) # LẤY VÀI BATCH ĐẦU ĐỂ TEST
    train_ds = train_full.skip(n_test_batches) # PHẦN CÒN LẠI ĐỂ TRAIN

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE) # LOAD DỮ LIỆU SONG SONG
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, num_classes

# BUILD CNN MODEL
def build_model(num_classes: int):
	#tạo data ảo
    data_augmentation = tf.keras.Sequential([ # Tăng dữ liệu bằng cách lật ngang, xoay nhẹ, zoom, đổi độ tương phản bằng thuật toán Random #
        tf.keras.layers.RandomFlip("horizontal"), #lật ảnh
        tf.keras.layers.RandomRotation(0.05), # xoay
        tf.keras.layers.RandomZoom(0.05), # zoom/thu phóng
        tf.keras.layers.RandomContrast(0.1), #độ tương phản
    ], name="augment")
# Standardize pixel values to [0,1] and apply data augmentation
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)) #(96,96,)3
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs) #chia tỉ lệ
    x = data_augmentation(x)
#Conv Block 1: 32 filters, kernel 3x3 
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x) 
    x = tf.keras.layers.MaxPool2D()(x)
#Conv Block 2 learn low-level features like edges, textures
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
#Conv Block 3 learn high-level features like shapes, facial parts
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
#FULLY CONNECTED
# chuyển feature map qua vector
# Dense học phân loại
# Softmax xuất xác suất từng người
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="SimpleCNN_FaceClassifier")
    return model

# ĐÁNH GIÁ KẾT QUẢ BẰNG MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
def eval_test(model, test_ds, class_names):
    y_true_all = []
    y_pred_all = []

    for x, y_true in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred = np.argmax(probs, axis=1) 
        y_true_all.extend(y_true.numpy().tolist())
        y_pred_all.extend(y_pred.tolist())
# Ma trận nhầm lẫn
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true_all, y_pred_all))

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=class_names,
        digits=4
    ))

def main():
    tf.random.set_seed(SEED) 
    np.random.seed(SEED)

    ensure_data_dir()
    class_names = list_class_names()
    print("Classes:", class_names)

    train_ds, val_ds, test_ds, class_names, num_classes = make_datasets()
    print("Num classes:", num_classes)

    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint( 
            filepath=str(SAVE_DIR / "best.keras"), 
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

# SAVING MODEL AND LABELS
    model.save(str(SAVE_DIR / "final.keras"))
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "img_size": list(IMG_SIZE)}, f, ensure_ascii=False, indent=2)

    print("\nSaved model to:", SAVE_DIR)
    print("Saved labels to:", LABELS_JSON)

    best_model = tf.keras.models.load_model(str(SAVE_DIR / "best.keras"))
    print("\n=== Final Test Evaluation (best model) ===")
    eval_test(best_model, test_ds, class_names)


if __name__ == "__main__":
    main()
