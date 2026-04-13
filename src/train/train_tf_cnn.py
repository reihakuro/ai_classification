import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='Version tag cho model')
args = parser.parse_args()

version = args.version

# ==========================================
# PARAMETERS & PATHS
# ==========================================
current_file = Path(__file__).resolve().parent
project_root = current_file.parent.parent 

DATASET_BASE = project_root / "data" / "dataset_split"
TRAIN_DIR = DATASET_BASE / "train"
VAL_DIR = DATASET_BASE / "val"
TEST_DIR = DATASET_BASE / "test"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32         
EPOCHS = 100 
LR_INIT = 1e-4 
SEED = 42           

SAVE_DIR  = project_root / "models" / f"tf_cnn_face_model_v{version}"
LABELS_JSON = SAVE_DIR / "class_names.json"

# ==========================================
# LOADING DATASET
# ==========================================
def make_datasets():
    print("Loading datasets...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory( 
        TRAIN_DIR, labels="inferred", label_mode="int",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="int",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="int",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(AUTOTUNE), 
        val_ds.prefetch(AUTOTUNE), 
        test_ds.prefetch(AUTOTUNE), 
        class_names, 
        num_classes
    )

# ==========================================
# MOBILENETV2 + FINE-TUNING
# ==========================================
def build_model(num_classes: int):
    
    data_augmentation = tf.keras.Sequential([ 
        tf.keras.layers.RandomFlip("horizontal"), 
        tf.keras.layers.RandomRotation(0.1), 
        tf.keras.layers.RandomZoom(0.1), 
    ], name="augment")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True 
    for layer in base_model.layers[:100]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    
    x = base_model(x, training=False)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="FaceNet_MobileNetV2_Final")
    return model

# ==========================================
# EVALUATION
# ==========================================
def eval_test(model, test_ds, class_names):
    y_true_all = []
    y_pred_all = []

    for x, y_true in test_ds:
        probs = model.predict(x, verbose=2)
        y_pred = np.argmax(probs, axis=1) 
        y_true_all.extend(y_true.numpy().tolist())
        y_pred_all.extend(y_pred.tolist())

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true_all, y_pred_all))

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

# ==========================================
# 5. MAIN FUNCTION
# ==========================================
def main():

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, num_classes = make_datasets()
    
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INIT),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint( 
            filepath=str(SAVE_DIR / "best.keras"), 
            monitor="val_accuracy", mode="max", save_best_only=True, verbose=2
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=2
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-7, verbose=2
        )
    ]

    print("\nTraining...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names, 
            "img_size": list(IMG_SIZE)
        }, f, ensure_ascii=False, indent=2)

    print("\nEvaluating on Test set...")
    best_model = tf.keras.models.load_model(str(SAVE_DIR / "best.keras"))
    eval_test(best_model, test_ds, class_names)

if __name__ == "__main__":
    main()