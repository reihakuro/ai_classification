import os
from pathlib import Path
import tensorflow as tf

current_file = Path(__file__).resolve().parent
ROOT = current_file.parent.parent 
DATA_DIR = ROOT / "data" / "raw"

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEED = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15

SAVE_DIR = ROOT / "models" / "tf_cnn_face_model_pruned"

def make_datasets():
    train_full = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int", 
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_RATIO,
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
        validation_split=VAL_RATIO,
        subset="validation",
    )

    class_names = train_full.class_names
    num_classes = len(class_names)

    test_in_train_full_ratio = TEST_RATIO / (1.0 - VAL_RATIO)
    n_batches = tf.data.experimental.cardinality(train_full).numpy()
    n_test_batches = max(1, int(round(n_batches * test_in_train_full_ratio)))
    
    test_ds = train_full.take(n_test_batches)
    train_ds = train_full.skip(n_test_batches)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, num_classes

def build_pruned_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augment")

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)

    # STRUCTURED PRUNING
    # Conv Block 1: 32 -> 16
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Conv Block 2: 64 -> 32
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Flatten & Dense
    x = tf.keras.layers.Flatten()(x)
    # Layer Dense 128 -> 64
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="PrunedCNN_FaceClassifier")
    return model

def main():
    tf.random.set_seed(SEED)

    print("Load...")

    train_ds, val_ds, test_ds, class_names, num_classes = make_datasets()
    print(f"Classes: {num_classes} - {class_names}")

    print("\nBuilding pruned model...")
    model = build_pruned_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint( 
            filepath=str(SAVE_DIR / "best_pruned.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )
    ]

    print("\nStarting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks 
    )

    save_path = SAVE_DIR / "final_pruned.keras"
    model.save(str(save_path)) 
    print(f"\nTraining completed! Final model saved at: {save_path}")

if __name__ == "__main__":
    main()
