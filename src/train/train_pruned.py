import os
import tensorflow as tf

# =========================
# PROJECT ROOT
# =========================
ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================
# CONFIG
# =========================
DATA_DIR = os.path.join(ROOT, "face_dataset")

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEED = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15

SAVE_DIR = os.path.join(ROOT, "tf_cnn_face_model_v1pruned")

os.makedirs(SAVE_DIR, exist_ok=True)
# =========================
# 2. HÀM LOAD DỮ LIỆU
# =========================
def make_datasets():
    # Load training data
    train_full = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int", # Chú ý: trả về label dạng số nguyên
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_RATIO,
        subset="training",
    )

    # Load validation data
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

    # Tối ưu tải dữ liệu
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_full.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes

# =========================
# 3. BUILD PRUNED MODEL (Giảm filters)
# =========================
def build_pruned_model(num_classes):
    # Giữ lại Data Augmentation để model học tốt hơn
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augment")

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)

    # STRUCTURED PRUNING: Giảm số filters đi một nửa so với baseline
    # Conv Block 1: 32 -> 16
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Conv Block 2: 64 -> 32
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Flatten & Dense
    x = tf.keras.layers.Flatten()(x)
    # Giảm nốt layer Dense từ 256 xuống 64 theo tài liệu lab
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name="PrunedCNN_FaceClassifier")
    return model

# =========================
# 4. HÀM MAIN (Khởi chạy)
# =========================
def main():
    tf.random.set_seed(SEED)

    print("Đang load dữ liệu...")
    train_ds, val_ds, class_names, num_classes = make_datasets()
    print(f"Số lớp (classes): {num_classes} - {class_names}")

    print("\nĐang khởi tạo Pruned Model...")
    model = build_pruned_model(num_classes)
    
    # CHÚ Ý: Dùng sparse_categorical_crossentropy vì label_mode="int"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nBắt đầu huấn luyện mô hình cắt tỉa (Pruned)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Lưu lại model đuôi .keras
    save_path = os.path.join(SAVE_DIR, "pruned.keras")
    model.save(save_path)
    print(f"\nĐã huấn luyện xong! Đã lưu mô hình Pruned tại: {save_path}")

if __name__ == "__main__":
    main()
