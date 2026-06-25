"""
Entraînement CNN compact EcoPulse.

Commande :
python train_model.py

Entrée :
prepared_dataset/ecopulse_esc50_fsd50k.csv

Sorties :
models/ecopulse_cnn.keras
models/ecopulse_cnn_int8.tflite
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    MERGED_CSV,
    CLASSES,
    BATCH_SIZE,
    EPOCHS,
    SEED,
)
from audio_features import load_example


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    filepaths = df["filepath"].values
    labels = df["label_id"].values.astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_tiny_cnn(input_shape, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(24, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def representative_dataset(ds: tf.data.Dataset):
    for x, _ in ds.take(100):
        yield [tf.cast(x, tf.float32)]


def export_int8_tflite(model: tf.keras.Model, rep_ds: tf.data.Dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(rep_ds)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = MODEL_DIR / "ecopulse_cnn_int8.tflite"
    out_path.write_bytes(tflite_model)

    print(f"[OK] Export TFLite int8 : {out_path}")
    print(f"[INFO] Taille modèle : {out_path.stat().st_size / 1024:.1f} ko")


def main():
    tf.random.set_seed(SEED)

    if not MERGED_CSV.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {MERGED_CSV}. Lance d'abord build_dataset.py"
        )

    df = pd.read_csv(MERGED_CSV)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    sample_x, _ = next(iter(train_ds))
    input_shape = sample_x.shape[1:]

    print(f"[INFO] Input shape : {input_shape}")
    print(f"[INFO] Classes : {CLASSES}")

    model = build_tiny_cnn(input_shape=input_shape, num_classes=len(CLASSES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "ecopulse_cnn_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("[INFO] Évaluation test...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[RESULT] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    keras_path = MODEL_DIR / "ecopulse_cnn.keras"
    model.save(keras_path)
    print(f"[OK] Modèle Keras : {keras_path}")

    export_int8_tflite(model, train_ds)


if __name__ == "__main__":
    main()
