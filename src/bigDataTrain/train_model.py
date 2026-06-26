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
from sklearn.utils.class_weight import compute_class_weight

from config import (
    MERGED_CSV,
    CLASSES,
    UNKNOWN_THRESHOLD,
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
    # ds = ds.map(
    #     lambda fp, y: load_example(fp, y, training=training),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    # )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_tiny_cnn(input_shape, num_classes: int) -> tf.keras.Model:
    """
    CNN compact mais moins sous-capacitaire que la version initiale.

    Objectif :
    - rester compatible TinyML après quantification int8 ;
    - viser un modèle utile, pas seulement 2k paramètres ;
    - améliorer la capacité sur 6 classes audio réelles.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.SeparableConv2D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.SeparableConv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.SeparableConv2D(160, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

def representative_dataset(ds: tf.data.Dataset):
    for x, _ in ds.take(100):
        yield [tf.cast(x, tf.float32)]


def export_int8_tflite(model: tf.keras.Model, rep_ds: tf.data.Dataset):
    for layer in model.layers:
        layer.trainable = False

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, *model.input_shape[1:]],
                dtype=tf.float32,
            )
        ]
    )
    def serving_fn(x):
        return model(x, training=False)

    concrete_func = serving_fn.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func],
        model,
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for x, _ in rep_ds.unbatch().batch(1).take(200):
            yield [tf.cast(x, tf.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
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
    print(f"[INFO] Seuil unknown conseillé en inférence : {UNKNOWN_THRESHOLD}")

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

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASSES)),
        y=train_df["label_id"].values,
    )

    class_weight = {
        i: float(w)
        for i, w in enumerate(class_weights_array)
    }

    print("[INFO] Class weights :")
    print(class_weight)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
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
