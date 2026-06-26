import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

from config import MERGED_CSV, CLASSES, BATCH_SIZE
from audio_features import load_example


MODEL_PATH = "models/ecopulse_cnn.keras"


def make_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    filepaths = df["filepath"].values
    labels = df["label_id"].values.astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def main():
    df = pd.read_csv(MERGED_CSV)
    test_df = df[df["split"] == "test"].copy()

    test_ds = make_dataset(test_df)

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []
    y_conf = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)
        y_conf.extend(confs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)

    print("\n[INFO] Classification report :")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASSES,
            digits=4,
        )
    )

    print("\n[INFO] Matrice de confusion brute :")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n[INFO] Matrice de confusion normalisée par vraie classe :")
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    print(np.round(cm_norm, 3))

    print("\n[INFO] Confiance moyenne :")
    for idx, class_name in enumerate(CLASSES):
        mask = y_true == idx
        if mask.sum() == 0:
            continue

        acc_class = (y_pred[mask] == y_true[mask]).mean()
        conf_class = y_conf[mask].mean()

        print(
            f"{class_name:10s} | "
            f"acc={acc_class:.3f} | "
            f"confidence={conf_class:.3f} | "
            f"n={mask.sum()}"
        )


if __name__ == "__main__":
    main()