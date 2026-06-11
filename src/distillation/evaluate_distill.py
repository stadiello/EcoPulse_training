import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

CSV = "data/ecopulse_esc50.csv"
MODEL = "models/ecopulse_esc50_distilled.keras"

LABELS = ["bird", "human", "motor", "nature"]
label_to_id = {l: i for i, l in enumerate(LABELS)}

SAMPLE_RATE = 16000
DURATION = 2
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 40
FRAME_LENGTH = 512
FRAME_STEP = 320

# --------------------
# Feature extraction
# --------------------

def load_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)

    audio = tf.squeeze(audio, axis=-1)

    audio = audio[:SAMPLES]

    audio = tf.pad(
        audio,
        [[0, tf.maximum(0, SAMPLES - tf.shape(audio)[0])]]
    )

    return audio


def log_mel(audio):
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH
    )

    spectrogram = tf.abs(stft)

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80,
        upper_edge_hertz=7600
    )

    mel = tf.matmul(tf.square(spectrogram), mel_matrix)

    logmel = tf.math.log(mel + 1e-6)

    mean = tf.reduce_mean(logmel)
    std = tf.math.reduce_std(logmel)

    logmel = (logmel - mean) / (std + 1e-6)

    return logmel[..., tf.newaxis]


def tf_process(path, label):
    audio = load_audio(path)
    features = log_mel(audio)

    label_id = tf.py_function(
        lambda x: label_to_id[x.numpy().decode()],
        [label],
        tf.int64
    )

    label_id.set_shape([])

    return features, label_id


# --------------------
# Dataset
# --------------------

df = pd.read_csv(CSV)

val_df = df[df["split"] == "val"]

ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["path"].values,
        val_df["eco_label"].values
    )
)

ds = ds.map(tf_process)
ds = ds.batch(32)

# --------------------
# Model
# --------------------

model = tf.keras.models.load_model(MODEL)

# --------------------
# Prediction
# --------------------

y_true = []
y_pred = []

for x, y in ds:

    logits = model.predict(x, verbose=0)
    pred = tf.nn.softmax(logits).numpy()

    y_true.extend(y.numpy())

    y_pred.extend(
        np.argmax(pred, axis=1)
    )

# --------------------
# Accuracy
# --------------------

accuracy = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print(f"\nValidation accuracy : {accuracy:.4f}\n")

# --------------------
# Classification report
# --------------------

print(
    classification_report(
        y_true,
        y_pred,
        target_names=LABELS
    )
)

# --------------------
# Confusion matrix
# --------------------

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion matrix:\n")
print(cm)

# --------------------
# Plot
# --------------------

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("EcoPulse Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nMatrice sauvegardée dans :\n"
    "models/confusion_matrix_distilled.png"
)