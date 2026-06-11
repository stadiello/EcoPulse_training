import tensorflow as tf
import pandas as pd
from pathlib import Path

CSV = "data/ecopulse_esc50.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 2
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 40
FRAME_LENGTH = 512
FRAME_STEP = 320

LABELS = ["bird", "human", "motor", "nature"]
NUM_CLASSES = len(LABELS)
label_to_id = {label: i for i, label in enumerate(LABELS)}

df = pd.read_csv(CSV)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

def load_audio(path):
    audio = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # ESC-50 est en 44.1 kHz.
    # Pour un vrai resampling, utiliser tensorflow-io.
    # Ici, on coupe directement après décodage si les fichiers sont déjà convertis en 16 kHz.
    audio = audio[:SAMPLES]
    audio = tf.pad(audio, [[0, tf.maximum(0, SAMPLES - tf.shape(audio)[0])]])

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
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )

    mel = tf.matmul(tf.square(spectrogram), mel_matrix)
    logmel = tf.math.log(mel + 1e-6)

    mean = tf.reduce_mean(logmel)
    std = tf.math.reduce_std(logmel)
    logmel = (logmel - mean) / (std + 1e-6)

    return logmel[..., tf.newaxis]

def process(path, label):
    label = label_to_id[label.numpy().decode("utf-8")]
    return path, label

def tf_process(path, label):
    label_id = tf.py_function(
        func=lambda x: label_to_id[x.numpy().decode("utf-8")],
        inp=[label],
        Tout=tf.int64,
    )
    label_id.set_shape([])

    audio = load_audio(path)
    features = log_mel(audio)

    return features, label_id

def make_ds(dataframe, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["path"].values,
            dataframe["eco_label"].values,
        )
    )

    ds = ds.map(tf_process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512)

    return ds.batch(32).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df)
val_ds = make_ds(val_df, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(99, 40, 1)),

    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/ecopulse_esc50_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
)

model.save("models/ecopulse_esc50_final.keras")