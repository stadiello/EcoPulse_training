from pathlib import Path
import pandas as pd
import tensorflow as tf

CSV = "data/ecopulse_esc50_yamnet_soft.csv"

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

ALPHA = 0.5
TEMPERATURE = 3.0


def load_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    audio = audio[:SAMPLES]
    audio = tf.pad(audio, [[0, tf.maximum(0, SAMPLES - tf.shape(audio)[0])]])

    return audio


def augment_audio(audio):
    noise = tf.random.normal(tf.shape(audio), stddev=0.005)
    gain = tf.random.uniform([], 0.8, 1.2)
    audio = audio * gain + noise
    audio = tf.clip_by_value(audio, -1.0, 1.0)
    return audio


def log_mel(audio):
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
    )

    spectrogram = tf.abs(stft)

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )

    mel = tf.matmul(tf.square(spectrogram), mel_matrix)
    logmel = tf.math.log(mel + 1e-6)

    mean = tf.reduce_mean(logmel)
    std = tf.math.reduce_std(logmel)
    logmel = (logmel - mean) / (std + 1e-6)

    return logmel[..., tf.newaxis]


def make_model():
    return tf.keras.Sequential([
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
        tf.keras.layers.Dense(NUM_CLASSES),
    ])


def distillation_loss(y_true_hard, y_true_soft, student_logits):
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_hard,
        student_logits,
        from_logits=True,
    )

    teacher_soft = tf.nn.softmax(y_true_soft / TEMPERATURE)
    student_soft = tf.nn.softmax(student_logits / TEMPERATURE)

    kl = tf.keras.losses.KLDivergence()(
        teacher_soft,
        student_soft,
    )

    return ALPHA * hard_loss + (1.0 - ALPHA) * kl * (TEMPERATURE ** 2)


class DistilledModel(tf.keras.Model):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc]

    def train_step(self, data):
        x, y = data
        y_hard = y["hard"]
        y_soft = y["soft"]

        with tf.GradientTape() as tape:
            logits = self.student(x, training=True)
            loss = distillation_loss(y_hard, y_soft, logits)

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )

        self.loss_tracker.update_state(loss)
        self.acc.update_state(y_hard, tf.nn.softmax(logits))

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc.result(),
        }

    def test_step(self, data):
        x, y = data
        y_hard = y["hard"]

        logits = self.student(x, training=False)

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_hard,
            logits,
            from_logits=True,
        )

        self.loss_tracker.update_state(loss)
        self.acc.update_state(y_hard, tf.nn.softmax(logits))

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc.result(),
        }


def tf_process(path, hard_label, soft_label, training):
    audio = load_audio(path)

    if training:
        audio = augment_audio(audio)

    features = log_mel(audio)

    hard_label = tf.py_function(
        lambda x: label_to_id[x.numpy().decode("utf-8")],
        [hard_label],
        tf.int64,
    )
    hard_label.set_shape([])

    soft_label = tf.cast(soft_label, tf.float32)
    soft_label.set_shape([NUM_CLASSES])

    return features, {
        "hard": hard_label,
        "soft": soft_label,
    }


def make_ds(dataframe, training=True):
    paths = dataframe["path"].values
    hard = dataframe["eco_label"].values
    soft = dataframe[[f"soft_{l}" for l in LABELS]].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, hard, soft))

    ds = ds.map(
        lambda p, h, s: tf_process(p, h, s, training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.shuffle(512)

    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def main():
    df = pd.read_csv(CSV)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_ds = make_ds(train_df, training=True)
    val_ds = make_ds(val_df, training=False)

    student = make_model()
    model = DistilledModel(student)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
    )

    student.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            mode="max",
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        callbacks=callbacks,
    )

    student.save("models/ecopulse_esc50_distilled.keras")
    print("Modèle sauvegardé : models/ecopulse_esc50_distilled.keras")


if __name__ == "__main__":
    main()