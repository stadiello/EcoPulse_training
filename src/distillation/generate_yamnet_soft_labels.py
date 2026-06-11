from pathlib import Path
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

CSV = "data/ecopulse_esc50.csv"
OUT = "data/ecopulse_esc50_yamnet_soft.csv"

SAMPLE_RATE = 16000
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

ECO_LABELS = ["bird", "human", "motor", "nature"]

YAMNET_TO_ECO = {
    "bird": [
        "Bird",
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
        "Crow",
        "Chicken, rooster",
    ],
    "human": [
        "Speech",
        "Conversation",
        "Laughter",
        "Cough",
        "Sneeze",
        "Clapping",
        "Breathing",
        "Baby cry, infant cry",
    ],
    "motor": [
        "Engine",
        "Vehicle",
        "Motor vehicle (road)",
        "Car",
        "Truck",
        "Motorcycle",
        "Aircraft",
        "Helicopter",
        "Chainsaw",
        "Siren",
        "Train",
    ],
    "nature": [
        "Rain",
        "Wind",
        "Thunderstorm",
        "Water",
        "Waves, surf",
        "Insect",
        "Cricket",
    ],
}


def load_class_names():
    class_map_path = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    )

    names = []
    with open(class_map_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"])

    return names


def load_audio_16k(path: str):
    audio = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav


def build_indices(class_names):
    indices = {}

    for eco_label, yamnet_names in YAMNET_TO_ECO.items():
        ids = []
        for name in yamnet_names:
            if name in class_names:
                ids.append(class_names.index(name))
            else:
                print(f"[WARN] Classe YAMNet non trouvée : {name}")

        indices[eco_label] = ids

    return indices


def yamnet_to_ecopulse(scores, indices):
    """
    scores: shape [frames, 521]
    """
    clip_scores = tf.reduce_mean(scores, axis=0).numpy()

    eco_scores = []

    for eco_label in ECO_LABELS:
        ids = indices[eco_label]

        if len(ids) == 0:
            eco_scores.append(0.0)
        else:
            # max = si une sous-classe est forte, la classe EcoPulse est forte
            eco_scores.append(float(np.max(clip_scores[ids])))

    eco_scores = np.array(eco_scores, dtype=np.float32)

    if eco_scores.sum() <= 1e-8:
        eco_scores = np.ones(len(ECO_LABELS), dtype=np.float32) / len(ECO_LABELS)
    else:
        eco_scores = eco_scores / eco_scores.sum()

    return eco_scores


def main():
    df = pd.read_csv(CSV)

    print("Chargement YAMNet...")
    yamnet = hub.load(YAMNET_URL)

    class_names = load_class_names()
    indices = build_indices(class_names)

    soft_labels = []

    for i, row in df.iterrows():
        path = row["path"]
        wav = load_audio_16k(path)

        scores, embeddings, spectrogram = yamnet(wav)

        eco_probs = yamnet_to_ecopulse(scores, indices)
        soft_labels.append(eco_probs)

        if i % 50 == 0:
            print(f"{i}/{len(df)}")

    soft = np.vstack(soft_labels)

    for idx, label in enumerate(ECO_LABELS):
        df[f"soft_{label}"] = soft[:, idx]

    df.to_csv(OUT, index=False)

    print(f"\nSoft labels sauvegardés : {OUT}")
    print(df.head())


if __name__ == "__main__":
    main()