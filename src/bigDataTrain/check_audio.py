import tensorflow as tf
import pandas as pd

from config import MERGED_CSV

df = pd.read_csv(MERGED_CSV)

bad = []

for i, row in df.iterrows():
    path = row["filepath"]

    try:
        audio = tf.io.read_file(path)
        tf.audio.decode_wav(audio)
    except Exception as e:
        bad.append((path, row["source"], row["label"], str(e).split("\n")[0]))
        print("[BAD]", path, row["source"], row["label"], str(e).split("\n")[0])

print(f"\n[RESULT] fichiers invalides : {len(bad)}")