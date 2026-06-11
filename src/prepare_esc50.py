from pathlib import Path
import pandas as pd

from model_data import LABEL_MAP

ROOT = Path("data/ESC-50")
META = ROOT / "meta/esc50.csv"
# AUDIO = ROOT / "audio"
# Fichier converti en 16 kHz pour éviter d'avoir à faire du resampling à la volée pendant l'entraînement. 
# Voir convertionData.bash à la racine
AUDIO = ROOT / "audio_16k" 
df = pd.read_csv(META)
df = df[df["category"].isin(LABEL_MAP.keys())].copy()

df["eco_label"] = df["category"].map(LABEL_MAP)
df["path"] = df["filename"].apply(lambda x: str(AUDIO / x))

# ESC-50 fournit 5 folds propres
# On garde fold 5 pour validation
df["split"] = df["fold"].apply(lambda f: "val" if f == 5 else "train")

df = df[["path", "category", "eco_label", "fold", "split"]]
df.to_csv("data/ecopulse_esc50.csv", index=False)

print(df["eco_label"].value_counts())
print(df.groupby(["split", "eco_label"]).size())