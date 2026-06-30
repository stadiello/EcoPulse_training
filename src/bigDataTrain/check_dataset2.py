from pathlib import Path
import wave
import contextlib

import pandas as pd

from config import MERGED_CSV, CLASS_TO_ID


REQUIRED_COLUMNS = {
    "filepath", "source", "original_label", "label", "label_id", "split"
}

EXPECTED_SPLITS = {"train", "val", "test"}


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_wav_info(path: Path):
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav:
            sr = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.getnframes()
            sampwidth = wav.getsampwidth()
            duration = frames / sr if sr else 0
            return {
                "sample_rate": sr,
                "channels": channels,
                "duration": duration,
                "sample_width_bytes": sampwidth,
                "ok": True,
            }
    except Exception:
        return {
            "sample_rate": None,
            "channels": None,
            "duration": None,
            "sample_width_bytes": None,
            "ok": False,
        }


def main():
    score = 100
    df = pd.read_csv(MERGED_CSV)

    section("Résumé global")
    print(f"CSV : {MERGED_CSV}")
    print(f"Lignes : {len(df)}")
    print(f"Colonnes : {list(df.columns)}")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"[ERREUR] Colonnes manquantes : {missing_cols}")
        score -= 20
    else:
        print("[OK] Colonnes attendues présentes")

    section("Distribution labels")
    print(df["label"].value_counts())
    print("\nPourcentage :")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    unknown_labels = set(df["label"]) - set(CLASS_TO_ID.keys())
    if unknown_labels:
        print(f"[ERREUR] Labels inconnus : {unknown_labels}")
        score -= 20
    else:
        print("[OK] Labels connus")

    section("Splits")
    print(df["split"].value_counts())
    unknown_splits = set(df["split"]) - EXPECTED_SPLITS
    if unknown_splits:
        print(f"[ERREUR] Splits inconnus : {unknown_splits}")
        score -= 10
    else:
        print("[OK] Splits valides")

    section("Split x label")
    split_label = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(split_label)

    section("Sources")
    print(df["source"].value_counts())

    section("Source x label")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))

    section("Présence fichiers")
    paths = df["filepath"].astype(str).map(Path)
    exists = paths.map(lambda p: p.exists())

    print(f"Présents : {exists.sum()} / {len(df)}")
    print(f"Manquants : {(~exists).sum()}")

    if (~exists).any():
        score -= 20
        print(df.loc[~exists, ["filepath", "source", "label"]].head(30))
    else:
        print("[OK] Aucun fichier manquant")

    section("Doublons")
    duplicates = df[df.duplicated("filepath", keep=False)]
    print(f"Doublons filepath : {len(duplicates)}")

    if len(duplicates):
        score -= 10
        print(duplicates[["filepath", "source", "label", "split"]].head(30))
    else:
        print("[OK] Aucun doublon")

    section("Label IDs")
    expected_ids = df["label"].map(CLASS_TO_ID)
    bad_ids = df["label_id"] != expected_ids
    print(f"Label IDs incorrects : {bad_ids.sum()}")

    if bad_ids.any():
        score -= 10
        print(df.loc[bad_ids, ["label", "label_id"]].head(30))
    else:
        print("[OK] Label IDs cohérents")

    section("Équilibre par split")
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        counts = part["label"].value_counts()
        print(f"\n[{split}]")
        print(counts)

        if len(counts) > 0:
            ratio = counts.max() / counts.min()
            print(f"Ratio max/min : {ratio:.2f}")

            if ratio > 10:
                print("[WARN] Déséquilibre très fort")
                score -= 8
            elif ratio > 5:
                print("[WARN] Déséquilibre marqué")
                score -= 4
            else:
                print("[OK] Équilibre acceptable")

    section("Oiseaux")
    bird_df = df[df["label"] == "bird"]
    if len(bird_df):
        print(bird_df["original_label"].value_counts().head(50))
    else:
        print("[WARN] Aucun oiseau")
        score -= 15

    section("Insectes")
    insect_df = df[df["label"] == "insect"]
    if len(insect_df):
        print(insect_df["source"].value_counts())
        print("\nEspèces insectes :")
        print(insect_df["original_label"].value_counts().head(50))
    else:
        print("[WARN] Aucun insecte")
        score -= 15

    if len(insect_df) < 300:
        print("[WARN] Classe insect faible")
        score -= 5

    section("Statistiques audio WAV")
    sample = df.copy()
    sample["path_obj"] = sample["filepath"].astype(str).map(Path)

    infos = []
    for _, row in sample.iterrows():
        info = safe_wav_info(row["path_obj"])
        info["label"] = row["label"]
        info["source"] = row["source"]
        infos.append(info)

    info_df = pd.DataFrame(infos)

    print("\nSample rates :")
    print(info_df["sample_rate"].value_counts(dropna=False))

    print("\nCanaux :")
    print(info_df["channels"].value_counts(dropna=False))

    print("\nSample width bytes :")
    print(info_df["sample_width_bytes"].value_counts(dropna=False))

    print("\nDurées par label :")
    print(
        info_df.groupby("label")["duration"]
        .agg(["count", "min", "mean", "max"])
        .round(3)
    )

    unreadable = (~info_df["ok"]).sum()
    print(f"\nFichiers WAV illisibles par wave : {unreadable}")
    bad_audio = sample.loc[~info_df["ok"], ["filepath", "source", "label", "original_label"]]
    print(bad_audio)

    if unreadable:
        score -= 15

    section("Score qualité dataset")
    score = max(score, 0)
    print(f"Score : {score}/100")

    if score >= 90:
        print("[OK] Dataset très sain")
    elif score >= 75:
        print("[OK] Dataset exploitable, quelques points à surveiller")
    elif score >= 50:
        print("[WARN] Dataset utilisable mais fragile")
    else:
        print("[ERREUR] Dataset à corriger avant entraînement")


if __name__ == "__main__":
    main()