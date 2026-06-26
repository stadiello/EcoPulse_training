"""
Fusion ESC-50 + FSD50K vers un CSV unique EcoPulse.

Sortie :
data/ecopulse_esc50_fsd50k.csv

Colonnes :
filepath, source, original_label, label, label_id, split

Note :
- Pas de classe "other".
- Les fichiers hors taxonomie retournent None et sont exclus.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ESC50_ROOT,
    FSD50K_ROOT,
    MERGED_CSV,
    CLASS_TO_ID,
    SEED,
)
from label_mapping import map_esc50_label, map_fsd50k_labels


def build_esc50_dataframe() -> pd.DataFrame:
    meta_path = ESC50_ROOT / "meta" / "esc50.csv"
    audio_dir = ESC50_ROOT / "audio"

    if not meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata introuvable : {meta_path}")

    df = pd.read_csv(meta_path)

    rows = []
    excluded = 0

    for _, row in df.iterrows():
        filepath = audio_dir / row["filename"]
        if not filepath.exists():
            continue

        label = map_esc50_label(row["category"])
        if label is None:
            excluded += 1
            continue

        rows.append({
            "filepath": str(filepath),
            "source": "ESC-50",
            "original_label": row["category"],
            "label": label,
            "label_id": CLASS_TO_ID[label],
        })

    print(f"[INFO] ESC-50 exclus : {excluded}")
    return pd.DataFrame(rows)


def build_fsd50k_dataframe(split: str) -> pd.DataFrame:
    """
    split: 'dev' ou 'eval'
    """
    gt_path = FSD50K_ROOT / "FSD50K.ground_truth" / f"{split}.csv"
    audio_dir = FSD50K_ROOT / f"FSD50K.{split}_audio_16k"
    
    if audio_dir.exists():
        print(f"[INFO] Utilisation des fichiers convertis 16 kHz : {audio_dir}")
    else:
        print(f"[WARN] Aucun dossier 16 kHz trouvé pour '{split}', utilisation des fichiers d'origine.")
        audio_dir = FSD50K_ROOT / f"FSD50K.{split}_audio"

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Dossier audio FSD50K introuvable : {audio_dir}"
        )

    if not gt_path.exists():
        print(f"[WARN] FSD50K metadata absent : {gt_path}")
        return pd.DataFrame()

    if not audio_dir.exists():
        raise FileNotFoundError(f"Dossier audio FSD50K introuvable : {audio_dir}")

    df = pd.read_csv(gt_path)

    rows = []
    excluded = 0
    missing_audio = 0

    for _, row in df.iterrows():
        fname = str(row["fname"])
        labels = str(row["labels"])

        wav_path = audio_dir / f"{fname}.wav"
        if not wav_path.exists():
            wav_path = audio_dir / fname

        if not wav_path.exists():
            missing_audio += 1
            continue

        label = map_fsd50k_labels(labels)
        if label is None:
            excluded += 1
            continue

        rows.append({
            "filepath": str(wav_path),
            "source": f"FSD50K-{split}",
            "original_label": labels,
            "label": label,
            "label_id": CLASS_TO_ID[label],
        })

    print(f"[INFO] FSD50K {split} exclus hors taxonomie : {excluded}")
    print(f"[INFO] FSD50K {split} audio manquant : {missing_audio}")

    return pd.DataFrame(rows)


def balance_dataframe(df: pd.DataFrame, max_per_class: int = 3000) -> pd.DataFrame:
    sampled_parts = []

    for label_name, class_df in df.groupby("label"):
        n = min(len(class_df), max_per_class)
        sampled_parts.append(class_df.sample(n=n, random_state=SEED))

    if not sampled_parts:
        raise ValueError("Aucune donnée après filtrage/labellisation.")

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def add_train_val_test_split(df: pd.DataFrame) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main():
    print("[INFO] Construction ESC-50...")
    esc50_df = build_esc50_dataframe()
    print(esc50_df["label"].value_counts())

    print("[INFO] Construction FSD50K dev...")
    fsd_dev_df = build_fsd50k_dataframe("dev")

    print("[INFO] Construction FSD50K eval...")
    fsd_eval_df = build_fsd50k_dataframe("eval")

    df = pd.concat([esc50_df, fsd_dev_df, fsd_eval_df], ignore_index=True)
    df = df.drop_duplicates(subset=["filepath"])
    df = df[df["label"].isin(CLASS_TO_ID.keys())].copy()

    # Sécurité : recalcul des ids après suppression de "other"
    df["label_id"] = df["label"].map(CLASS_TO_ID)

    print("[INFO] Distribution brute :")
    print(df["label"].value_counts())

    df = balance_dataframe(df, max_per_class=3000)

    print("[INFO] Distribution équilibrée :")
    print(df["label"].value_counts())

    df = add_train_val_test_split(df)

    MERGED_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(MERGED_CSV, index=False)

    print(f"[OK] Dataset fusionné : {MERGED_CSV}")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
