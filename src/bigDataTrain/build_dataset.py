"""
Fusion ESC-50 + FSD50K + InsectSet32 + Bird Hugging Face
vers un CSV unique EcoPulse.

Sortie :
    data/ecopulse_esc50_fsd50k.csv

Colonnes :
    filepath, source, original_label, label, label_id, split

Règles :
- Pas de classe ``other``.
- Les fichiers hors taxonomie sont exclus.
- Les splits officiels d'InsectSet32 et du dataset oiseaux sont conservés.
- Les autres sources sont découpées de manière stratifiée.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

INSECT_ROOT = DATA_ROOT / "7072196"
INSECT_AUDIO_DIR = INSECT_ROOT / "audio"
# INSECT_AUDIO_DIR = INSECT_ROOT / "audio_16k"
INSECT_METADATA_FILES = (
    INSECT_ROOT / "Orthoptera.csv",
    INSECT_ROOT / "Cicadidae.csv",
)

BIRD_DATASET_NAME = "greenarcade/wav2vec2-vd-bird-sound-classification-dataset"
BIRD_CACHE_DIR = DATA_ROOT / "bird_hf_cache"
BIRD_CONVERTED_DIR = DATA_ROOT / "bird_audio_16k"
MAX_BIRD_FILES = 3000

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
}

ALLOWED_BIRD_LABELS = {
    "Common_kingfisher",
    "Golden_oriole",
    "Great_egret",
    "Grey_Heron",
    "Little_egret",
    "Eurasian_spoonbill",
    "Garganey",
    "Northern_shoveler",
    "Glossy_ibis",
}
EXCLUDED_BIRD_LABELS = {"background", "silence"}

def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "filepath",
            "source",
            "original_label",
            "label",
            "label_id",
            "split",
        ]
    )


def _normalise_split(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return SPLIT_ALIASES.get(str(value).strip().lower())


def _resolve_audio_file(audio_dir: Path, filename: str) -> Path | None:
    """Retrouve un fichier audio dans ``audio_dir``, même récursivement."""
    filename = str(filename)
    direct_path = audio_dir / filename
    if direct_path.exists():
        return direct_path.resolve()

    matches = list(audio_dir.rglob(filename))
    if matches:
        return matches[0].resolve()

    # Certains jeux de données référencent un nom sans extension.
    stem = Path(filename).stem
    for extension in AUDIO_EXTENSIONS:
        matches = list(audio_dir.rglob(f"{stem}{extension}"))
        if matches:
            return matches[0].resolve()

    return None


def build_esc50_dataframe() -> pd.DataFrame:
    meta_path = ESC50_ROOT / "meta" / "esc50.csv"
    audio_dir = ESC50_ROOT / "audio"

    if not meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata introuvable : {meta_path}")

    df = pd.read_csv(meta_path)
    rows: list[dict[str, Any]] = []
    excluded = 0

    for _, row in df.iterrows():
        filepath = audio_dir / row["filename"]
        if not filepath.exists():
            continue

        label = map_esc50_label(row["category"])
        if label is None:
            excluded += 1
            continue

        rows.append(
            {
                "filepath": str(filepath.resolve()),
                "source": "ESC-50",
                "original_label": row["category"],
                "label": label,
                "label_id": CLASS_TO_ID[label],
                "split": None,
            }
        )

    print(f"[INFO] ESC-50 exclus : {excluded}")
    return pd.DataFrame(rows) if rows else _empty_dataframe()


def build_fsd50k_dataframe(split: str) -> pd.DataFrame:
    """Construit FSD50K pour ``split`` égal à ``dev`` ou ``eval``."""
    gt_path = FSD50K_ROOT / "FSD50K.ground_truth" / f"{split}.csv"
    audio_dir = FSD50K_ROOT / f"FSD50K.{split}_audio_16k"

    if audio_dir.exists():
        print(f"[INFO] Utilisation des fichiers convertis 16 kHz : {audio_dir}")
    else:
        print(
            f"[WARN] Aucun dossier 16 kHz trouvé pour '{split}', "
            "utilisation des fichiers d'origine."
        )
        audio_dir = FSD50K_ROOT / f"FSD50K.{split}_audio"

    if not gt_path.exists():
        print(f"[WARN] FSD50K metadata absent : {gt_path}")
        return _empty_dataframe()

    if not audio_dir.exists():
        raise FileNotFoundError(f"Dossier audio FSD50K introuvable : {audio_dir}")

    df = pd.read_csv(gt_path)
    rows: list[dict[str, Any]] = []
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

        rows.append(
            {
                "filepath": str(wav_path.resolve()),
                "source": f"FSD50K-{split}",
                "original_label": labels,
                "label": label,
                "label_id": CLASS_TO_ID[label],
                "split": None,
            }
        )

    print(f"[INFO] FSD50K {split} exclus hors taxonomie : {excluded}")
    print(f"[INFO] FSD50K {split} audio manquant : {missing_audio}")
    return pd.DataFrame(rows) if rows else _empty_dataframe()


def build_insect_dataframe() -> pd.DataFrame:
    """Ajoute Orthoptera et Cicadidae sous la classe EcoPulse ``insect``."""
    if "insect" not in CLASS_TO_ID:
        raise KeyError("La classe 'insect' est absente de CLASS_TO_ID.")

    if not INSECT_AUDIO_DIR.exists():
        print(f"[WARN] Dossier audio insectes absent : {INSECT_AUDIO_DIR}")
        return _empty_dataframe()

    rows: list[dict[str, Any]] = []
    missing_audio = 0

    for metadata_path in INSECT_METADATA_FILES:
        if not metadata_path.exists():
            print(f"[WARN] Metadata insectes absente : {metadata_path}")
            continue

        family = metadata_path.stem
        metadata = pd.read_csv(metadata_path)
        required_columns = {"file_name", "species", "data_set"}
        missing_columns = required_columns - set(metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans {metadata_path}: "
                f"{sorted(missing_columns)}"
            )

        for _, row in metadata.iterrows():
            filepath = _resolve_audio_file(INSECT_AUDIO_DIR, row["file_name"])
            if filepath is None:
                missing_audio += 1
                continue

            rows.append(
                {
                    "filepath": str(filepath),
                    "source": f"InsectSet32-{family}",
                    "original_label": str(row["species"]),
                    "label": "insect",
                    "label_id": CLASS_TO_ID["insect"],
                    "split": _normalise_split(row["data_set"]),
                }
            )

    print(f"[INFO] InsectSet32 ajouté : {len(rows)} fichiers")
    print(f"[INFO] InsectSet32 audio manquant : {missing_audio}")
    return pd.DataFrame(rows) if rows else _empty_dataframe()


def _normalise_bird_split_from_path(filepath: Path) -> str | None:
    """Déduit le split depuis le chemin local du dataset oiseaux."""
    parts = {part.lower() for part in filepath.parts}

    for candidate in ("train", "training"):
        if candidate in parts:
            return "train"

    for candidate in ("validation", "valid", "val", "dev"):
        if candidate in parts:
            return "val"

    for candidate in ("test", "eval", "evaluation"):
        if candidate in parts:
            return "test"

    return None


def _extract_bird_species_from_path(filepath: Path, dataset_root: Path) -> str:
    """Extrait un label lisible depuis l'arborescence locale."""
    try:
        relative = filepath.relative_to(dataset_root)
    except ValueError:
        return filepath.parent.name

    parts = relative.parts
    if len(parts) >= 3 and _normalise_split(parts[0]) is not None:
        return parts[1]

    if len(parts) >= 2:
        return parts[-2]

    return "bird"

def _convert_bird_audio_to_wav16k(src: Path, dataset_root: Path) -> Path | None:
    relative = src.relative_to(dataset_root)
    dst = (BIRD_CONVERTED_DIR / relative).with_suffix(".wav")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst.resolve()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-y",
                "-i", str(src),
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                str(dst),
            ],
            check=True,
        )
        return dst.resolve()
    except subprocess.CalledProcessError:
        return None

def build_bird_dataframe() -> pd.DataFrame:
    """
    Télécharge une fois le dataset Hugging Face en cache local, puis indexe les
    fichiers audio directement avec Path.rglob().

    Cette version évite load_dataset(...)["audio"]["array"], qui décode tous les
    fichiers audio et ralentit énormément la construction du CSV.
    """
    if "bird" not in CLASS_TO_ID:
        raise KeyError("La classe 'bird' est absente de CLASS_TO_ID.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet 'huggingface_hub' est requis pour télécharger le dataset oiseaux : "
            "poetry add huggingface-hub"
        ) from exc

    print(f"[INFO] Vérification/téléchargement local du dataset oiseaux : {BIRD_DATASET_NAME}")
    dataset_root = Path(
        snapshot_download(
            repo_id=BIRD_DATASET_NAME,
            repo_type="dataset",
            local_dir=BIRD_CACHE_DIR,
            local_dir_use_symlinks=False,
        )
    )

    audio_files = sorted(
        path.resolve()
        for extension in AUDIO_EXTENSIONS
        for path in dataset_root.rglob(f"*{extension}")
    )

    if MAX_BIRD_FILES is not None:
        audio_files = audio_files[:MAX_BIRD_FILES]

    rows: list[dict[str, Any]] = []

    failed_conversion = 0

    for filepath in audio_files:
        converted_path = _convert_bird_audio_to_wav16k(filepath, dataset_root)

        if converted_path is None:
            failed_conversion += 1
            continue

        species = _extract_bird_species_from_path(filepath, dataset_root)
        original_label = str(species).strip()

        if original_label in EXCLUDED_BIRD_LABELS:
            continue

        if original_label not in ALLOWED_BIRD_LABELS:
            continue

        rows.append(
            {
                "filepath": str(converted_path),
                "source": "HuggingFace-bird-local",
                "original_label": original_label,
                "label": "bird",
                "label_id": CLASS_TO_ID["bird"],
                "split": None,
            }
        )

    print(f"[INFO] Conversions oiseaux échouées : {failed_conversion}")

    print(f"[INFO] Oiseaux indexés localement : {len(rows)} fichiers")
    print(f"[INFO] Limite oiseaux appliquée : {MAX_BIRD_FILES}")
    return pd.DataFrame(rows) if rows else _empty_dataframe()


def balance_dataframe(df: pd.DataFrame, max_per_class: int = 3000) -> pd.DataFrame:
    sampled_parts = []

    for _, class_df in df.groupby("label"):
        n = min(len(class_df), max_per_class)
        sampled_parts.append(class_df.sample(n=n, random_state=SEED))

    if not sampled_parts:
        raise ValueError("Aucune donnée après filtrage/labellisation.")

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def _random_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    class_counts = df["label"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 4:
        raise ValueError(
            "Pas assez d'exemples pour effectuer un split stratifié 70/15/15 :\n"
            f"{class_counts}"
        )

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


def add_train_val_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """Conserve les splits officiels et découpe seulement les lignes sans split."""
    df = df.copy()
    df["split"] = df["split"].apply(_normalise_split)

    predefined = df[df["split"].notna()].copy()
    unsplit = df[df["split"].isna()].copy()
    generated = _random_stratified_split(unsplit)

    return (
        pd.concat([predefined, generated], ignore_index=True)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def main() -> None:
    print("[INFO] Construction ESC-50...")
    esc50_df = build_esc50_dataframe()

    print("[INFO] Construction FSD50K dev...")
    fsd_dev_df = build_fsd50k_dataframe("dev")

    print("[INFO] Construction FSD50K eval...")
    fsd_eval_df = build_fsd50k_dataframe("eval")

    print("[INFO] Construction InsectSet32...")
    insect_df = build_insect_dataframe()

    print("[INFO] Construction dataset oiseaux...")
    bird_df = build_bird_dataframe()

    df = pd.concat(
        [esc50_df, fsd_dev_df, fsd_eval_df, insect_df, bird_df],
        ignore_index=True,
    )
    df = df.drop_duplicates(subset=["filepath"])
    df = df[df["label"].isin(CLASS_TO_ID)].copy()
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
