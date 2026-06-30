from pathlib import Path
import pandas as pd

from config import MERGED_CSV, CLASS_TO_ID


REQUIRED_COLUMNS = {
    "filepath",
    "source",
    "original_label",
    "label",
    "label_id",
    "split",
}

EXPECTED_SPLITS = {"train", "val", "test"}


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    csv_path = Path(MERGED_CSV)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    print_section("Résumé global")
    print(f"CSV : {csv_path}")
    print(f"Nombre total de lignes : {len(df)}")
    print(f"Colonnes : {list(df.columns)}")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        print(f"[ERREUR] Colonnes manquantes : {missing_columns}")
    else:
        print("[OK] Colonnes attendues présentes")

    print_section("Labels")
    print(df["label"].value_counts())
    print("\nRépartition en % :")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    unknown_labels = set(df["label"]) - set(CLASS_TO_ID.keys())
    if unknown_labels:
        print(f"[ERREUR] Labels inconnus : {unknown_labels}")
    else:
        print("[OK] Tous les labels sont connus")

    print_section("Splits")
    print(df["split"].value_counts())

    unknown_splits = set(df["split"]) - EXPECTED_SPLITS
    if unknown_splits:
        print(f"[ERREUR] Splits inconnus : {unknown_splits}")
    else:
        print("[OK] Splits train/val/test valides")

    print_section("Répartition split x label")
    split_label = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(split_label)

    print_section("Sources")
    print(df["source"].value_counts())

    print_section("Répartition source x label")
    source_label = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    print(source_label)

    print_section("Fichiers audio")
    paths = df["filepath"].astype(str).map(Path)

    exists = paths.map(lambda p: p.exists())
    print(f"Fichiers présents : {exists.sum()} / {len(df)}")
    print(f"Fichiers manquants : {(~exists).sum()}")

    if (~exists).any():
        print("\nExemples fichiers manquants :")
        print(df.loc[~exists, ["filepath", "source", "label"]].head(20))

    print_section("Doublons")
    duplicate_paths = df[df.duplicated("filepath", keep=False)]
    print(f"Doublons filepath : {len(duplicate_paths)}")

    if len(duplicate_paths) > 0:
        print(duplicate_paths[["filepath", "source", "label", "split"]].head(20))

    print_section("Label IDs")
    expected_ids = df["label"].map(CLASS_TO_ID)
    bad_ids = df["label_id"] != expected_ids

    print(f"Label IDs incorrects : {bad_ids.sum()}")

    if bad_ids.any():
        print(df.loc[bad_ids, ["label", "label_id"]].head(20))

    print_section("Alerte équilibre")
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        counts = part["label"].value_counts()

        print(f"\n[{split}]")
        print(counts)

        if len(counts) > 0:
            ratio = counts.max() / counts.min()
            print(f"Ratio max/min : {ratio:.2f}")

            if ratio > 5:
                print("[WARN] Déséquilibre fort")
            else:
                print("[OK] Équilibre acceptable")

    print_section("Résumé final")
    print("[OK] Vérification terminée")


if __name__ == "__main__":
    main()