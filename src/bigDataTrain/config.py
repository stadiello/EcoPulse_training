from pathlib import Path

# =========================
# Paths à adapter à ta machine
# =========================

DATA_ROOT = Path("data")

ESC50_ROOT = DATA_ROOT / "ESC-50"
FSD50K_ROOT = Path("/Volumes/Extreme SSD/data/4060432")

OUTPUT_ROOT = Path("prepared_dataset")
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

MERGED_CSV = Path("data/ecopulse_esc50_fsd50k.csv")

# =========================
# Taxonomie EcoPulse
# =========================
# Pas de classe "other" à l'entraînement.
# Les sons hors taxonomie sont filtrés par seuil de confiance en inférence.

CLASSES = [
    "bird",
    "human",
    "motor",
    "rain_wind",
    "insect",
    "animal",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

UNKNOWN_THRESHOLD = 0.60

# Audio
SAMPLE_RATE = 16000
CLIP_DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# Spectrogramme
N_FFT = 512
HOP_LENGTH = 320
N_MELS = 40

# Entraînement
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42
