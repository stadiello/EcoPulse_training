"""
Optionnel mais recommandé.

Convertit tous les WAV ESC-50/FSD50K en mono 16 kHz avant entraînement.

Dépendance système :
brew install ffmpeg

Commande :
python resample_audio.py data/FSD50K/FSD50K.dev_audio data/FSD50K/FSD50K.dev_audio_16k
"""

import argparse
import subprocess
from pathlib import Path


def convert_wav(src: Path, dst: Path):
    dst.parent.mkdir(exist_ok=True, parents=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        str(dst),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    wavs = list(args.input_dir.rglob("*.wav"))
    print(f"[INFO] {len(wavs)} fichiers trouvés")

    for i, src in enumerate(wavs, 1):
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        convert_wav(src, dst)

        if i % 100 == 0:
            print(f"[INFO] {i}/{len(wavs)}")

    print("[OK] Conversion terminée")


if __name__ == "__main__":
    main()
