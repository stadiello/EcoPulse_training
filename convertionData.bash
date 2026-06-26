#!/usr/bin/env bash
set -euo pipefail

progress_bar() {
    local current="$1"
    local total="$2"
    local width=40

    if [ "$total" -eq 0 ]; then
        return
    fi

    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r["
    printf "%0.s#" $(seq 1 "$filled")
    printf "%0.s-" $(seq 1 "$empty")
    printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

convert_dir() {
    local input_dir="$1"
    local output_dir="$2"

    if [ ! -d "$input_dir" ]; then
        echo "[WARN] Dossier introuvable, conversion ignorée : $input_dir"
        return
    fi

    mkdir -p "$output_dir"

    local total
    total=$(find "$input_dir" -maxdepth 1 -type f -name "*.wav" | wc -l | tr -d ' ')

    if [ "$total" -eq 0 ]; then
        echo "[WARN] Aucun fichier WAV trouvé dans : $input_dir"
        return
    fi

    echo "[INFO] Conversion : $input_dir -> $output_dir"
    echo "[INFO] Nombre de fichiers : $total"

    local current=0

    for f in "$input_dir"/*.wav; do
        [ -e "$f" ] || continue

        current=$((current + 1))

        ffmpeg -loglevel error -y \
            -i "$f" \
            -ac 1 \
            -ar 16000 \
            "$output_dir/$(basename "$f")"

        progress_bar "$current" "$total"
    done

    echo
    echo "[OK] Conversion terminée : $output_dir"
}

convert_dir "data/ESC-50/audio" \
            "data/ESC-50/audio_16k"

convert_dir "/Volumes/Extreme SSD/data/4060432/FSD50K.dev_audio" \
            "/Volumes/Extreme SSD/data/4060432/FSD50K.dev_audio_16k"

convert_dir "/Volumes/Extreme SSD/data/4060432/FSD50K.eval_audio" \
            "/Volumes/Extreme SSD/data/4060432/FSD50K.eval_audio_16k"

# convert_dir "data/7072196/audio" \
#             "data/7072196/audio_16k"