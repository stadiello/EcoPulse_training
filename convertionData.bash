mkdir -p data/ESC-50/audio_16k

for f in data/ESC-50/audio/*.wav; do
  ffmpeg -y -i "$f" -ac 1 -ar 16000 "data/ESC-50/audio_16k/$(basename "$f")"
done