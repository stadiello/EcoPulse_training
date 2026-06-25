mkdir -p data/ESC-50/audio_16k

for f in data/ESC-50/audio/*.wav; do
  ffmpeg -y -i "$f" -ac 1 -ar 16000 "data/ESC-50/audio_16k/$(basename "$f")"
done

mkdir -p /Volumes/Extreme\ SSD/data/4060432/FSD50K.dev_audio_16k

for f in /Volumes/Extreme\ SSD/data/4060432/FSD50K.dev_audio/*.wav; do
  ffmpeg -y -i "$f" -ac 1 -ar 16000 "/Volumes/Extreme SSD/data/4060432/FSD50K.dev_audio_16k/$(basename "$f")"
done