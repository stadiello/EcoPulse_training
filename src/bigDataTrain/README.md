# EcoPulse — Fusion ESC-50 + FSD50K + entraînement CNN TinyML

## Objectif

Créer un dataset audio pour EcoPulse avec 7 classes :

```text
bird
human
motor
rain_wind
insect
animal
other
```

Puis entraîner un petit CNN exportable en TFLite int8 pour Raspberry Pi Pico 2 / TFLite Micro.

## Structure attendue

```text
project/
  config.py
  label_mapping.py
  build_dataset.py
  audio_features.py
  train_model.py
  resample_audio.py

  data/
    ESC-50/
      audio/
      meta/
        esc50.csv

    FSD50K/
      FSD50K.dev_audio/
      FSD50K.eval_audio/
      FSD50K.ground_truth/
        dev.csv
        eval.csv
```

## Installation

```bash
pip install -r requirements.txt
```

Ou avec Poetry :

```bash
poetry add tensorflow pandas numpy scikit-learn
```

## Étape 1 — fusionner et labelliser

```bash
python build_dataset.py
```

Sortie :

```text
prepared_dataset/ecopulse_esc50_fsd50k.csv
```

## Étape 2 — entraîner

```bash
python train_model.py
```

Sorties :

```text
models/ecopulse_cnn.keras
models/ecopulse_cnn_best.keras
models/ecopulse_cnn_int8.tflite
```

## Note importante

Le script suppose que les fichiers WAV sont déjà en mono 16 kHz.

Si ce n'est pas le cas, utiliser `resample_audio.py` avec ffmpeg :

```bash
brew install ffmpeg
python resample_audio.py data/FSD50K/FSD50K.dev_audio data/FSD50K/FSD50K.dev_audio_16k
```

Ensuite, adapter `config.py` pour pointer vers les dossiers 16 kHz.

## Recommandation

Pour Pico 2, vise :

```text
TFLite int8 < 150 ko
RAM inférence < 200 ko
input log-mel compact
```

La prochaine étape sera d'ajouter une évaluation par matrice de confusion pour vérifier les confusions :

```text
bird vs insect
human vs animal
rain_wind vs other
motor vs other
```
