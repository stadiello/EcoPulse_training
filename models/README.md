---
language:
  - en

license: mit

tags:
  - audio
  - bioacoustics
  - environmental-sound-classification
  - tinyml
  - tensorflow
  - tflite
  - raspberry-pi-pico
  - edge-ai
  - ecopulse

datasets:
  - ESC-50
  - FSD50K
  - InsectSet32
  - HuggingFace-bird-local

library_name: tensorflow

pipeline_tag: audio-classification

thumbnail: https://huggingface.co/front/assets/huggingface_logo.svg
---
## Resume
Tiny audio classifier designed for environmental monitoring on ultra-low-power microcontrollers such as the Raspberry Pi Pico 2 W. The model classifies environmental sounds into six coarse classes and is optimized for TinyML deployment.

## Specifications
Task: bioacoustic coarse classification

Classes: bird, human, motor, rain_wind, insect, animal

Input: log-mel spectrogram 149x40x1

Model size: 73.6 KB INT8

Target: Raspberry Pi Pico 2 W / TinyML

Test accuracy: 87.44 %

Datasets: ESC-50, FSD50K, InsectSet32, filtered bird dataset

Limitations: modèle expérimental, pas encore validé terrain

## Usage

```py
import tensorflow as tf

model = tf.keras.models.load_model("ecopulse_cnn.keras")
```