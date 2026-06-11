import tensorflow as tf
from pathlib import Path

model = tf.keras.models.load_model("models/ecopulse_esc50_best.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

Path("models").mkdir(exist_ok=True)

with open("models/ecopulse_esc50.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite exporté : models/ecopulse_esc50.tflite")
print("Taille KB :", len(tflite_model) / 1024)