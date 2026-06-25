"""
Prétraitement audio pour entraînement CNN compact.

Entrée :
- wav quelconque

Sortie :
- log-mel spectrogram normalisé
- shape approximative : [time, n_mels, 1]
"""

import tensorflow as tf

from config import (
    SAMPLE_RATE,
    N_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)


def decode_wav(filepath: tf.Tensor) -> tf.Tensor:
    audio_bytes = tf.io.read_file(filepath)
    audio, sr = tf.audio.decode_wav(audio_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # Hypothèse : dataset déjà en 16 kHz ou converti en amont.
    # Pour une version production, convertir avec librosa/ffmpeg avant.
    audio = audio[:N_SAMPLES]
    padding = N_SAMPLES - tf.shape(audio)[0]
    audio = tf.cond(
        padding > 0,
        lambda: tf.pad(audio, [[0, padding]]),
        lambda: audio,
    )

    return audio


def waveform_to_logmel(audio: tf.Tensor) -> tf.Tensor:
    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
    )

    spectrogram = tf.abs(stft)

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )

    mel = tf.matmul(tf.square(spectrogram), mel_matrix)
    logmel = tf.math.log(mel + 1e-6)

    mean = tf.reduce_mean(logmel)
    std = tf.math.reduce_std(logmel)
    logmel = (logmel - mean) / (std + 1e-6)

    logmel = tf.expand_dims(logmel, axis=-1)
    return logmel


def load_example(filepath: tf.Tensor, label_id: tf.Tensor):
    audio = decode_wav(filepath)
    features = waveform_to_logmel(audio)
    return features, label_id
