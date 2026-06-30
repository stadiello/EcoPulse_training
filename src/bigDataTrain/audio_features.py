"""
Prétraitement audio pour entraînement CNN compact.

Entrée :
- wav quelconque

Sortie :
- log-mel spectrogram normalisé
- shape approximative : [time, n_mels, 1]
"""

import tensorflow as tf
import librosa
import numpy as np

from config import (
    SAMPLE_RATE,
    N_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)


def _load_audio_numpy(filepath: bytes) -> np.ndarray:
    """
    Charge un fichier audio avec librosa au lieu de tf.audio.decode_wav.

    Avantage : supporte les WAV PCM 24 bits, WAV extensible, FLAC, etc.
    Sortie : mono float32, resamplé à SAMPLE_RATE, longueur N_SAMPLES.
    """
    path = filepath.decode("utf-8")

    audio, sr = librosa.load(path, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

    audio = audio.astype(np.float32)

    # Normalisation douce pour éviter les amplitudes aberrantes.
    max_abs = np.max(np.abs(audio)) if audio.size else 0.0
    if max_abs > 0:
        audio = audio / max_abs

    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))

    return audio.astype(np.float32)


def decode_wav(filepath: tf.Tensor) -> tf.Tensor:
    audio = tf.numpy_function(
        func=_load_audio_numpy,
        inp=[filepath],
        Tout=tf.float32,
    )
    audio.set_shape([N_SAMPLES])
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
