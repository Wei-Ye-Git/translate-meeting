import numpy as np


def highpass(audio, samplerate=16000, cutoff_hz=80):
    samples = np.asarray(audio, dtype=np.float32)
    if len(samples) < 2 or samplerate <= 0 or cutoff_hz <= 0:
        return samples
    radius = float(np.exp(-2.0 * np.pi * cutoff_hz / samplerate))
    filtered = np.empty_like(samples)
    filtered[0] = 0.0
    for index in range(1, len(samples)):
        filtered[index] = radius * (filtered[index - 1] + samples[index] - samples[index - 1])
    return filtered
