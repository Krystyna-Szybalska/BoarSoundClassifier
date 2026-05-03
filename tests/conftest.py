"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def synthetic_boar_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a 4-second synthetic boar-like WAV (low-freq growl approximation).

    Sine sweep 80-300 Hz + bandpass noise. Used by tests that need a 4 s WAV
    but do not require real boar audio (real audio is gitignored).
    """
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sweep = 0.5 * np.sin(2 * np.pi * (80 + (300 - 80) * t / duration) * t)
    noise = 0.1 * np.random.RandomState(0).randn(len(t))
    audio = (sweep + noise).astype(np.float32)

    path = tmp_path_factory.mktemp("audio") / "synthetic_boar.wav"
    sf.write(path, audio, sr)
    return path
