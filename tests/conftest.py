import io
import subprocess
import wave

import numpy as np
import pytest

from music_genre.audio import SAMPLE_RATE


@pytest.fixture
def wav_bytes():
    output = io.BytesIO()
    times = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    samples = (0.3 * np.sin(2 * np.pi * 440 * times) * 32767).astype("<i2")
    with wave.open(output, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(samples.tobytes())
    return output.getvalue()


@pytest.fixture
def mp3_bytes(wav_bytes):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "mp3", "pipe:1"],
        input=wav_bytes, capture_output=True, check=True,
    )
    return result.stdout
