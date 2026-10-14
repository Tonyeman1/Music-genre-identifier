"""Bounded WAV/MP3 decoding through FFmpeg, without storing user audio."""

from pathlib import Path
import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 120
WINDOW_SECONDS = 10
MAX_WINDOWS = 3
ALLOWED_EXTENSIONS = {".wav", ".mp3"}


class AudioError(ValueError):
    """An unsupported, invalid, or unusable audio input."""


def check_filename(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise AudioError("Please choose a WAV or MP3 file.")


def decode_audio(data: bytes, filename: str) -> np.ndarray:
    check_filename(filename)
    if not data:
        raise AudioError("The audio file is empty.")
    if len(data) > MAX_BYTES:
        raise AudioError("The audio file exceeds the 25 MB limit.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required. Install it and restart the application.")
    # Force the expected demuxer; disable network protocols and video decoding.
    file_format = "mp3" if Path(filename).suffix.lower() == ".mp3" else "wav"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-protocol_whitelist", "pipe", "-f", file_format, "-i", "pipe:0",
        "-t", str(MAX_SECONDS), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio decoding took too long. Try a shorter file.") from exc
    if result.returncode or not result.stdout:
        raise AudioError("The file could not be decoded. Use a valid WAV or MP3 file.")
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(waveform) < SAMPLE_RATE:
        raise AudioError("Please provide at least one second of audio.")
    if not np.isfinite(waveform).all():
        raise AudioError("The audio contains invalid sample values.")
    if float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) < 1e-5:
        raise AudioError("The audio is silent or too quiet to analyze.")
    return waveform


def audio_windows(waveform: np.ndarray) -> list[np.ndarray]:
    """Sample up to three evenly spaced ten-second windows of decoded audio."""
    width = SAMPLE_RATE * WINDOW_SECONDS
    if len(waveform) <= width:
        return [waveform]
    count = min(MAX_WINDOWS, int(np.ceil(len(waveform) / width)))
    starts = np.linspace(0, len(waveform) - width, count, dtype=int)
    return [waveform[start:start + width] for start in starts]
