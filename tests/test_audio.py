import io
import wave

import numpy as np
import pytest

from music_genre.audio import AudioError, MAX_BYTES, SAMPLE_RATE, audio_windows, decode_audio


def test_decode_wav(wav_bytes):
    audio = decode_audio(wav_bytes, "song.WAV")
    assert len(audio) == 2 * SAMPLE_RATE
    assert audio.dtype == np.float32
    assert 0.19 < float(np.sqrt(np.mean(audio ** 2))) < 0.23


def test_decode_mp3(mp3_bytes):
    audio = decode_audio(mp3_bytes, "song.mp3")
    assert 1.9 < len(audio) / SAMPLE_RATE < 2.3
    assert float(np.max(np.abs(audio))) > 0.2


def test_stereo_resampling():
    stream = io.BytesIO()
    rate = 44100
    times = np.arange(rate * 2) / rate
    mono = (np.sin(2 * np.pi * 440 * times) * 10000).astype("<i2")
    stereo = np.column_stack([mono, mono]).astype("<i2")
    with wave.open(stream, "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(stereo.tobytes())
    audio = decode_audio(stream.getvalue(), "stereo.wav")
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE * 2


@pytest.mark.parametrize("data,name,message", [
    (b"", "song.wav", "empty"),
    (b"garbage", "song.wav", "could not be decoded"),
    (b"garbage", "song.mp3", "could not be decoded"),
    (b"garbage", "song.exe", "WAV or MP3"),
])
def test_invalid_inputs(data, name, message):
    with pytest.raises(AudioError, match=message):
        decode_audio(data, name)


def test_size_limit():
    with pytest.raises(AudioError, match="25 MB"):
        decode_audio(b"x" * (MAX_BYTES + 1), "song.wav")


@pytest.mark.parametrize("seconds,sample,message", [(2, 0, "silent"), (0.25, 1000, "one second")])
def test_unusable_recordings(seconds, sample, message):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(np.full(int(SAMPLE_RATE * seconds), sample, dtype="<i2").tobytes())
    with pytest.raises(AudioError, match=message):
        decode_audio(stream.getvalue(), "recording.wav")


def test_evenly_spaced_windows():
    waveform = np.arange(120 * SAMPLE_RATE, dtype=np.float32)
    windows = audio_windows(waveform)
    assert len(windows) == 3
    assert all(len(item) == 10 * SAMPLE_RATE for item in windows)
    assert [window[0] for window in windows] == [0, 55 * SAMPLE_RATE, 110 * SAMPLE_RATE]
    assert windows[-1][-1] == waveform[-1]


def test_short_clip_window_is_unpadded():
    waveform = np.ones(SAMPLE_RATE * 2, dtype=np.float32)
    windows = audio_windows(waveform)
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0], waveform)


def test_duration_limit():
    # A valid 121-second WAV must only decode the first 120 seconds.
    stream = io.BytesIO()
    with wave.open(stream, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(np.full(121 * SAMPLE_RATE, 1000, dtype="<i2").tobytes())
    assert len(decode_audio(stream.getvalue(), "long.wav")) == 120 * SAMPLE_RATE
