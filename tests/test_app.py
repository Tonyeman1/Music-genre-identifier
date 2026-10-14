import pytest
from fastapi.testclient import TestClient

from music_genre.app import create_app
from music_genre.audio import MAX_BYTES, SAMPLE_RATE
from music_genre.classifier import ModelUnavailable


class TestPredictor:
    __test__ = False
    loaded = True

    def predict(self, waveform):
        assert len(waveform) >= SAMPLE_RATE
        return {"genre": "Jazz", "confidence": .8,
                "predictions": [{"genre": "Jazz", "confidence": .8, "evidence": "Jazz"}],
                "analyzed_seconds": len(waveform) / SAMPLE_RATE,
                "sampled_seconds": len(waveform) / SAMPLE_RATE,
                "windows": 1, "model": "test-double", "warnings": []}


@pytest.fixture
def client():
    with TestClient(create_app(TestPredictor())) as instance:
        yield instance


def test_index_and_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="upload"' in response.text
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_wav_upload(client, wav_bytes):
    response = client.post("/api/predict", files={"file": ("song.wav", wav_bytes, "audio/wav")})
    assert response.status_code == 200
    assert response.json()["genre"] == "Jazz"
    assert response.json()["analyzed_seconds"] == 2


def test_mp3_upload(client, mp3_bytes):
    response = client.post("/api/predict", files={"file": ("song.mp3", mp3_bytes, "audio/mpeg")})
    assert response.status_code == 200
    assert response.json()["genre"] == "Jazz"


@pytest.mark.parametrize("name,content", [("bad.txt", b"hi"), ("bad.wav", b"no audio"), ("empty.mp3", b"")])
def test_invalid_file_returns_400(client, name, content):
    response = client.post("/api/predict", files={"file": (name, content)})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)


def test_missing_file_returns_422(client):
    assert client.post("/api/predict").status_code == 422


def test_oversized_file_returns_413(client):
    response = client.post("/api/predict", files={"file": ("large.wav", b"x" * (MAX_BYTES + 1))})
    assert response.status_code == 413


def test_unavailable_model_returns_503(wav_bytes):
    class Unavailable(TestPredictor):
        def predict(self, waveform):
            raise ModelUnavailable("Download the model first.")

    with TestClient(create_app(Unavailable())) as client:
        response = client.post("/api/predict", files={"file": ("song.wav", wav_bytes)})
    assert response.status_code == 503
    assert "Download" in response.json()["detail"]


def test_failed_prediction_does_not_lock_out_subsequent_requests(wav_bytes):
    class FailsOnce(TestPredictor):
        calls = 0

        def predict(self, waveform):
            self.calls += 1
            if self.calls == 1:
                raise ModelUnavailable("Temporarily unavailable.")
            return super().predict(waveform)

    with TestClient(create_app(FailsOnce())) as client:
        first = client.post("/api/predict", files={"file": ("song.wav", wav_bytes)})
        second = client.post("/api/predict", files={"file": ("song.wav", wav_bytes)})
    assert first.status_code == 503
    assert second.status_code == 200
