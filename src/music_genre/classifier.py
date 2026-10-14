"""Use AST's AudioSet music tags to rank broad music genres."""

from dataclasses import dataclass
import os
import threading

import numpy as np

from .audio import SAMPLE_RATE, audio_windows

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
GENRE_TAGS = {
    "Pop": ("Pop music",),
    "Hip-hop": ("Hip hop music",),
    "Rock": ("Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
             "Rock and roll", "Psychedelic rock"),
    "Jazz": ("Jazz", "Swing music"),
    "Classical": ("Classical music", "Opera"),
    "Blues": ("Blues",),
    "Country": ("Country", "Bluegrass"),
    "Electronic": ("Electronic music", "House music", "Techno", "Dubstep",
                   "Drum and bass", "Electronica", "Electronic dance music",
                   "Ambient music", "Trance music"),
    "Reggae": ("Reggae", "Ska"),
    "Folk": ("Folk music",),
    "Funk": ("Funk",),
    "Soul / R&B": ("Soul music", "Rhythm and blues"),
    "Disco": ("Disco",),
    "Latin": ("Music of Latin America", "Salsa music", "Flamenco"),
    "Gospel": ("Gospel music", "Christian music"),
    "World / traditional": ("Traditional music", "Music of Africa", "Afrobeat",
                            "Music of Asia", "Carnatic music", "Music of Bollywood",
                            "Middle Eastern music"),
}


class ModelUnavailable(RuntimeError):
    """The model cannot currently be loaded or used."""


@dataclass(frozen=True)
class Prediction:
    genre: str
    confidence: float
    evidence: str


def rank_genres(scores: np.ndarray, labels: dict[int, str]) -> list[Prediction]:
    """Max over each genre's related tags, after averaging windows."""
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ModelUnavailable("The model returned invalid scores.")
    index = {label.casefold(): i for i, label in labels.items()}
    predictions = []
    for genre, tags in GENRE_TAGS.items():
        candidates = [(float(scores[index[tag.casefold()]]), tag)
                      for tag in tags if tag.casefold() in index]
        if candidates:
            confidence, evidence = max(candidates)
            predictions.append(Prediction(genre, round(confidence, 6), evidence))
    if len(predictions) < 5:
        raise ModelUnavailable("The model's labels do not contain the expected music genres.")
    return sorted(predictions, key=lambda p: p.confidence, reverse=True)


class GenreClassifier:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._extractor = None
        self._labels: dict[int, str] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                offline = os.getenv("MUSIC_GENRE_OFFLINE", "0") == "1"
                extractor = AutoFeatureExtractor.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, local_files_only=offline,
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, local_files_only=offline, use_safetensors=True,
                ).eval()
                labels = {int(key): value for key, value in model.config.id2label.items()}
                rank_genres(np.zeros(model.config.num_labels), labels)
                if int(extractor.sampling_rate) != SAMPLE_RATE:
                    raise ValueError("Unexpected model sample rate")
            except Exception as exc:
                raise ModelUnavailable(
                    "The pretrained model is unavailable. Run 'music-genre download-model' "
                    "with access to Hugging Face, then retry. See the README for offline setup."
                ) from exc
            self._extractor, self._labels, self._model = extractor, labels, model

    def predict(self, waveform: np.ndarray) -> dict:
        import torch
        with self._lock:
            self.load()
            windows = audio_windows(waveform)
            per_window = []
            # One window at a time bounds memory on CPU-only machines.
            with torch.inference_mode():
                for window in windows:
                    inputs = self._extractor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                    logits = self._model(**inputs).logits
                    per_window.append(torch.sigmoid(logits)[0].cpu().numpy())
            scores = np.mean(per_window, axis=0)
            ranked = rank_genres(scores, self._labels)
            music_index = next((i for i, tag in self._labels.items() if tag == "Music"), None)
            warnings = []
            if ranked[0].confidence < 0.15:
                warnings.append("Low genre confidence. Try a longer, clearer music excerpt.")
            if music_index is not None and float(scores[music_index]) < 0.15:
                warnings.append("This recording may contain little or no music.")
            if len(waveform) / SAMPLE_RATE < 5:
                warnings.append("Short clips are less reliable; use at least 10 seconds when possible.")
            return {
                "genre": ranked[0].genre,
                "confidence": ranked[0].confidence,
                "predictions": [vars(item) for item in ranked[:5]],
                "analyzed_seconds": round(len(waveform) / SAMPLE_RATE, 2),
                "sampled_seconds": round(sum(len(w) for w in windows) / SAMPLE_RATE, 2),
                "windows": len(windows),
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "warnings": warnings,
            }
