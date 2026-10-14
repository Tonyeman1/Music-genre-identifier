from types import SimpleNamespace

import numpy as np
import pytest
import torch

from music_genre.audio import SAMPLE_RATE
from music_genre.classifier import GenreClassifier, ModelUnavailable, rank_genres

LABELS = {0: "Pop music", 1: "Rock music", 2: "Heavy metal", 3: "Jazz",
          4: "Classical music", 5: "Blues", 6: "Music"}


def test_genre_ranking_uses_strongest_related_tag():
    predictions = rank_genres(np.array([.2, .4, .8, .6, .1, .3, .9]), LABELS)
    assert [item.genre for item in predictions] == ["Rock", "Jazz", "Blues", "Pop", "Classical"]
    assert predictions[0].confidence == .8
    assert predictions[0].evidence == "Heavy metal"


def test_incompatible_model_labels_fail():
    with pytest.raises(ModelUnavailable, match="expected music genres"):
        rank_genres(np.array([.7]), {0: "LABEL_0"})


def test_nonfinite_model_scores_fail():
    with pytest.raises(ModelUnavailable, match="invalid scores"):
        rank_genres(np.array([float("nan")] * 7), LABELS)


def test_predict_averages_windows_and_sigmoid_scores():
    predictor = GenreClassifier()
    predictor._labels = LABELS
    logits = [torch.tensor([[0., 2., 1., -1., -2., -1., 3.]]),
              torch.tensor([[0., 0., 1., -1., -2., -1., 3.]])]
    calls = []

    def extractor(window, **kwargs):
        calls.append((len(window), kwargs))
        return {"input_values": torch.zeros(1, 1024, 128)}

    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits.pop(0))

    predictor._extractor = extractor
    predictor._model = Model()
    result = predictor.predict(np.ones(20 * SAMPLE_RATE, dtype=np.float32))
    # Max after averaging: heavy metal sigmoid(1) beats averaged rock score.
    assert result["genre"] == "Rock"
    assert result["confidence"] == pytest.approx(.731059, abs=1e-6)
    assert result["predictions"][0]["evidence"] == "Heavy metal"
    assert result["windows"] == 2
    assert result["sampled_seconds"] == 20
    assert len(calls) == 2
    assert all(call[1]["sampling_rate"] == SAMPLE_RATE for call in calls)
    assert result["warnings"] == []


def test_load_failure_has_actionable_error(monkeypatch):
    from transformers import AutoFeatureExtractor

    def fail(*args, **kwargs):
        raise OSError("No network")

    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained", fail)
    predictor = GenreClassifier()
    with pytest.raises(ModelUnavailable, match="download-model"):
        predictor.load()
    assert not predictor.loaded


def test_real_ast_feature_extraction_and_forward_pass():
    """Exercise real dependencies with tiny untrained weights; not an accuracy test."""
    from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

    torch.set_num_threads(2)
    config = ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                       intermediate_size=64, max_length=32, num_labels=len(LABELS),
                       id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    predictor = GenreClassifier()
    predictor._extractor = ASTFeatureExtractor(max_length=32)
    predictor._model = ASTForAudioClassification(config).eval()
    predictor._labels = LABELS
    waveform = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE).astype(np.float32)
    result = predictor.predict(waveform)
    assert result["windows"] == 1
    assert len(result["predictions"]) == 5
    assert all(0 <= item["confidence"] <= 1 for item in result["predictions"])
    assert "Short clips" in result["warnings"][0]
