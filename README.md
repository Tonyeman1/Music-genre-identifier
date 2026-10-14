# Music Genre Identifier

A Python application that accepts **WAV or MP3 audio** and returns the most likely music genre, the top five matches, and confidence scores. Use the browser interface, JSON API, or command line.

## How it works

The app uses the pretrained [MIT Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), trained on AudioSet. It ranks the model's music genre tags into 16 broad categories: pop, hip-hop, rock, jazz, classical, blues, country, electronic, reggae, folk, funk, soul/R&B, disco, Latin, gospel, and world/traditional.

FFmpeg decodes audio to mono at 16 kHz. The app samples up to three evenly spaced 10-second excerpts from the first 120 seconds, averages each AudioSet tag's sigmoid score across excerpts, and uses the strongest related tag for each genre. It does not require training your own model or an API key. Inference runs locally on CPU.

These are estimated **tag confidence scores, not calibrated genre probabilities**; scores do not sum to 100%. This is a general audio tagging model, not a genre-specific benchmark classifier. Mixed genres, speech, unfamiliar styles, short clips, and noisy recordings can produce unreliable predictions. Low-confidence results include warnings. A quiet or silent recording is rejected. A 10–30 second music excerpt works best.

## Install

Requirements: Python 3.10–3.13, FFmpeg, Internet access for the initial model download, and a few GB of free disk space/RAM. The pinned CPU PyTorch packages support Linux/Windows x86-64; macOS/other architectures need platform-compatible PyTorch packages.

On Ubuntu/Debian, install FFmpeg if needed:

```bash
sudo apt update
sudo apt install ffmpeg
```

Create a virtual environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
HF_HUB_DISABLE_XET=1 music-genre download-model
```

On Windows, activate with `.venv\Scripts\activate` and install FFmpeg separately. On macOS/ARM, install compatible `torch==2.6.0` and `torchaudio==2.6.0` builds using [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/), then run `pip install -e .` and install `pytest==8.3.5 httpx==0.28.1` for tests. The `+cpu` installation requirements target x86-64 cloud machines.

For this Codex cloud checkout, `bash scripts/install.sh` uses `/workspace/.venvs/music-genre-identifier` by default. Override `MUSIC_GENRE_VENV` to choose another location.

The model is pinned to revision `f826b80d28226b62986cc218e5cec390b1096902` and downloaded once to Hugging Face's normal cache using safetensors weights. Initial download requires access to `huggingface.co` and its download destinations, including `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.huggingface.co`, `cas-bridge.xethub.hf.co`, and `us.aws.cdn.hf.co`. `HF_HUB_DISABLE_XET=1` selects Hugging Face's standard HTTP transfer, which works with cloud HTTPS proxies when those destinations are allowed; on Windows, set it as an environment variable before running the command. No Hugging Face token is required for this public model. TLS verification remains enabled. `HF_HOME` can select a cache location; `MUSIC_GENRE_OFFLINE=1` enforces cached-only operation once `download-model` succeeds. An unavailable model causes a clear error; the app never substitutes fake predictions. The pretrained model has its own BSD-3-Clause license; see its model card.

## Run

```bash
music-genre serve
```

Open `http://127.0.0.1:8000` in your browser on your own machine. Upload a WAV or MP3 file (maximum 25 MB, at least one second). The first prediction loads the model into memory. `music-genre serve --reload` enables development reloads. `--host` and `--port` customize binding. Keep one worker to avoid loading duplicate models.

The cloud onboarding UI does not provide a localhost preview; use the API locally to validate it.

CLI prediction:

```bash
music-genre predict /path/to/song.mp3
```

API prediction:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/predict \
  -F 'file=@/path/to/song.wav'
```

Successful responses contain `genre`, `confidence`, `predictions` (each with `genre`, `confidence`, and the detected AudioSet `evidence` tag), `analyzed_seconds`, `sampled_seconds`, `windows`, `model`, `model_revision`, and `warnings`. `analyzed_seconds` describes the decoded portion, capped at 120 seconds; `sampled_seconds` is the amount supplied to the model. Overlapping windows can occur for files just over 10 seconds.

`GET /health` reports server liveness and whether the model is loaded; liveness alone does not validate predictions. `GET /docs` exposes the API reference. Invalid uploads return 400, oversized audio payloads return 413, concurrent analysis returns 429, and unavailable dependencies/model return 503. The service is intended for local development. Public deployment needs authentication, rate limiting, and a reverse proxy with a total request-body limit (multipart parsing happens before the application checks file size).

Uploaded audio is not saved by the application, sent to a model inference API, or used for training. FastAPI can temporarily spool large multipart uploads to disk and removes them when closed. Prediction processing stays on the server that runs the application.

## Test

```bash
pytest -q
```

The suite exercises real FFmpeg decoding of WAV and MP3, invalid/silent/short uploads, size limits, window selection, genre ranking, model aggregation, and API error handling. It also runs real AST feature extraction and inference with a small **untrained** model to check dependency compatibility. API unit tests use a controlled test double. These tests do not download pretrained weights or establish real-world genre accuracy. To validate the actual pretrained workflow, run `music-genre download-model` and then predict a known music recording. Both need to succeed before claiming end-to-end readiness.

## Project layout

```text
src/music_genre/
  audio.py          Audio decoding, validation, and excerpt selection
  classifier.py     Pretrained AST model and genre ranking
  app.py            FastAPI application
  cli.py            Command-line interface
  static/index.html Browser upload interface
tests/              Automated tests
scripts/install.sh  Repeatable cloud installation
```
