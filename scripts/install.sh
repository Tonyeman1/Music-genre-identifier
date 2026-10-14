#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"
if ! command -v ffmpeg >/dev/null; then
  echo 'FFmpeg is required. Install it using your OS package manager (e.g. sudo apt install ffmpeg).' >&2
  exit 1
fi
venv_path="${MUSIC_GENRE_VENV:-/workspace/.venvs/music-genre-identifier}"
python3 -m venv "$venv_path"
"$venv_path/bin/python" -m pip install -r requirements-dev.txt
"$venv_path/bin/python" -m pip install -e . --no-deps
echo "Installed. Activate with: source $venv_path/bin/activate"
echo "Then run: HF_HUB_DISABLE_XET=1 music-genre download-model && music-genre serve"
