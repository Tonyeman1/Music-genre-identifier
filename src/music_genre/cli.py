"""Command-line setup, serving, and prediction."""

import argparse
import json
from pathlib import Path
import sys

from .audio import AudioError, MAX_BYTES, decode_audio
from .classifier import GenreClassifier, ModelUnavailable


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify music genres in WAV and MP3 files.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model", help="Download and validate the pretrained model.")
    serve = commands.add_parser("serve", help="Start the web application.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    predict = commands.add_parser("predict", help="Print a prediction as JSON.")
    predict.add_argument("file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "serve":
            import uvicorn
            uvicorn.run("music_genre.app:app", host=args.host, port=args.port, reload=args.reload)
        elif args.command == "download-model":
            GenreClassifier().load()
            print("Model downloaded and validated. Ready for offline inference.")
        else:
            with args.file.open("rb") as source:
                data = source.read(MAX_BYTES + 1)
            waveform = decode_audio(data, args.file.name)
            result = GenreClassifier().predict(waveform)
            print(json.dumps(result, indent=2))
    except (AudioError, ModelUnavailable, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
