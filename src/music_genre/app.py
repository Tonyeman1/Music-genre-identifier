"""FastAPI upload API and a small browser interface."""

from pathlib import Path
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .audio import AudioError, MAX_BYTES, check_filename, decode_audio
from .classifier import GenreClassifier, ModelUnavailable


def create_app(classifier=None) -> FastAPI:
    application = FastAPI(title="Music Genre Identifier", version="0.1.0")
    predictor = classifier if classifier is not None else GenreClassifier()
    analysis_lock = threading.Lock()

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/health")
    def health():
        return {"status": "ok", "model_loaded": predictor.loaded}

    def analyze(data: bytes, filename: str):
        if not analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Another file is being analyzed. Try again shortly.")
        try:
            waveform = decode_audio(data, filename)
            return predictor.predict(waveform)
        except AudioError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Audio processing is unavailable. Check the server logs and FFmpeg installation.") from exc
        finally:
            analysis_lock.release()

    @application.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            filename = file.filename or ""
            try:
                check_filename(filename)
            except AudioError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            data = bytearray()
            while chunk := await file.read(64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="The audio file exceeds the 25 MB limit.")
            return await run_in_threadpool(analyze, bytes(data), filename)
        finally:
            await file.close()

    return application


app = create_app()
