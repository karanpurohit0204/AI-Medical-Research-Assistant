"""
Phase 1 — STT Routes
POST /api/stt/transcribe  — upload audio file, get back transcribed text
GET  /api/stt/models      — list available Whisper model sizes
"""

import time
import tempfile
import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from backend.services.whisper_service import WhisperSTTService
from backend.models.schemas import TranscriptionResponse
from backend.core.logging import logger

router = APIRouter()

_stt_service: WhisperSTTService | None = None


def get_stt_service() -> WhisperSTTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = WhisperSTTService()
    return _stt_service


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file — WAV, MP3, M4A, FLAC"),
):
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    ext = Path(file.filename or "audio.wav").suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # Windows fix: delete=False keeps file on disk until we manually remove it
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()  # must close before Whisper opens it on Windows

        start = time.time()
        result = get_stt_service().transcribe_file(tmp_path)
        elapsed_ms = int((time.time() - start) * 1000)

        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            segments_count=len(result.segments),
            source_file=file.filename,
            processing_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)  # manually delete after done
        except Exception:
            pass


@router.get("/models")
async def list_models():
    from backend.core.config import get_settings
    current = get_settings().whisper_model
    models = [
        {"name": "tiny",   "size_mb": 32,   "speed": "fastest", "accuracy": "low"},
        {"name": "base",   "size_mb": 150,  "speed": "fast",    "accuracy": "good"},
        {"name": "small",  "size_mb": 500,  "speed": "medium",  "accuracy": "better"},
        {"name": "medium", "size_mb": 1500, "speed": "slow",    "accuracy": "great"},
        {"name": "large",  "size_mb": 3000, "speed": "slowest", "accuracy": "best"},
    ]
    for m in models:
        m["active"] = m["name"] == current
    return {"models": models, "current": current}