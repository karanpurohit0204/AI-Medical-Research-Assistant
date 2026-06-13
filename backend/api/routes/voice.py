"""
Phase 4 — Voice Route
POST /api/voice/ask  — full end-to-end pipeline:
  1. Upload audio file (WAV/MP3)
  2. Whisper STT → text
  3. RAG → relevant PubMed papers
  4. Groq LLM → structured answer
  5. gTTS/ElevenLabs → audio response
  6. Returns answer text + citations + audio file URL

This is the single endpoint the doctor uses.
"""

import os
import time
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.services.whisper_service import WhisperSTTService
from backend.services.rag.rag_pipeline import RAGPipeline
from backend.services.llm.llm_service import LLMService
from backend.services.tts.tts_service import TTSService
from backend.core.logging import logger

router = APIRouter()

# Service singletons
_stt: WhisperSTTService | None = None
_rag: RAGPipeline | None = None
_llm: LLMService | None = None
_tts: TTSService | None = None


def get_stt(): global _stt; _stt = _stt or WhisperSTTService(); return _stt
def get_rag(): global _rag; _rag = _rag or RAGPipeline(); return _rag
def get_llm(): global _llm; _llm = _llm or LLMService(); return _llm
def get_tts(): global _tts; _tts = _tts or TTSService(); return _tts


# ── Response model ─────────────────────────────────────────────────────────

class CitedPaper(BaseModel):
    pmid: str
    title: str
    authors: str = ""
    journal: str = ""
    year: str = ""
    url: str = ""
    relevance_score: float = 0.0


class VoiceAskResponse(BaseModel):
    transcribed_query: str
    answer: str
    citations: List[CitedPaper]
    citations_text: str
    confidence_score: float
    confidence_reasoning: str
    llm_model: str
    audio_filename: Optional[str] = None   # download via GET /api/voice/audio/{filename}
    processing_time_ms: int
    stages_ms: dict                        # breakdown: stt, rag, llm, tts


# ── POST /api/voice/ask ─────────────────────────────────────────────────────

@router.post("/ask", response_model=VoiceAskResponse)
async def voice_ask(
    file: UploadFile = File(..., description="Doctor's voice question (WAV, MP3, M4A)"),
    max_papers: int = 5,
    top_k: int = 3,
):
    """
    Full voice pipeline in one call:
    Audio → STT → RAG → LLM → TTS → Response

    Upload a WAV/MP3 recording of your medical question.
    Get back a spoken answer + citations + confidence score.
    """
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'",
        )

    # Save upload to temp file (Windows-safe)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    timings = {}

    try:
        tmp.write(await file.read())
        tmp.close()

        total_start = time.time()

        # ── Stage 1: STT ───────────────────────────────────────────────
        t0 = time.time()
        stt_result = get_stt().transcribe_file(tmp_path)
        timings["stt_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"STT: '{stt_result.text[:60]}'")

        query = stt_result.text
        if not query.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio — please speak clearly.")

        # ── Stage 2: RAG ───────────────────────────────────────────────
        t0 = time.time()
        rag_result = get_rag().retrieve(query=query, max_pubmed_results=max_papers, top_k=top_k)
        timings["rag_ms"] = int((time.time() - t0) * 1000)

        if not rag_result.has_results():
            raise HTTPException(status_code=404, detail="No relevant papers found.")

        # ── Stage 3: LLM ───────────────────────────────────────────────
        t0 = time.time()
        llm_response = get_llm().generate_answer(query=query, context=rag_result.context)
        timings["llm_ms"] = int((time.time() - t0) * 1000)

        # ── Stage 4: TTS ───────────────────────────────────────────────
        t0 = time.time()
        # Only speak the answer section, not the citations
        tts_text = llm_response.answer[:800]   # cap at 800 chars for TTS
        audio_path = get_tts().speak(tts_text)
        timings["tts_ms"] = int((time.time() - t0) * 1000)

        total_ms = int((time.time() - total_start) * 1000)
        audio_filename = Path(audio_path).name

        logger.info(
            f"Voice pipeline complete | "
            f"stt={timings['stt_ms']}ms | rag={timings['rag_ms']}ms | "
            f"llm={timings['llm_ms']}ms | tts={timings['tts_ms']}ms | "
            f"total={total_ms}ms"
        )

        return VoiceAskResponse(
            transcribed_query=query,
            answer=llm_response.answer,
            citations=[CitedPaper(**p) for p in rag_result.papers],
            citations_text=llm_response.citations_text,
            confidence_score=llm_response.confidence_score,
            confidence_reasoning=llm_response.confidence_reasoning,
            llm_model=llm_response.model,
            audio_filename=audio_filename,
            processing_time_ms=total_ms,
            stages_ms=timings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── GET /api/voice/audio/{filename} ────────────────────────────────────────

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Download the generated TTS audio file by filename.
    Filename comes from the audio_filename field in the /ask response.
    """
    from backend.core.config import get_settings
    settings = get_settings()
    audio_path = Path(settings.temp_audio_dir) / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found or expired.")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename,
    )


# ── GET /api/voice/pipeline-status ─────────────────────────────────────────

@router.get("/pipeline-status")
async def pipeline_status():
    """Shows which services are loaded and ready."""
    from backend.services.whisper_service import _model as whisper_model
    return {
        "whisper_loaded": whisper_model is not None,
        "tts_engine": get_tts().engine,
        "status": "ready",
    }