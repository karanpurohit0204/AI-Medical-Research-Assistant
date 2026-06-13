from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── STT ──────────────────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    text: str
    language: str = "en"
    segments_count: int = 0
    source_file: Optional[str] = None
    processing_time_ms: Optional[int] = None


# ── Query / RAG (used in Phase 4+) ───────────────────────────────────────────

class MedicalQueryRequest(BaseModel):
    query: str = Field(..., min_length=5)
    max_results: int = Field(default=5, ge=1, le=20)
    language: str = "en"


class PaperCitation(BaseModel):
    pmid: str
    title: str
    authors: List[str] = []
    journal: str = ""
    year: Optional[int] = None
    abstract_snippet: str = ""
    url: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MedicalQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[PaperCitation] = []
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    audio_url: Optional[str] = None
    llm_model: str = ""
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = ""
    version: str = "0.1.0"
    whisper_model_loaded: bool = False