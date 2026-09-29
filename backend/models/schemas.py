from typing import Optional

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str
    language: str = "en"
    segments_count: int = 0
    source_file: Optional[str] = None
    processing_time_ms: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = ""
    version: str = "0.1.0"
    whisper_model_loaded: bool = False
