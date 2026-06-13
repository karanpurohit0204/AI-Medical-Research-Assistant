import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import whisper

from backend.core.config import get_settings
from backend.core.logging import logger


_model: Optional[whisper.Whisper] = None


def load_whisper_model() -> whisper.Whisper:
    global _model
    if _model is None:
        settings = get_settings()
        model_name = settings.whisper_model
        logger.info(f"Loading Whisper model: '{model_name}' ...")
        _model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' ready.")
    return _model


class TranscriptionResult:
    def __init__(self, text: str, language: str = "en", segments: list = None, source_file: str = ""):
        self.text = text
        self.language = language
        self.segments = segments or []
        self.source_file = source_file

    def __repr__(self):
        return f"TranscriptionResult(lang={self.language!r}, text={self.text[:60]!r})"

    def to_dict(self):
        return {
            "text": self.text,
            "language": self.language,
            "segments_count": len(self.segments),
            "source_file": self.source_file,
        }


class WhisperSTTService:

    def __init__(self):
        self.settings = get_settings()
        self.model = load_whisper_model()

    def transcribe_file(self, audio_path: str | Path) -> TranscriptionResult:
        audio_path = Path(audio_path).resolve()  # get absolute path

        logger.debug(f"Transcribing full path: {audio_path}")
        logger.debug(f"File exists: {audio_path.exists()}")
        logger.debug(f"File size: {audio_path.stat().st_size if audio_path.exists() else 'N/A'} bytes")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            language=None,
            task="transcribe",
            fp16=False,
            verbose=False,
        )

        transcription = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            segments=result.get("segments", []),
            source_file=str(audio_path),
        )
        logger.info(f"Transcribed ({transcription.language}): '{transcription.text[:80]}'")
        return transcription

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        audio = audio_array.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        try:
            import soundfile as sf
            sf.write(tmp_path, audio, sample_rate)
            tmp.close()
            return self.transcribe_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass