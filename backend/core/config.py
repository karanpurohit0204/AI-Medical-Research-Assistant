from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Medical RAG Assistant"
    app_env: str = "development"
    # Avoid the generic DEBUG variable, which is commonly set by host tooling
    # to non-boolean values such as "release".
    debug: bool = Field(default=True, validation_alias=AliasChoices("APP_DEBUG"))
    port: int = 8000

    # Whisper STT
    whisper_model: str = "base"

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Mistral LLM
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    # ElevenLabs TTS
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # PubMed
    ncbi_api_key: str = ""
    ncbi_email: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "medical_papers"

    # LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "medical-rag"

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_record_seconds: int = 10
    temp_audio_dir: str = "./data/temp_audio"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
