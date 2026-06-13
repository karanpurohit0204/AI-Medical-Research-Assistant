from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file into environment variables

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")  # Default voice ID
ncbi_api_key = os.getenv("NCBI_API_KEY")
ncbi_email = os.getenv("NCBI_EMAIL")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

print(mistral_api_key)

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Medical RAG Assistant"
    app_env: str = "development"
    debug: bool = True
    port: int = 8000

    # Whisper STT
    whisper_model: str = "base"

    # Deepgram (optional)
    # deepgram_api_key: str = ""

    # Groq LLM
    groq_api_key: str = groq_api_key
    groq_model: str = "llama3-8b-8192"

    # Mistral LLM
    mistral_api_key: str = mistral_api_key
    mistral_model: str = "mistral-small-latest"

    # ElevenLabs TTS
    elevenlabs_api_key: str = elevenlabs_api_key
    elevenlabs_voice_id: str = elevenlabs_voice_id

    # PubMed
    ncbi_api_key: str = ncbi_api_key
    ncbi_email: str = ncbi_email

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "medical_papers"

    # LangSmith
    langchain_api_key: str = langchain_api_key
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