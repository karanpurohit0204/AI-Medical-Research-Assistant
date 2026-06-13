"""
main.py — FastAPI application entry point.
Start with: uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import get_settings
from backend.core.logging import logger, setup_logging
from backend.models.schemas import HealthResponse
from backend.api.routes import stt as stt_router
from backend.api.routes import rag as rag_router
from backend.api.routes import query_route as query_router
from backend.api.routes import voice as voice_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} [{settings.app_env}]")

    logger.info("Pre-loading Whisper model...")
    from backend.services.whisper_service import load_whisper_model
    load_whisper_model()

    yield
    logger.info("Shutting down.")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Voice-first medical research assistant with RAG",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────
app.include_router(stt_router.router,   prefix="/api/stt",   tags=["Phase 1 — STT"])
app.include_router(rag_router.router,   prefix="/api/rag",   tags=["Phase 2 — RAG"])
app.include_router(query_router.router, prefix="/api/query", tags=["Phase 3 — LLM"])
app.include_router(voice_router.router, prefix="/api/voice", tags=["Phase 4 — Voice Pipeline"])


@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"{settings.app_name} is running. Visit /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    from backend.services.whisper_service import _model
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        whisper_model_loaded=_model is not None,
    )