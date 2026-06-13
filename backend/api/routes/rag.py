"""
Phase 2 — RAG Routes
POST /api/rag/search   — takes a text query, returns ranked papers + context
GET  /api/rag/status   — shows how many papers are stored in ChromaDB
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.services.rag.rag_pipeline import RAGPipeline
from backend.core.logging import logger

router = APIRouter()

# Single pipeline instance — shared across requests
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────

class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Medical question to search for")
    max_pubmed_results: int = Field(default=8, ge=1, le=20)
    top_k: int = Field(default=3, ge=1, le=10)


class PaperResult(BaseModel):
    pmid: str
    title: str
    authors: str = ""
    journal: str = ""
    year: str = ""
    url: str = ""
    abstract_snippet: str = ""
    relevance_score: float = 0.0


class RAGSearchResponse(BaseModel):
    query: str
    papers: List[PaperResult]
    context: str
    total_papers_fetched: int
    total_papers_stored: int
    processing_time_ms: int


# ── POST /api/rag/search ──────────────────────────────────────────────────────

@router.post("/search", response_model=RAGSearchResponse)
async def search_papers(request: RAGSearchRequest):
    """
    Search PubMed + ChromaDB for papers relevant to the query.

    This is the Phase 2 endpoint. In Phase 3, the LLM will call this
    automatically and use the returned context to generate an answer.
    """
    logger.info(f"RAG search request: '{request.query[:60]}'")

    try:
        start = time.time()
        result = get_pipeline().retrieve(
            query=request.query,
            max_pubmed_results=request.max_pubmed_results,
            top_k=request.top_k,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        return RAGSearchResponse(
            query=result.query,
            papers=[PaperResult(**p) for p in result.papers],
            context=result.context,
            total_papers_fetched=result.total_papers_fetched,
            total_papers_stored=result.total_papers_stored,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/rag/status ───────────────────────────────────────────────────────

@router.get("/status")
async def rag_status():
    """Shows how many papers are currently stored in ChromaDB."""
    try:
        pipeline = get_pipeline()
        count = pipeline.chroma.count()
        return {
            "status": "ok",
            "papers_in_db": count,
            "message": f"{count} papers stored in ChromaDB",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))