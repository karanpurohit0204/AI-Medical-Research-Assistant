"""
Phase 3 — Query Route
POST /api/query/ask  — full pipeline: STT text → RAG → LLM → structured answer

This is the main endpoint the doctor interacts with.
Accepts text (post-STT) and returns a full medical answer with citations.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.services.rag.rag_pipeline import RAGPipeline
from backend.services.llm.llm_service import LLMService
from backend.core.logging import logger

router = APIRouter()

_pipeline: RAGPipeline | None = None
_llm: LLMService | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


# ── Models ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Doctor's medical question (text)")
    max_papers: int = Field(default=5, ge=1, le=20)
    top_k: int = Field(default=3, ge=1, le=10)


class CitedPaper(BaseModel):
    pmid: str
    title: str
    authors: str = ""
    journal: str = ""
    year: str = ""
    url: str = ""
    relevance_score: float = 0.0


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitedPaper]
    citations_text: str
    confidence_score: float
    confidence_reasoning: str
    llm_model: str
    processing_time_ms: int


# ── POST /api/query/ask ───────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Full pipeline: query → PubMed → ChromaDB → Groq LLM → answer + citations.

    This is what gets called after the doctor speaks and Whisper transcribes.
    In Phase 4 we'll chain STT → this endpoint automatically.
    """
    logger.info(f"Full query pipeline: '{request.query[:60]}'")

    try:
        start = time.time()

        # Step 1: RAG — fetch and retrieve relevant papers
        rag_result = get_pipeline().retrieve(
            query=request.query,
            max_pubmed_results=request.max_papers,
            top_k=request.top_k,
        )

        if not rag_result.has_results():
            raise HTTPException(
                status_code=404,
                detail="No relevant papers found for this query. Try rephrasing.",
            )

        # Step 2: LLM — generate answer using paper context
        llm_response = get_llm().generate_answer(
            query=request.query,
            context=rag_result.context,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        return AskResponse(
            query=request.query,
            answer=llm_response.answer,
            citations=[CitedPaper(**p) for p in rag_result.papers],
            citations_text=llm_response.citations_text,
            confidence_score=llm_response.confidence_score,
            confidence_reasoning=llm_response.confidence_reasoning,
            llm_model=llm_response.model,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))