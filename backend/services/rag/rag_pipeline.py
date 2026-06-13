"""
Phase 2 — RAG Pipeline
Orchestrates the full Retrieve-Augment flow:
  1. Search PubMed for papers matching the query
  2. Embed and store them in ChromaDB
  3. Retrieve the most relevant chunks for the query

This is the single entry point for the RAG layer.
The LLM (Phase 3) will call this to get context before generating an answer.
"""

from typing import List
from dataclasses import dataclass, field

from backend.services.rag.pubmed_service import PubMedService, PubMedPaper
from backend.services.rag.chroma_service import ChromaService
from backend.core.logging import logger


@dataclass
class RetrievalResult:
    """
    Everything retrieved for a single query.
    Passed directly to the LLM in Phase 3.
    """
    query: str
    papers: List[dict] = field(default_factory=list)   # ranked by relevance
    context: str = ""                                   # formatted text for LLM prompt
    total_papers_fetched: int = 0
    total_papers_stored: int = 0

    def has_results(self) -> bool:
        return len(self.papers) > 0


class RAGPipeline:
    """
    Full RAG pipeline: PubMed → ChromaDB → ranked retrieval.

    Usage:
        rag = RAGPipeline()
        result = rag.retrieve("What are the latest treatments for Type 2 Diabetes?")
        print(result.context)   # pass this to the LLM
        print(result.papers)    # use these for citations
    """

    def __init__(self):
        self.pubmed = PubMedService()
        self.chroma = ChromaService()

    def retrieve(self, query: str, max_pubmed_results: int = 8, top_k: int = 3) -> RetrievalResult:
        """
        Full pipeline: fetch from PubMed → store in ChromaDB → retrieve top-k.

        Args:
            query:              The doctor's question (from STT)
            max_pubmed_results: How many papers to fetch from PubMed
            top_k:              How many to return after similarity ranking

        Returns:
            RetrievalResult with ranked papers and formatted context
        """
        logger.info(f"RAG pipeline start | query='{query[:60]}'")

        # Step 1: Fetch papers from PubMed
        papers: List[PubMedPaper] = self.pubmed.search(query, max_results=max_pubmed_results)

        # Step 2: Embed and store in ChromaDB (skips duplicates automatically)
        added = self.chroma.add_papers(papers)

        # Step 3: Retrieve most relevant papers via vector similarity
        ranked = self.chroma.search(query, n_results=top_k)

        # Step 4: Format context string for the LLM prompt
        context = self._build_context(ranked)

        result = RetrievalResult(
            query=query,
            papers=ranked,
            context=context,
            total_papers_fetched=len(papers),
            total_papers_stored=added,
        )

        logger.info(
            f"RAG pipeline done | fetched={len(papers)} "
            f"| stored={added} | retrieved={len(ranked)}"
        )
        return result

    def _build_context(self, papers: List[dict]) -> str:
        """
        Format retrieved papers into a context block for the LLM.
        Each paper is numbered and includes title + abstract snippet.
        """
        if not papers:
            return "No relevant research papers found."

        sections = []
        for i, paper in enumerate(papers, 1):
            section = (
                f"[{i}] {paper.get('title', 'Untitled')}\n"
                f"Authors: {paper.get('authors', 'Unknown')}\n"
                f"Journal: {paper.get('journal', 'Unknown')} ({paper.get('year', 'Unknown')})\n"
                f"PMID: {paper.get('pmid', '')} | URL: {paper.get('url', '')}\n"
                f"Relevance: {paper.get('relevance_score', 0):.2%}\n\n"
                f"{paper.get('abstract_snippet', 'No abstract.')}\n"
            )
            sections.append(section)

        return (
            "RELEVANT MEDICAL RESEARCH PAPERS:\n"
            + "=" * 50 + "\n"
            + "\n" + "-" * 40 + "\n".join(sections)
        )