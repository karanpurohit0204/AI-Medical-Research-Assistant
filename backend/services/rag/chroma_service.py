"""
Phase 2 — ChromaDB Vector Store Service
Embeds PubMed papers using Sentence Transformers and stores/retrieves
them from a local ChromaDB collection.

All free — no API key needed. Runs fully on your machine.
"""

from typing import List
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.logging import logger
from backend.services.rag.pubmed_service import PubMedPaper


# Embedding model loaded once at module level
# all-MiniLM-L6-v2 = fast, small (80MB), good quality, free
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2 ...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model ready.")
    return _embedder


class ChromaService:
    """
    Stores PubMed papers as vector embeddings in ChromaDB.
    Provides similarity search to find papers relevant to a query.

    Usage:
        chroma = ChromaService()
        chroma.add_papers(papers)                       # store papers
        results = chroma.search("diabetes treatment", n=3)  # retrieve relevant ones
    """

    def __init__(self):
        settings = get_settings()

        # Create persist directory if it doesn't exist
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence so vectors survive restarts
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

        self.embedder = get_embedder()
        logger.debug(
            f"ChromaDB ready | collection='{settings.chroma_collection}' "
            f"| docs={self.collection.count()}"
        )

    def add_papers(self, papers: List[PubMedPaper]) -> int:
        """
        Embed and store a list of PubMed papers in ChromaDB.
        Skips papers that are already stored (deduplication by PMID).

        Returns:
            Number of new papers actually added
        """
        if not papers:
            return 0

        # Check which PMIDs are already stored
        existing_ids = set()
        if self.collection.count() > 0:
            existing = self.collection.get(ids=[p.pmid for p in papers])
            existing_ids = set(existing["ids"])

        new_papers = [p for p in papers if p.pmid not in existing_ids]
        if not new_papers:
            logger.debug("All papers already in ChromaDB — skipping")
            return 0

        # Build texts to embed (title + abstract combined)
        texts = [p.full_text() for p in new_papers]
        ids = [p.pmid for p in new_papers]
        metadatas = [
            {
                "pmid": p.pmid,
                "title": p.title,
                "authors": ", ".join(p.authors),
                "journal": p.journal,
                "year": p.year,
                "url": p.url,
                # Store truncated abstract as metadata for retrieval display
                "abstract_snippet": p.abstract[:500],
            }
            for p in new_papers
        ]

        # Embed — SentenceTransformer handles batching internally
        logger.debug(f"Embedding {len(new_papers)} papers ...")
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(new_papers)} papers to ChromaDB (total: {self.collection.count()})")
        return len(new_papers)

    def search(self, query: str, n_results: int = 3) -> List[dict]:
        """
        Find the most relevant papers for a query using cosine similarity.

        Args:
            query:     The doctor's question (after STT transcription)
            n_results: How many papers to return

        Returns:
            List of dicts with keys: pmid, title, abstract_snippet, url,
            authors, journal, year, relevance_score
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty — no papers to search")
            return []

        # Embed the query using the same model as the documents
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()

        # Retrieve top-n most similar papers
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        # Convert cosine distance → similarity score (0–1, higher = better)
        papers = []
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for meta, distance in zip(metadatas, distances):
            relevance_score = round(1 - distance, 4)   # cosine distance → similarity
            papers.append({
                **meta,
                "relevance_score": relevance_score,
            })

        logger.info(f"ChromaDB search: found {len(papers)} relevant papers for '{query[:50]}'")
        return papers

    def count(self) -> int:
        """Return total number of papers stored."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all papers from the collection. Useful for testing."""
        settings = get_settings()
        self.client.delete_collection(settings.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("ChromaDB collection cleared.")
        