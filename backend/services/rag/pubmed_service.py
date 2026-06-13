"""
Phase 2 — PubMed Service
Fetches real medical research papers from NCBI/PubMed using Biopython.
No paid API needed. Free key at ncbi.nlm.nih.gov/account (optional but raises rate limit).
"""

from typing import List
from dataclasses import dataclass, field
from Bio import Entrez
from backend.core.config import get_settings
from backend.core.logging import logger


@dataclass
class PubMedPaper:
    pmid: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    url: str = ""

    def full_text(self) -> str:
        """Returns title + abstract combined — used for embedding."""
        return f"{self.title}\n\n{self.abstract}"

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
        }


class PubMedService:
    """
    Searches PubMed and returns structured paper objects.

    Usage:
        svc = PubMedService()
        papers = svc.search("type 2 diabetes latest treatments", max_results=5)
        for p in papers:
            print(p.title)
    """

    def __init__(self):
        settings = get_settings()
        # Tell NCBI who is making requests (required by their terms of service)
        Entrez.email = settings.ncbi_email or "dev@example.com"
        if settings.ncbi_api_key:
            Entrez.api_key = settings.ncbi_api_key
            logger.debug("PubMed: using NCBI API key (10 req/sec limit)")
        else:
            logger.debug("PubMed: no API key — limited to 3 req/sec")

    def search(self, query: str, max_results: int = 5) -> List[PubMedPaper]:
        """
        Search PubMed for papers matching the query.

        Args:
            query:       The doctor's question or keywords
            max_results: How many papers to return (1–20)

        Returns:
            List of PubMedPaper objects sorted by relevance
        """
        logger.info(f"PubMed search: '{query}' (max={max_results})")

        try:
            # Step 1: Search for matching PMIDs
            pmids = self._search_pmids(query, max_results)
            if not pmids:
                logger.warning("PubMed returned no results")
                return []

            # Step 2: Fetch full details for those PMIDs
            papers = self._fetch_papers(pmids)
            logger.info(f"PubMed returned {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Run esearch to get a list of PMIDs matching the query."""
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",       # most relevant first
            usehistory="n",
        )
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])

    def _fetch_papers(self, pmids: List[str]) -> List[PubMedPaper]:
        """Run efetch to get full records for a list of PMIDs."""
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()

        papers = []
        for article in records.get("PubmedArticle", []):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        return papers

    def _parse_article(self, article: dict) -> PubMedPaper | None:
        """Parse a single PubMed XML article dict into a PubMedPaper."""
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]

            # PMID
            pmid = str(medline["PMID"])

            # Title
            title = str(art.get("ArticleTitle", "No title"))

            # Abstract — can be structured (with sections) or plain
            abstract = ""
            abstract_data = art.get("Abstract", {})
            if abstract_data:
                abstract_text = abstract_data.get("AbstractText", "")
                if isinstance(abstract_text, list):
                    # Structured abstract: list of sections
                    parts = []
                    for section in abstract_text:
                        label = getattr(section.attributes, "get", lambda k, d: d)("Label", "")
                        if hasattr(section, "attributes") and section.attributes.get("Label"):
                            parts.append(f"{section.attributes['Label']}: {str(section)}")
                        else:
                            parts.append(str(section))
                    abstract = " ".join(parts)
                else:
                    abstract = str(abstract_text)

            if not abstract:
                abstract = "No abstract available."

            # Authors
            authors = []
            author_list = art.get("AuthorList", [])
            for author in author_list[:5]:  # max 5 authors
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

            # Journal and year
            journal_info = art.get("Journal", {})
            journal = str(journal_info.get("Title", ""))
            year = ""
            pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")))

            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )

        except Exception as e:
            logger.warning(f"Article parse error: {e}")
            return None