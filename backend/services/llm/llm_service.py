"""
Phase 3 — LLM Service using Groq (free tier, Llama 3)
Takes the doctor's question + retrieved paper context and generates
a structured medical answer with citations and confidence score.

Get your free key at: console.groq.com
"""

import re
from groq import Groq
from backend.core.config import get_settings
from backend.core.logging import logger


MEDICAL_SYSTEM_PROMPT = """You are a clinical research assistant helping doctors find evidence-based answers.

You will be given:
1. A doctor's question
2. A set of relevant PubMed research papers

Your job is to:
- Answer the question based ONLY on the provided papers
- Be concise, clinical, and precise
- Always cite papers using their [number] from the context
- End with a confidence score (0-100%) based on how well the papers support your answer

Format your response EXACTLY like this:

ANSWER:
<your evidence-based answer here, 3-5 sentences, citing papers as [1], [2] etc>

CITATIONS:
<list each cited paper as: [N] Author et al. - Title (Journal, Year) - PMID>

CONFIDENCE: <number>%
REASONING: <one sentence explaining your confidence level>"""


class LLMService:
    """
    Generates medical answers using Groq's free LLM API (Llama 3).

    Usage:
        llm = LLMService()
        response = llm.generate_answer(query, context)
        print(response.answer)
        print(response.confidence_score)
    """

    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env. "
                "Get your free key at console.groq.com"
            )
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        logger.info(f"LLM service ready | model={self.model}")

    def generate_answer(self, query: str, context: str) -> "LLMResponse":
        """
        Generate a medical answer from the query and retrieved paper context.

        Args:
            query:   The doctor's question (from STT)
            context: Formatted paper context from RAGPipeline._build_context()

        Returns:
            LLMResponse with answer, citations, and confidence score
        """
        logger.info(f"LLM generating answer for: '{query[:60]}'")

        user_message = f"DOCTOR'S QUESTION:\n{query}\n\n{context}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.3,     # low temperature = more factual, less creative
                max_tokens=1024,
            )

            raw = completion.choices[0].message.content
            logger.debug(f"LLM raw response:\n{raw[:200]}...")
            return LLMResponse.parse(raw, model=self.model)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


class LLMResponse:
    """Parsed response from the LLM."""

    def __init__(
        self,
        answer: str,
        citations_text: str,
        confidence_score: float,
        confidence_reasoning: str,
        raw: str,
        model: str,
    ):
        self.answer = answer
        self.citations_text = citations_text
        self.confidence_score = confidence_score        # 0.0 – 1.0
        self.confidence_reasoning = confidence_reasoning
        self.raw = raw
        self.model = model

    @classmethod
    def parse(cls, raw: str, model: str) -> "LLMResponse":
        """
        Parse the structured LLM output into fields.
        Strips <think>...</think> blocks from reasoning models (deepseek-r1 etc.)
        """
        # Strip <think>...</think> blocks — used by reasoning models like deepseek-r1
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # If nothing remains after stripping think blocks, use raw as fallback
        if not cleaned:
            cleaned = raw

        answer = ""
        citations_text = ""
        confidence_score = 0.75
        confidence_reasoning = ""

        lines = cleaned.split("\n")
        section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("ANSWER:"):
                section = "answer"
                rest = line[len("ANSWER:"):].strip()
                if rest:
                    answer += rest + " "
            elif line.startswith("CITATIONS:"):
                section = "citations"
                rest = line[len("CITATIONS:"):].strip()
                if rest:
                    citations_text += rest + "\n"
            elif line.startswith("CONFIDENCE:"):
                section = "confidence"
                val = line[len("CONFIDENCE:"):].strip().replace("%", "").strip()
                try:
                    confidence_score = float(val) / 100
                except ValueError:
                    confidence_score = 0.75
            elif line.startswith("REASONING:"):
                confidence_reasoning = line[len("REASONING:"):].strip()
                section = None
            elif section == "answer":
                answer += line + " "
            elif section == "citations":
                citations_text += line + "\n"

        # Fallback: if parsing failed, return full cleaned text as answer
        if not answer.strip():
            answer = cleaned

        return cls(
            answer=answer.strip(),
            citations_text=citations_text.strip(),
            confidence_score=round(min(max(confidence_score, 0.0), 1.0), 2),
            confidence_reasoning=confidence_reasoning,
            raw=raw,
            model=model,
        )

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations_text": self.citations_text,
            "confidence_score": self.confidence_score,
            "confidence_reasoning": self.confidence_reasoning,
            "model": self.model,
        }