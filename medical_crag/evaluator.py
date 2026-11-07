"""Fast evaluator determining whether retrieved evidence is sufficient to verify a claim."""

from __future__ import annotations

import logging
import re
from typing import List
from medical_crag.llm import LLMGateway
from medical_crag.types import Document, EvaluationResult

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are a strict clinical relevance evaluator in a medical fact-checking system.
Your job is to assess whether the provided retrieved passages contain clear, trustworthy evidence to either SUPPORT, REFUTE, or CLARIFY the user's health claim.

Decision Guidelines:
1. If the passages directly discuss the exact substances, conditions, and mechanisms in the claim: set is_sufficient = true, confidence >= 0.70.
2. If the passages are only tangentially related, missing the primary substance/treatment, or inconclusive: set is_sufficient = false, confidence < 0.65.
3. If is_sufficient is false, identify:
   - search_source: "fda" if the claim involves prescription/OTC drugs, supplements, or recalls; "who" if it involves vaccines, viral outbreaks, or public health declarations; otherwise "pubmed" for medical/clinical trials.
   - search_keywords: 2-5 concise search terms combining the key entities (e.g., 'metformin lactic acidosis' or 'turmeric osteoarthritis').

Respond ONLY with valid JSON:
{
  "is_sufficient": true | false,
  "confidence": 0.0 - 1.0,
  "rationale": "Brief justification",
  "search_source": "pubmed" | "fda" | "who",
  "search_keywords": "string"
}"""


STOPWORDS = {"does", "what", "can", "will", "the", "for", "with", "and", "are", "about", "that", "this", "from", "have", "you"}


def _extract_fallback_keywords(claim: str) -> str:
    tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim) if w.lower() not in STOPWORDS]
    return " ".join(tokens) if tokens else claim


class MedicalEvaluator:
    """Evaluates retrieved local evidence against claim sufficiency."""

    def __init__(self, llm_gateway: LLMGateway, confidence_threshold: float = 0.65) -> None:
        self.llm = llm_gateway
        self.threshold = confidence_threshold

    def evaluate(self, claim: str, documents: List[Document]) -> EvaluationResult:
        fallback_kw = _extract_fallback_keywords(claim)

        if not documents:
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.0,
                rationale="No local evidence was retrieved.",
                search_source="pubmed",
                search_keywords=fallback_kw,
            )

        context_lines = []
        for idx, doc in enumerate(documents, start=1):
            context_lines.append(f"[{idx}] Title: {doc.title} (Source: {doc.source})\nText: {doc.text[:2500]}")
        context_str = "\n\n".join(context_lines)

        user_prompt = f"Claim to evaluate: \"{claim}\"\n\nRetrieved Evidence Passages:\n{context_str}"

        try:
            data, _ = self.llm.generate_json(user_prompt, system_prompt=EVALUATOR_SYSTEM_PROMPT)
            is_suff = bool(data.get("is_sufficient", False))
            conf = float(data.get("confidence", 0.0))
            if conf < self.threshold:
                is_suff = False

            raw_kw = data.get("search_keywords")
            if isinstance(raw_kw, list):
                keywords = " ".join(str(k).strip() for k in raw_kw if k)
            elif raw_kw and str(raw_kw).strip().lower() != "none":
                keywords = str(raw_kw).strip()
            else:
                keywords = fallback_kw

            return EvaluationResult(
                is_sufficient=is_suff,
                confidence=round(conf, 3),
                rationale=str(data.get("rationale", "")),
                search_source=str(data.get("search_source", "pubmed")).lower(),
                search_keywords=keywords,
            )
        except Exception as e:
            logger.warning("Evaluator failed to parse LLM response: %s; triggering corrective search", e)
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.3,
                rationale=f"Evaluation fallback triggered: {e}",
                search_source="pubmed",
                search_keywords=fallback_kw,
            )
