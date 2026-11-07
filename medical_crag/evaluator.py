"""Ternary CRAG Evaluator implementing per-passage grading, pruning, and multi-MCP routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from medical_crag.llm import LLMGateway
from medical_crag.types import CRAGState, Document, EvaluationResult

logger = logging.getLogger(__name__)

STOPWORDS = {
    "does", "what", "can", "will", "the", "for", "with", "and", "are", "about",
    "that", "this", "from", "have", "you", "should", "could", "would", "any", "all",
}

EVALUATOR_SYSTEM_PROMPT = """You are a rigorous clinical relevance evaluator in an Autonomous Medical Corrective RAG (CRAG) system.
Your job is to evaluate each retrieved evidence passage independently against the health claim, assign a relevance score between 0.0 and 1.0, and determine the optimal authoritative search sources if additional evidence is needed.

Evaluation Criteria:
- 0.80 - 1.00: Directly addresses the claim with concrete clinical trial data, official regulatory guidance, or meta-analyses.
- 0.50 - 0.79: Partially addresses or provides relevant background, but lacks definitive clinical endpoints or direct mechanism resolution.
- 0.00 - 0.49: Irrelevant, tangential, off-topic, or discusses different substances/conditions.

Routing & MCP Sources:
- target_mcps: List containing one or more of ["pubmed", "fda", "who"]:
  * "pubmed": Clinical trials, biomedical research, meta-analyses, and efficacy studies.
  * "fda": Drug safety warnings, adverse reactions, package inserts, and drug-drug interactions.
  * "who": Global disease outbreaks, infectious epidemic alerts, and vaccine safety statements.
  If the claim involves both clinical efficacy and safety/interactions, include both (e.g. ["pubmed", "fda"]).

Respond ONLY with valid JSON:
{
  "passage_scores": {
    "1": 0.0 to 1.0,
    "2": 0.0 to 1.0
  },
  "confidence": 0.0 to 1.0,
  "rationale": "Clinical rationale explaining evidence quality",
  "target_mcps": ["pubmed", "fda"],
  "search_keywords": "concise search query (e.g. 'atorvastatin grapefruit cyp3a4 interaction')"
}"""


def extract_fallback_keywords(claim: str) -> str:
    """Extract clean medical search keywords by filtering common stopwords."""
    tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", claim) if w.lower() not in STOPWORDS]
    return " ".join(tokens) if tokens else claim


def determine_fallback_mcps(claim: str) -> List[str]:
    """Deterministically infer target MCP sources from claim entities if LLM routing is unavailable."""
    claim_lower = claim.lower()
    targets: List[str] = []

    # FDA indicators: drug warnings, interactions, recalls, dosages
    if any(k in claim_lower for k in [
        "drug", "interaction", "side effect", "adverse", "warning", "recall",
        "statin", "dose", "dosage", "fda", "prescription", "otc", "medication", "toxic"
    ]):
        targets.append("fda")

    # WHO indicators: pandemics, viruses, outbreaks, vaccines, public health
    if any(k in claim_lower for k in [
        "outbreak", "epidemic", "pandemic", "virus", "infection", "vaccine", "who", "ebola", "cholera", "mpox", "covid"
    ]):
        targets.append("who")

    # PubMed is always relevant for biomedical evidence
    if "pubmed" not in targets:
        targets.append("pubmed")

    return targets


class MedicalEvaluator:
    """Evaluates retrieved local evidence with per-passage grading and ternary CRAG routing."""

    def __init__(self, llm_gateway: LLMGateway, confidence_threshold: float = 0.65) -> None:
        self.llm = llm_gateway
        self.threshold = confidence_threshold

    @staticmethod
    def prune_passages(
        documents: List[Document],
        passage_scores: Any,
        min_score: float = 0.50,
    ) -> List[Document]:
        """Prune any local passage with score < min_score so irrelevant chunks are discarded."""
        if not isinstance(passage_scores, dict):
            return documents

        surviving: List[Document] = []
        for doc in documents:
            try:
                score = float(passage_scores.get(doc.id, 0.0))
            except (TypeError, ValueError):
                score = 0.0

            if score >= min_score:
                doc_copy = Document(
                    id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    url=doc.url,
                    published=doc.published,
                    score=round(score, 3),
                    content_hash=doc.content_hash,
                    metadata=doc.metadata.copy(),
                )
                surviving.append(doc_copy)
        return surviving

    @staticmethod
    def determine_crag_state(
        passage_scores: Dict[str, float],
        min_prune_score: float = 0.50,
        correct_threshold: float = 0.80,
        min_surviving_for_correct: int = 2,
    ) -> Tuple[CRAGState, float]:
        """Determine ternary CRAG state based on passage scores:

        - CORRECT: max_score >= 0.80 AND at least 2 passages (or all available) survive pruning.
        - AMBIGUOUS: 0.50 <= max_score < 0.80 (or max >= 0.80 with insufficient surviving).
        - INCORRECT: max_score < 0.50 (all local passages discarded).
        """
        if not passage_scores:
            return "INCORRECT", 0.0

        scores = [float(s) for s in passage_scores.values() if isinstance(s, (int, float))]
        if not scores:
            return "INCORRECT", 0.0

        max_score = max(scores)
        surviving_count = sum(1 for s in scores if s >= min_prune_score)
        required_surviving = min(len(scores), min_surviving_for_correct)

        if max_score >= correct_threshold and surviving_count >= required_surviving:
            return "CORRECT", round(max_score, 3)
        elif max_score >= min_prune_score:
            return "AMBIGUOUS", round(max_score, 3)
        else:
            return "INCORRECT", round(max_score, 3)

    def evaluate(self, claim: str, documents: List[Document]) -> EvaluationResult:
        fallback_kw = extract_fallback_keywords(claim)
        fallback_mcps = determine_fallback_mcps(claim)

        if not documents:
            return EvaluationResult(
                status="INCORRECT",
                confidence=0.0,
                rationale="No local evidence was retrieved from the database.",
                passage_scores={},
                target_mcps=fallback_mcps,
                search_keywords=fallback_kw,
            )

        context_lines = []
        id_to_num_map: Dict[str, str] = {}
        num_to_id_map: Dict[str, str] = {}

        for idx, doc in enumerate(documents, start=1):
            num_str = str(idx)
            id_to_num_map[doc.id] = num_str
            num_to_id_map[num_str] = doc.id
            context_lines.append(
                f"[{idx}] (ID: {doc.id}) Title: {doc.title} (Source: {doc.source})\nText: {doc.text[:2500]}"
            )
        context_str = "\n\n".join(context_lines)

        user_prompt = f"Health Claim: \"{claim}\"\n\nCandidate Evidence Passages:\n{context_str}"

        try:
            data, _ = self.llm.generate_json(user_prompt, system_prompt=EVALUATOR_SYSTEM_PROMPT)

            # 1. Parse per-passage scores
            raw_scores = data.get("passage_scores", {})
            passage_scores: Dict[str, float] = {}

            if isinstance(raw_scores, dict):
                for k, v in raw_scores.items():
                    k_str = str(k).strip()
                    try:
                        score_val = float(v)
                        score_val = max(0.0, min(1.0, score_val))
                    except (ValueError, TypeError):
                        score_val = 0.0

                    # Map index number to real document ID if returned as numeric string
                    if k_str in num_to_id_map:
                        doc_id = num_to_id_map[k_str]
                        passage_scores[doc_id] = score_val
                    elif k_str in id_to_num_map:
                        passage_scores[k_str] = score_val
                    else:
                        matched = False
                        for orig_id in id_to_num_map:
                            if k_str in orig_id or orig_id in k_str:
                                passage_scores[orig_id] = score_val
                                matched = True
                                break
                        if not matched and documents:
                            passage_scores[k_str] = score_val

            # Handle legacy is_sufficient fallback
            explicit_suff = data.get("is_sufficient")
            explicit_conf = data.get("confidence")

            for doc in documents:
                if doc.id not in passage_scores:
                    if explicit_suff is not None:
                        passage_scores[doc.id] = 0.88 if explicit_suff else 0.35
                    else:
                        passage_scores[doc.id] = 0.0

            # 2. Compute ternary state transition
            computed_state, computed_conf = self.determine_crag_state(passage_scores)

            if explicit_suff is True:
                state = "CORRECT"
            elif explicit_suff is False:
                state = "AMBIGUOUS" if float(explicit_conf or 0.0) >= 0.50 else "INCORRECT"
            else:
                state = computed_state

            final_conf = float(explicit_conf) if explicit_conf is not None else computed_conf
            if explicit_suff is True and final_conf < self.threshold:
                state = "AMBIGUOUS"

            # 3. Parse target MCPs
            raw_mcps = data.get("target_mcps")
            target_mcps: List[str] = []
            valid_sources = {"pubmed", "fda", "who"}

            if isinstance(raw_mcps, list):
                for m in raw_mcps:
                    m_clean = str(m).lower().strip()
                    if m_clean in valid_sources and m_clean not in target_mcps:
                        target_mcps.append(m_clean)
            elif isinstance(raw_mcps, str):
                m_clean = raw_mcps.lower().strip()
                if m_clean in valid_sources:
                    target_mcps.append(m_clean)

            if not target_mcps and "search_source" in data:
                src = str(data["search_source"]).lower().strip()
                if src in valid_sources:
                    target_mcps.append(src)

            if not target_mcps:
                target_mcps = fallback_mcps

            # 4. Parse search keywords
            raw_kw = data.get("search_keywords")
            if isinstance(raw_kw, list):
                keywords = " ".join(str(k).strip() for k in raw_kw if k)
            elif raw_kw and str(raw_kw).strip().lower() not in ("none", "null", ""):
                keywords = str(raw_kw).strip()
            else:
                keywords = fallback_kw

            rationale = str(data.get("rationale") or data.get("reasoning") or f"CRAG evaluated state as {state}")

            return EvaluationResult(
                status=state,
                confidence=round(final_conf, 3),
                rationale=rationale,
                passage_scores=passage_scores,
                target_mcps=target_mcps,
                search_keywords=keywords,
            )

        except Exception as e:
            logger.warning("Evaluator failed to parse LLM response: %s; using deterministic fallback", e)
            claim_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", claim.lower()))
            passage_scores = {}
            for doc in documents:
                doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", f"{doc.title} {doc.text}".lower()))
                overlap = len(claim_tokens & doc_tokens) / max(len(claim_tokens), 1)
                passage_scores[doc.id] = round(overlap, 3)

            state, conf = self.determine_crag_state(passage_scores)
            return EvaluationResult(
                status="AMBIGUOUS" if conf >= 0.50 else "INCORRECT",
                confidence=round(conf, 3),
                rationale=f"Evaluation fallback triggered: {e}",
                passage_scores=passage_scores,
                target_mcps=fallback_mcps,
                search_keywords=fallback_kw,
            )
