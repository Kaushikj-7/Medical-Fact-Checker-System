"""Medical synthesizer producing grounded dual-layer verification results with deterministic citation verification."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from medical_crag.llm import LLMGateway
from medical_crag.types import Document, MedicalCRAGResult, SourceCitation, Verdict

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a grounded medical science fact-checking synthesizer.
Your task is to synthesize an authoritative verdict on a health claim strictly based on the provided evidence passages.

Requirements:
1. Verdict MUST be one of:
   - "SUPPORTED": Evidence confirms the claim is medically sound and safe.
   - "REFUTED": Evidence disproves the claim or shows it is dangerous/ineffective.
   - "MISLEADING": Evidence shows the claim contains partial truth taken out of context or exaggerated.
   - "INSUFFICIENT_EVIDENCE": Evidence is lacking, contradictory, or inconclusive.
2. Confidence (0.0 to 1.0): Reflect your certainty based strictly on the cited literature.
3. Consumer Summary: 2-3 clear, compassionate sentences in plain English for a patient/layperson.
4. Clinical Explanation: 2-4 sentences detailing biomedical mechanisms, citing sources explicitly using bracket numbers like [1], [2].
5. Never state medical claims not supported by the numbered passages.

Respond ONLY with valid JSON:
{
  "verdict": "SUPPORTED" | "REFUTED" | "MISLEADING" | "INSUFFICIENT_EVIDENCE",
  "confidence": 0.0 - 1.0,
  "consumer_summary": "Plain English empathetic summary.",
  "clinical_explanation": "Biomedical explanation with explicit citations like [1]."
}"""

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_:\-,\s]+)\]")
DIGIT_PATTERN = re.compile(r"\b\d+\b")
DIRECT_ID_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-:]+\b")


def verify_citations(
    result: MedicalCRAGResult,
    valid_doc_ids: Set[str],
    id_to_citation_map: Optional[Dict[str, SourceCitation]] = None,
    index_to_id_map: Optional[Dict[int, str]] = None,
) -> MedicalCRAGResult:
    """Deterministically verify and ground all bracket citations in clinical_explanation.

    Extracts all bracketed citations using regex r"\\[([A-Za-z0-9_:-]+)\\]", verifies existence
    in valid_doc_ids, strips or flags hallucinated references, and calculates citation_precision.
    """
    explanation = result.clinical_explanation or ""
    if not explanation or not valid_doc_ids:
        result.citation_precision = 1.0 if not re.search(r"\[.+?\]", explanation) else 0.0
        result.verified_citations = []
        return result

    raw_bracket_matches = list(CITATION_PATTERN.finditer(explanation))
    if not raw_bracket_matches:
        result.citation_precision = 1.0
        result.verified_citations = []
        return result

    total_citations_cited = 0
    verified_ids: List[str] = []
    hallucinated_tokens: Set[str] = set()

    for match in raw_bracket_matches:
        inner_content = match.group(1).strip()
        tokens = [t.strip() for t in inner_content.split(",") if t.strip()]
        for token in tokens:
            total_citations_cited += 1
            matched_id: Optional[str] = None

            # 1. Check if token is a 1-based numeric index [1], [2]
            if token.isdigit() and index_to_id_map:
                idx = int(token)
                if idx in index_to_id_map:
                    matched_id = index_to_id_map[idx]
            # 2. Check direct document ID e.g. [cdc-antibiotics-01] or [pmid-40731001]
            elif token in valid_doc_ids:
                matched_id = token
            else:
                # 3. Check partial ID match
                for valid_id in valid_doc_ids:
                    if token.lower() == valid_id.lower() or token.lower() in valid_id.lower():
                        matched_id = valid_id
                        break

            if matched_id and matched_id in valid_doc_ids:
                if matched_id not in verified_ids:
                    verified_ids.append(matched_id)
            else:
                hallucinated_tokens.add(token)

    precision = len(verified_ids) / total_citations_cited if total_citations_cited > 0 else 1.0
    result.citation_precision = round(precision, 3)
    result.verified_citations = verified_ids

    # If hallucinated citations exist, strip invalid tokens or annotate
    if hallucinated_tokens:
        logger.warning(
            "Found %d hallucinated citation(s) in result: %s",
            len(hallucinated_tokens),
            hallucinated_tokens,
        )

    return result


class MedicalSynthesizer:
    """Synthesizes final dual-layer verification report with deterministic citation verification."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self.llm = llm_gateway

    def synthesize(
        self,
        claim: str,
        evidence_docs: List[Document],
        corrective_taken: bool = False,
        execution_ms: int = 0,
    ) -> MedicalCRAGResult:
        # Deterministic short-circuit on empty evidence
        if not evidence_docs:
            return MedicalCRAGResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                confidence=0.0,
                consumer_summary="No authoritative medical evidence was found to verify or refute this claim.",
                clinical_explanation="Medical databases (including local guidelines and external peer-reviewed literature) returned no verified documentation addressing this claim.",
                sources=[],
                corrective_action_taken=corrective_taken,
                execution_time_ms=execution_ms,
                llm_provider_used="none",
                citation_precision=1.0,
                verified_citations=[],
            )

        citations: List[SourceCitation] = []
        context_items: List[str] = []
        valid_doc_ids: Set[str] = set()
        index_to_id_map: Dict[int, str] = {}
        id_to_citation_map: Dict[str, SourceCitation] = {}

        for idx, doc in enumerate(evidence_docs, start=1):
            valid_doc_ids.add(doc.id)
            index_to_id_map[idx] = doc.id
            context_items.append(
                f"[{idx}] (ID: {doc.id}) Title: {doc.title}\nSource: {doc.source}\nURL: {doc.url}\nExcerpt: {doc.text[:2500]}"
            )
            citation = SourceCitation(
                id=doc.id,
                title=doc.title,
                url=doc.url,
                source=doc.source,
                snippet=doc.text[:250],
            )
            citations.append(citation)
            id_to_citation_map[doc.id] = citation

        prompt = f"Health Claim: \"{claim}\"\n\nAuthoritative Medical Passages:\n" + "\n\n".join(context_items)

        try:
            data, provider = self.llm.generate_json(prompt, system_prompt=SYNTHESIZER_SYSTEM_PROMPT)
            verdict_str = str(data.get("verdict", Verdict.INSUFFICIENT_EVIDENCE.value)).upper()
            if verdict_str not in [v.value for v in Verdict]:
                verdict_str = Verdict.INSUFFICIENT_EVIDENCE.value

            explanation = str(data.get("clinical_explanation", ""))

            # Extract bracket citations e.g. [1], [2], [1, 2], [1-3]
            bracket_blocks = re.findall(r"\[([^\]]+)\]", explanation)
            cited_indices: List[int] = []
            for block in bracket_blocks:
                for num in DIGIT_PATTERN.findall(block):
                    cited_indices.append(int(num))

            grounded_sources: List[SourceCitation] = []
            if cited_indices:
                for idx in sorted(set(cited_indices)):
                    if 1 <= idx <= len(citations):
                        grounded_sources.append(citations[idx - 1])
            else:
                grounded_sources = citations[:2]

            initial_result = MedicalCRAGResult(
                claim=claim,
                verdict=verdict_str,
                confidence=round(float(data.get("confidence", 0.7)), 3),
                consumer_summary=str(data.get("consumer_summary", "")),
                clinical_explanation=explanation,
                sources=grounded_sources,
                corrective_action_taken=corrective_taken,
                execution_time_ms=execution_ms,
                llm_provider_used=provider,
            )

            # Apply deterministic post-processing citation verification
            return verify_citations(
                initial_result,
                valid_doc_ids=valid_doc_ids,
                id_to_citation_map=id_to_citation_map,
                index_to_id_map=index_to_id_map,
            )

        except Exception as e:
            logger.error("Synthesizer error: %s; falling back to cautious result", e)
            fallback_result = MedicalCRAGResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                confidence=0.2,
                consumer_summary="Unable to verify health claim due to synthesis error. Please consult a licensed physician.",
                clinical_explanation=f"Error occurred during evidence synthesis: {e}",
                sources=citations,
                corrective_action_taken=corrective_taken,
                execution_time_ms=execution_ms,
                llm_provider_used="fallback",
                citation_precision=0.0,
                verified_citations=[],
            )
            return fallback_result
