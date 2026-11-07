"""Medical synthesizer producing grounded dual-layer verification results."""

from __future__ import annotations

import logging
from typing import List
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


class MedicalSynthesizer:
    """Synthesizes final dual-layer verification report."""

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
            )

        citations: List[SourceCitation] = []
        context_items: List[str] = []
        for idx, doc in enumerate(evidence_docs, start=1):
            context_items.append(
                f"[{idx}] Title: {doc.title}\nSource: {doc.source}\nURL: {doc.url}\nExcerpt: {doc.text[:2500]}"
            )
            citations.append(
                SourceCitation(
                    id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    source=doc.source,
                    snippet=doc.text[:250],
                )
            )

        prompt = f"Health Claim: \"{claim}\"\n\nAuthoritative Medical Passages:\n" + "\n\n".join(context_items)

        try:
            data, provider = self.llm.generate_json(prompt, system_prompt=SYNTHESIZER_SYSTEM_PROMPT)
            verdict_str = str(data.get("verdict", Verdict.INSUFFICIENT_EVIDENCE.value)).upper()
            if verdict_str not in [v.value for v in Verdict]:
                verdict_str = Verdict.INSUFFICIENT_EVIDENCE.value

            explanation = str(data.get("clinical_explanation", ""))

            # Extract bracket citations e.g. [1], [2], [1, 2], [1-3] to strictly ground sources
            import re
            bracket_blocks = re.findall(r"\[([^\]]+)\]", explanation)
            cited_indices: List[int] = []
            for block in bracket_blocks:
                for num in re.findall(r"\b\d+\b", block):
                    cited_indices.append(int(num))

            grounded_sources: List[SourceCitation] = []
            if cited_indices:
                for idx in sorted(set(cited_indices)):
                    if 1 <= idx <= len(citations):
                        grounded_sources.append(citations[idx - 1])
            else:
                grounded_sources = citations[:2]

            return MedicalCRAGResult(
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
        except Exception as e:
            logger.error("Synthesizer error: %s; falling back to cautious result", e)
            return MedicalCRAGResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                confidence=0.2,
                consumer_summary="Unable to verify health claim due to synthesis error. Please consult a licensed physician.",
                clinical_explanation=f"Error occurred during evidence synthesis: {e}",
                sources=citations,
                corrective_action_taken=corrective_taken,
                execution_time_ms=execution_ms,
                llm_provider_used="fallback",
            )
