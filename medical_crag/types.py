"""Core dataclasses and contracts for Medical CRAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    MISLEADING = "MISLEADING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


CRAGState = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
MCPSource = Literal["pubmed", "fda", "who"]


@dataclass
class Document:
    """A retrievable medical document or guideline excerpt."""

    id: str
    title: str
    text: str
    source: str = "local"
    url: str = ""
    published: str = ""
    score: float = 0.0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCitation:
    """Authoritative source displayed in final output."""

    id: str
    title: str
    url: str
    source: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Result of CRAG sufficiency check and ternary state routing."""

    status: CRAGState = "AMBIGUOUS"
    confidence: float = 0.0
    rationale: str = ""
    passage_scores: dict[str, float] = field(default_factory=dict)
    target_mcps: list[str] = field(default_factory=lambda: ["pubmed"])
    search_keywords: str = ""

    def __init__(
        self,
        status: Optional[CRAGState] = None,
        confidence: float = 0.0,
        rationale: str = "",
        passage_scores: Optional[dict[str, float]] = None,
        target_mcps: Optional[list[str]] = None,
        search_keywords: str = "",
        is_sufficient: Optional[bool] = None,
        search_source: Optional[str] = None,
    ) -> None:
        if status is not None and isinstance(status, str):
            self.status = status
        elif is_sufficient is not None:
            self.status = "CORRECT" if is_sufficient else "AMBIGUOUS"
        else:
            self.status = "AMBIGUOUS"

        try:
            self.confidence = float(confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0

        self.rationale = str(rationale or "")
        self.passage_scores = passage_scores or {}

        if target_mcps is not None:
            self.target_mcps = target_mcps
        elif search_source is not None:
            self.target_mcps = [str(search_source)]
        else:
            self.target_mcps = ["pubmed"]

        self.search_keywords = str(search_keywords or "")

    @property
    def is_sufficient(self) -> bool:
        return self.status == "CORRECT"

    @property
    def search_source(self) -> str:
        return self.target_mcps[0] if self.target_mcps else "pubmed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "passage_scores": self.passage_scores,
            "target_mcps": self.target_mcps,
            "search_keywords": self.search_keywords,
            "is_sufficient": self.is_sufficient,
            "search_source": self.search_source,
        }


@dataclass
class MedicalCRAGResult:
    """Final dual-layer output of the CRAG pipeline with verified citations."""

    claim: str
    verdict: str  # Verdict enum value
    confidence: float
    consumer_summary: str
    clinical_explanation: str
    sources: List[SourceCitation] = field(default_factory=list)
    corrective_action_taken: bool = False
    execution_time_ms: int = 0
    llm_provider_used: str = "none"
    citation_precision: float = 1.0
    verified_citations: List[str] = field(default_factory=list)
    cached: bool = False
    state: CRAGState = "CORRECT"

    @property
    def patient_summary(self) -> str:
        return self.consumer_summary

    @patient_summary.setter
    def patient_summary(self, val: str) -> None:
        self.consumer_summary = val

    @property
    def clinical_proof_chain(self) -> str:
        return self.clinical_explanation

    @clinical_proof_chain.setter
    def clinical_proof_chain(self, val: str) -> None:
        self.clinical_explanation = val

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": str(self.claim),
            "verdict": str(getattr(self.verdict, "value", self.verdict)),
            "confidence": float(self.confidence) if isinstance(self.confidence, (int, float)) else 0.0,
            "patient_summary": str(self.patient_summary),
            "consumer_summary": str(self.consumer_summary),
            "clinical_proof_chain": str(self.clinical_proof_chain),
            "clinical_explanation": str(self.clinical_explanation),
            "sources": [s.to_dict() for s in self.sources if hasattr(s, "to_dict")],
            "corrective_action_taken": bool(self.corrective_action_taken),
            "execution_time_ms": int(self.execution_time_ms) if isinstance(self.execution_time_ms, (int, float)) else 0,
            "llm_provider_used": str(self.llm_provider_used),
            "citation_precision": float(self.citation_precision) if isinstance(self.citation_precision, (int, float)) else 1.0,
            "verified_citations": [str(c) for c in self.verified_citations],
            "cached": bool(self.cached),
            "state": str(getattr(self, "state", "CORRECT")),
        }

    def to_json(self) -> str:
        def _default(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return str(obj)

        return json.dumps(self.to_dict(), default=_default, indent=2)
