"""Core dataclasses and contracts for Medical CRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    MISLEADING = "MISLEADING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceCitation:
    """Authoritative source displayed in final output."""
    id: str
    title: str
    url: str
    source: str
    snippet: str


@dataclass
class EvaluationResult:
    """Result of CRAG sufficiency check."""
    is_sufficient: bool
    confidence: float
    rationale: str
    search_source: str = "pubmed"  # pubmed | fda | who
    search_keywords: str = ""


@dataclass
class MedicalCRAGResult:
    """Final dual-layer output of the CRAG pipeline."""
    claim: str
    verdict: str  # Verdict enum value
    confidence: float
    consumer_summary: str
    clinical_explanation: str
    sources: List[SourceCitation] = field(default_factory=list)
    corrective_action_taken: bool = False
    execution_time_ms: int = 0
    llm_provider_used: str = "none"
