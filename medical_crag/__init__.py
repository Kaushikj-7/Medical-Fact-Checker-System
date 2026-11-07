"""Autonomous Medical Corrective-RAG (CRAG) & MCP Platform."""

from medical_crag.config import Settings
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.mcp_tools import MCPClient, gate_mcp_documents
from medical_crag.pipeline import MedicalCRAGPipeline
from medical_crag.retriever import BM25Index, HybridRetriever, detect_clinical_tokens
from medical_crag.storage import ChromaStore, SQLiteDocumentStore
from medical_crag.synthesizer import MedicalSynthesizer, verify_citations
from medical_crag.types import (
    CRAGState,
    Document,
    EvaluationResult,
    MedicalCRAGResult,
    SourceCitation,
    Verdict,
)

__all__ = [
    "MedicalCRAGPipeline",
    "MedicalCRAGResult",
    "Document",
    "SourceCitation",
    "Verdict",
    "EvaluationResult",
    "CRAGState",
    "Settings",
    "ChromaStore",
    "SQLiteDocumentStore",
    "BM25Index",
    "HybridRetriever",
    "detect_clinical_tokens",
    "MedicalEvaluator",
    "MedicalSynthesizer",
    "verify_citations",
    "MCPClient",
    "gate_mcp_documents",
]

__version__ = "2.0.0"
