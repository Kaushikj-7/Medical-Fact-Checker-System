"""Medical Corrective RAG (CRAG) Standalone Package."""

from medical_crag.config import Settings
from medical_crag.pipeline import MedicalCRAGPipeline
from medical_crag.types import Document, MedicalCRAGResult, SourceCitation, Verdict

__all__ = [
    "MedicalCRAGPipeline",
    "MedicalCRAGResult",
    "Document",
    "SourceCitation",
    "Verdict",
    "Settings",
]

__version__ = "1.0.0"
