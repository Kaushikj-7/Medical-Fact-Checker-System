import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from medical_crag.config import Settings
from medical_crag.pipeline import MedicalCRAGPipeline
from medical_crag.types import Document, MedicalCRAGResult, SourceCitation


@pytest.fixture
def temp_chroma():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pipeline_sufficient_local_evidence_path(temp_chroma):
    settings = Settings(chroma_persist_dir=temp_chroma, gemini_api_key="mock_key")
    pipeline = MedicalCRAGPipeline(settings=settings)

    # Mock evaluator to return sufficient
    pipeline.evaluator.evaluate = MagicMock(return_value=MagicMock(
        is_sufficient=True,
        confidence=0.9,
        rationale="CDC advisory covers this.",
        search_source="pubmed",
        search_keywords="antibiotics cold",
    ))

    # Mock synthesizer
    pipeline.synthesizer.synthesize = MagicMock(return_value=MedicalCRAGResult(
        claim="Antibiotics cure colds",
        verdict="REFUTED",
        confidence=0.95,
        consumer_summary="Antibiotics only treat bacterial infections, not viral colds.",
        clinical_explanation="Viral upper respiratory infections do not respond to beta-lactam antibiotics [1].",
        sources=[SourceCitation(id="cdc-1", title="CDC Advisory", url="http://cdc.gov", source="CDC", snippet="...")],
        corrective_action_taken=False,
        execution_time_ms=120,
        llm_provider_used="gemini",
    ))

    # MCP should NOT be called
    pipeline.mcp.search = MagicMock()

    result = pipeline.verify("Antibiotics cure colds")
    assert result.verdict == "REFUTED"
    assert result.corrective_action_taken is False
    pipeline.mcp.search.assert_not_called()


def test_pipeline_insufficient_evidence_triggers_mcp_and_auto_ingest(temp_chroma):
    settings = Settings(chroma_persist_dir=temp_chroma, gemini_api_key="mock_key")
    pipeline = MedicalCRAGPipeline(settings=settings)

    # Empty store count before external fetch
    initial_count = pipeline.store.count()

    # Mock evaluator to return insufficient
    pipeline.evaluator.evaluate = MagicMock(return_value=MagicMock(
        is_sufficient=False,
        confidence=0.2,
        rationale="No local data on ashwagandha.",
        search_source="pubmed",
        search_keywords="ashwagandha sleep clinical trial",
    ))

    # Mock MCP returning new PubMed doc
    external_doc = Document(
        id="pubmed-112233",
        title="Ashwagandha for Sleep Quality",
        text="A randomized double-blind placebo-controlled study showed ashwagandha extract improved sleep latency.",
        source="PubMed",
        url="https://pubmed.ncbi.nlm.nih.gov/112233/",
    )
    pipeline.mcp.search = MagicMock(return_value=[external_doc])

    # Mock synthesizer
    pipeline.synthesizer.synthesize = MagicMock(return_value=MedicalCRAGResult(
        claim="Ashwagandha improves sleep",
        verdict="SUPPORTED",
        confidence=0.85,
        consumer_summary="Clinical trials show ashwagandha root extract helps with sleep quality.",
        clinical_explanation="Withania somnifera acts on GABAergic pathways [1].",
        sources=[SourceCitation(id="pubmed-112233", title="Ashwagandha Study", url="http://pubmed", source="PubMed", snippet="...")],
        corrective_action_taken=True,
        execution_time_ms=350,
        llm_provider_used="gemini",
    ))

    result = pipeline.verify("Ashwagandha improves sleep")
    assert result.verdict == "SUPPORTED"
    assert result.corrective_action_taken is True
    pipeline.mcp.search.assert_called_once_with("pubmed", "ashwagandha sleep clinical trial", max_results=3)

    # Verify auto-ingest: ChromaDB store count increased!
    assert pipeline.store.count() == initial_count + 1


def test_pipeline_short_circuit_when_mcp_returns_empty(temp_chroma):
    settings = Settings(chroma_persist_dir=temp_chroma, gemini_api_key="mock_key")
    pipeline = MedicalCRAGPipeline(settings=settings)

    pipeline.evaluator.evaluate = MagicMock(return_value=MagicMock(
        is_sufficient=False,
        confidence=0.1,
        rationale="Unknown claim.",
        search_source="pubmed",
        search_keywords="nonexistent magical stone cure",
    ))

    # MCP returns 0 hits
    pipeline.mcp.search = MagicMock(return_value=[])
    pipeline.synthesizer.synthesize = MagicMock()

    result = pipeline.verify("Magical blue crystal cures pneumonia")
    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == 0.0
    assert result.corrective_action_taken is True
    # Synthesizer bypassed
    pipeline.synthesizer.synthesize.assert_not_called()
