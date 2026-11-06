import pytest
from medical_crag.types import (
    Document,
    SourceCitation,
    EvaluationResult,
    MedicalCRAGResult,
)
from medical_crag.config import Settings


def test_document_and_citation_dataclasses():
    doc = Document(
        id="pmid-123",
        title="Sample Medical Study",
        text="Aspirin reduces cardiovascular events.",
        source="PubMed",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
    )
    assert doc.id == "pmid-123"
    assert doc.source == "PubMed"

    citation = SourceCitation(
        id="pmid-123",
        title="Sample Medical Study",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
        source="PubMed",
        snippet="Aspirin reduces cardiovascular events.",
    )
    assert citation.id == "pmid-123"


def test_evaluation_result_schema():
    eval_res = EvaluationResult(
        is_sufficient=False,
        confidence=0.4,
        rationale="Local evidence is incomplete.",
        search_source="pubmed",
        search_keywords="aspirin cardiovascular risk",
    )
    assert eval_res.is_sufficient is False
    assert eval_res.search_source == "pubmed"


def test_medical_crag_result_schema():
    res = MedicalCRAGResult(
        claim="Aspirin prevents heart attacks",
        verdict="SUPPORTED",
        confidence=0.92,
        consumer_summary="Aspirin has been shown to reduce risk for certain patients.",
        clinical_explanation="Multiple randomized controlled trials support antiplatelet benefits [1].",
        sources=[
            SourceCitation(
                id="pmid-123",
                title="Study Title",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                source="PubMed",
                snippet="Significant reduction in events.",
            )
        ],
        corrective_action_taken=True,
        execution_time_ms=450,
        llm_provider_used="gemini",
    )
    assert res.verdict == "SUPPORTED"
    assert len(res.sources) == 1
    assert res.corrective_action_taken is True


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./custom_chroma")
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "0.75")

    settings = Settings.from_env()
    assert settings.gemini_api_key == "test_gemini_key"
    assert settings.openrouter_api_key == "test_openrouter_key"
    assert settings.chroma_persist_dir == "./custom_chroma"
    assert settings.evaluator_confidence_threshold == 0.75
