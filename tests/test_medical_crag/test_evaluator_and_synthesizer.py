import pytest
from unittest.mock import MagicMock
from medical_crag.types import Document, EvaluationResult, MedicalCRAGResult
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.synthesizer import MedicalSynthesizer


def test_evaluator_sufficient_evidence():
    mock_gateway = MagicMock()
    mock_gateway.generate_json.return_value = (
        {
            "is_sufficient": True,
            "confidence": 0.88,
            "rationale": "Clear guideline directly refuting the claim.",
            "search_source": "who",
            "search_keywords": "vaccines autism",
        },
        "gemini",
    )

    evaluator = MedicalEvaluator(mock_gateway)
    docs = [
        Document(
            id="who-1",
            title="Vaccine Safety",
            text="Vaccines do not cause autism.",
            source="WHO",
        )
    ]
    res = evaluator.evaluate("Vaccines cause autism", docs)

    assert res.is_sufficient is True
    assert res.confidence == 0.88
    assert res.search_keywords == "vaccines autism"


def test_evaluator_insufficient_evidence():
    mock_gateway = MagicMock()
    mock_gateway.generate_json.return_value = (
        {
            "is_sufficient": False,
            "confidence": 0.35,
            "rationale": "No specific evidence on elderberry for hypertension.",
            "search_source": "pubmed",
            "search_keywords": "elderberry hypertension blood pressure",
        },
        "openrouter",
    )

    evaluator = MedicalEvaluator(mock_gateway)
    docs = [Document(id="misc-1", title="General Herbs", text="Herbal teas are popular.", source="Local")]
    res = evaluator.evaluate("Elderberry cures high blood pressure", docs)

    assert res.is_sufficient is False
    assert res.confidence == 0.35
    assert res.search_source == "pubmed"
    assert "elderberry" in res.search_keywords


def test_synthesizer_short_circuit_empty_evidence():
    mock_gateway = MagicMock()
    synthesizer = MedicalSynthesizer(mock_gateway)

    res = synthesizer.synthesize("Extremely obscure claim", [], corrective_taken=True, execution_ms=100)
    assert res.verdict == "INSUFFICIENT_EVIDENCE"
    assert res.confidence == 0.0
    assert len(res.sources) == 0
    # Did not call LLM because it short-circuited safely
    mock_gateway.generate_json.assert_not_called()


def test_synthesizer_with_evidence():
    mock_gateway = MagicMock()
    mock_gateway.generate_json.return_value = (
        {
            "verdict": "REFUTED",
            "confidence": 0.95,
            "consumer_summary": "Drinking bleach does not cure any disease and is extremely dangerous.",
            "clinical_explanation": "Ingestion causes severe corrosive injury and chemical pneumonitis [1].",
        },
        "gemini",
    )

    synthesizer = MedicalSynthesizer(mock_gateway)
    docs = [
        Document(
            id="fda-01",
            title="FDA Bleach Warning",
            text="Bleach ingestion causes caustic tissue burns.",
            source="FDA",
            url="https://www.fda.gov/bleach",
        )
    ]
    res = synthesizer.synthesize("Bleach cures infections", docs, corrective_taken=False, execution_ms=250)

    assert res.verdict == "REFUTED"
    assert res.confidence == 0.95
    assert len(res.sources) == 1
    assert res.sources[0].id == "fda-01"
    assert res.llm_provider_used == "gemini"
