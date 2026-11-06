import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from medical_crag.storage import ChromaStore
from medical_crag.types import Document
from medical_crag.synthesizer import MedicalSynthesizer
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.mcp_tools import MCPClient


@pytest.fixture
def temp_chroma():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_synthesizer_filters_only_cited_sources():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = (
        {
            "verdict": "REFUTED",
            "confidence": 0.9,
            "consumer_summary": "Vitamin C does not cure cancer.",
            "clinical_explanation": "Extensive trials show no antitumor efficacy [2].",
        },
        "gemini",
    )
    synthesizer = MedicalSynthesizer(mock_llm)
    doc1 = Document(id="doc-1", title="Irrelevant local guideline", text="General wellness info", source="CDC")
    doc2 = Document(id="doc-2", title="Lancet Oncology Trial", text="High dose ascorbic acid showed no benefit", source="Lancet")

    result = synthesizer.synthesize("Vitamin C cures cancer", [doc1, doc2])
    assert len(result.sources) == 1
    assert result.sources[0].id == "doc-2"
    assert result.sources[0].title == "Lancet Oncology Trial"


def test_chroma_store_in_batch_duplicate_ids(temp_chroma):
    store = ChromaStore(persist_dir=temp_chroma, collection_name="test_dedup", auto_seed=False)
    doc = Document(id="pmid-777", title="Trial A", text="Content A", source="PubMed")
    # Pass duplicate doc in same batch
    store.upsert_documents([doc, doc])
    assert store.count() == 1


def test_evaluator_keyword_sanitization_avoids_string_none():
    mock_llm = MagicMock()
    mock_llm.generate_json.return_value = (
        {
            "is_sufficient": True,
            "confidence": 0.95,
            "rationale": "Direct match",
            "search_source": "pubmed",
            "search_keywords": None,  # Model emits null
        },
        "gemini",
    )
    evaluator = MedicalEvaluator(mock_llm)
    res = evaluator.evaluate("Does aspirin reduce heart attack risk?", [Document(id="1", title="Aspirin", text="Reduces risk")])
    assert res.search_keywords != "None"
    assert "aspirin" in res.search_keywords.lower()


def test_mcp_who_multi_token_matching():
    client = MCPClient()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "value": [
            {
                "Title": "Statement on Vaccine Safety and Measles Outbreak",
                "Summary": "Recent epidemiological studies show MMR vaccine does not cause autism.",
                "Url": "/news/item/123",
                "PublicationDateAndTime": "2024",
            }
        ]
    }
    with patch("httpx.Client.get", return_value=mock_response):
        docs = client.search_who("vaccines autism", max_results=1)
        assert len(docs) == 1
        assert docs[0].id != "who-1"  # ID must be deterministic and content-based
        assert "who-" in docs[0].id
