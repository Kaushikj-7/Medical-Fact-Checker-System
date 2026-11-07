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


# ==============================================================================
# Synchronized Storage & Dynamic Weighted RRF Tests
# ==============================================================================

from medical_crag.mcp_tools import gate_mcp_documents
from medical_crag.retriever import HybridRetriever, detect_clinical_tokens
from medical_crag.types import MedicalCRAGResult


def test_sqlite_content_hash_deduplication(temp_chroma):
    store = ChromaStore(persist_dir=temp_chroma, collection_name="test_hash_sync", auto_seed=False)
    doc1 = Document(id="doc-original", title="Study 1", text="Identical clinical trial protocol text.")
    doc2 = Document(id="doc-duplicate-diff-id", title="Study 2", text="Identical clinical trial protocol text.")

    store.upsert_documents([doc1])
    assert store.count() == 1

    # Ingesting same content under different ID should be deduplicated by content hash
    store.upsert_documents([doc2])
    assert store.count() == 1


def test_claim_cache_with_ttl(temp_chroma):
    store = ChromaStore(persist_dir=temp_chroma, collection_name="test_cache", auto_seed=False)
    claim = "Zinc reduces common cold duration"
    result = MedicalCRAGResult(
        claim=claim,
        verdict="SUPPORTED",
        confidence=0.85,
        consumer_summary="Zinc lozenges reduce duration.",
        clinical_explanation="Meta-analyses show reduced cold symptoms.",
    )

    # 1. Uncached lookup returns None
    assert store.get_cached_claim(claim) is None

    # 2. Cache with 10s TTL
    store.cache_claim(claim, result, ttl_seconds=10.0)
    cached = store.get_cached_claim(claim)
    assert cached is not None
    assert cached.cached is True
    assert cached.verdict == "SUPPORTED"
    assert cached.confidence == 0.85

    # 3. Cache with expired TTL (0.001s)
    store.cache_claim(claim, result, ttl_seconds=-1.0)
    assert store.get_cached_claim(claim) is None


def test_bm25_index_synchronization_on_upsert(temp_chroma):
    store = ChromaStore(persist_dir=temp_chroma, collection_name="test_bm25_sync", auto_seed=False)
    assert store.bm25.total_docs == 0

    doc = Document(id="doc-paxlovid", title="Paxlovid Trials", text="Nirmatrelvir reduces hospitalization.")
    store.upsert_documents([doc])

    # BM25 index must be immediately updated without needing process restart
    assert store.bm25.total_docs == 1
    scores = store.bm25.score("paxlovid")
    assert scores["doc-paxlovid"] > 0.0


def test_clinical_token_detection_and_weighted_rrf(temp_chroma):
    store = ChromaStore(persist_dir=temp_chroma, collection_name="test_rrf_weights", auto_seed=False)
    retriever = HybridRetriever(store)

    # 1. Exact clinical token checks
    assert detect_clinical_tokens("Take 500mg amoxicillin twice daily") is True
    assert detect_clinical_tokens("Efficacy reported in trial NCT03456789") is True
    assert detect_clinical_tokens("Citing PMID: 28938472") is True
    assert detect_clinical_tokens("FDA issued a boxed warning for statin therapy") is True
    assert detect_clinical_tokens("Atorvastatin vs semaglutide outcomes") is True

    # 2. General non-clinical token checks
    assert detect_clinical_tokens("Drinking warm lemon water cures headaches") is False

    # 3. Weight shift verification
    alpha_dense_clin, alpha_sparse_clin = retriever.compute_weights("Metformin 1000mg dosage")
    assert alpha_sparse_clin == 0.7
    assert alpha_dense_clin == 0.3

    alpha_dense_gen, alpha_sparse_gen = retriever.compute_weights("Eating carrots improves night vision")
    assert alpha_sparse_gen == 0.5
    assert alpha_dense_gen == 0.5


def test_gated_mcp_write_back_filters_poisoned_or_boilerplate_docs():
    valid_doc = Document(
        id="valid-doc-1",
        title="Turmeric in knee osteoarthritis",
        text="A randomized double-blind placebo-controlled trial demonstrating curcumin significantly reduced WOMAC pain scores in knee osteoarthritis.",
    )
    short_boilerplate_doc = Document(
        id="short-doc-2",
        title="Short note",
        text="Click here to read more.",  # < 50 chars
    )
    irrelevant_doc = Document(
        id="irrelevant-doc-3",
        title="Space flight dynamics",
        text="Orbital trajectories of low Earth orbit communication satellites and propulsion telemetry systems.",
    )

    query = "turmeric reduces pain in knee osteoarthritis"
    admitted = gate_mcp_documents([valid_doc, short_boilerplate_doc, irrelevant_doc], query=query, min_relevance=0.70)

    admitted_ids = [d.id for d in admitted]
    assert "valid-doc-1" in admitted_ids
    assert "short-doc-2" not in admitted_ids
    assert "irrelevant-doc-3" not in admitted_ids
