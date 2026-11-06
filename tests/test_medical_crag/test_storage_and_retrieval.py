import pytest
import shutil
import tempfile
from medical_crag.types import Document
from medical_crag.storage import ChromaStore
from medical_crag.retriever import HybridRetriever, BM25Index


@pytest.fixture
def temp_chroma_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_chroma_store_upsert_and_query(temp_chroma_dir):
    store = ChromaStore(persist_dir=temp_chroma_dir, collection_name="test_med", auto_seed=False)
    assert store.count() == 0

    docs = [
        Document(
            id="pmid-001",
            title="Aspirin and Cardiovascular Disease",
            text="Low-dose aspirin is recommended for secondary prevention of cardiovascular events.",
            source="PubMed",
            url="https://pubmed.ncbi.nlm.nih.gov/001/",
        ),
        Document(
            id="pmid-002",
            title="Vitamin C and Common Cold",
            text="Vitamin C does not prevent common cold incidence in the general population.",
            source="PubMed",
            url="https://pubmed.ncbi.nlm.nih.gov/002/",
        ),
    ]

    store.upsert_documents(docs)
    assert store.count() == 2

    # Deduplication test: re-inserting same ID does not increase count
    store.upsert_documents([docs[0]])
    assert store.count() == 2

    # Query
    results = store.query("aspirin cardiovascular prevention", n_results=2)
    assert len(results) >= 1
    assert any("Aspirin" in r.title for r in results)


def test_bm25_index():
    docs = [
        Document(id="1", title="Cancer", text="Chemotherapy and immunotherapy for lung cancer."),
        Document(id="2", title="Diabetes", text="Metformin improves insulin sensitivity in type 2 diabetes."),
    ]
    bm25 = BM25Index(docs)
    scores = bm25.score("metformin diabetes")
    assert scores["2"] > scores["1"]


def test_hybrid_retriever(temp_chroma_dir):
    store = ChromaStore(persist_dir=temp_chroma_dir, collection_name="test_med_hybrid")
    docs = [
        Document(
            id="doc-1",
            title="Antibiotics and Viruses",
            text="Antibiotics treat bacterial infections, not viral infections like flu or cold.",
            source="CDC",
        ),
        Document(
            id="doc-2",
            title="Hypertension Guidelines",
            text="ACE inhibitors and ARBs are first-line treatments for high blood pressure.",
            source="AHA",
        ),
    ]
    store.upsert_documents(docs)

    retriever = HybridRetriever(store)
    ranked = retriever.retrieve("Do antibiotics cure the flu?", top_k=2)
    assert len(ranked) >= 1
    assert ranked[0].id == "doc-1"
