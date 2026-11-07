"""Dynamic Weighted Reciprocal Rank Fusion (RRF) retriever combining ChromaDB vector similarity and BM25 lexical scoring."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from medical_crag.storage import ChromaStore
from medical_crag.types import Document

logger = logging.getLogger(__name__)

# Regular expressions for exact biomedical / clinical identifiers
CLINICAL_PATTERNS = [
    # Dosages & units: e.g. 500mg, 10 mcg, 2.5 ml, 100 iu
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?)\b", re.IGNORECASE),
    # Clinical trial identifiers: e.g. NCT02345678
    re.compile(r"\bNCT\d{6,}\b", re.IGNORECASE),
    # PubMed / MEDLINE identifiers: e.g. PMID: 12345678, PMID12345678
    re.compile(r"\bPMID\s*:?\s*\d+\b", re.IGNORECASE),
    # Regulatory agencies & public health authorities
    re.compile(r"\b(FDA|EMA|WHO|CDC|NIH|MHRA)\b", re.IGNORECASE),
    # Pharmacological drug nomenclature & common active suffixes
    re.compile(
        r"\b[a-zA-Z]+(?:mab|nib|statin|sartan|gliflozin|tide|olol|pril|vir|cillin|mycin|floxacin|prazole)\b",
        re.IGNORECASE,
    ),
]


def detect_clinical_tokens(query: str) -> bool:
    """Detect whether query contains exact clinical tokens, dosages, or drug nomenclature."""
    for pattern in CLINICAL_PATTERNS:
        if pattern.search(query):
            return True
    return False


def tokenize(text: str) -> List[str]:
    """Tokenize clinical text preserving hyphens and medical alphanumeric acronyms."""
    return [token for token in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())]


class BM25Index:
    """Lightweight in-memory BM25 lexical ranker."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_lengths: Dict[str, int] = {}
        self.doc_tokens: Dict[str, Counter[str]] = {}
        self.df: Counter[str] = Counter()
        self.total_docs = len(documents)

        total_length = 0
        for doc in documents:
            tokens = tokenize(f"{doc.title} {doc.text}")
            length = len(tokens)
            self.doc_ids.append(doc.id)
            self.doc_lengths[doc.id] = length
            total_length += length
            counts = Counter(tokens)
            self.doc_tokens[doc.id] = counts
            for token in counts:
                self.df[token] += 1

        self.avg_doc_len = (total_length / self.total_docs) if self.total_docs > 0 else 1.0

    def score(self, query: str) -> Dict[str, float]:
        query_tokens = tokenize(query)
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        if not self.total_docs or not query_tokens:
            return scores

        for token in query_tokens:
            df = self.df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            for doc_id, counts in self.doc_tokens.items():
                tf = counts.get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom

        return scores


class HybridRetriever:
    """Merges ChromaDB dense search with BM25 sparse search using Dynamic Weighted RRF."""

    def __init__(self, store: ChromaStore) -> None:
        self.store = store

    def compute_weights(self, query: str) -> Tuple[float, float]:
        """Dynamically compute (alpha_dense, alpha_sparse) based on query clinical tokens."""
        if detect_clinical_tokens(query):
            # Shift weight to lexical matching so exact drug/dosage/trial tokens dominate
            return 0.3, 0.7
        return 0.5, 0.5

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        alpha_dense: Optional[float] = None,
        alpha_sparse: Optional[float] = None,
        k: float = 60.0,
    ) -> List[Document]:
        all_docs = self.store.get_all_documents()
        if not all_docs:
            return []

        doc_map = {doc.id: doc for doc in all_docs}

        # 1. Determine dynamic RRF weights if not explicitly provided
        if alpha_dense is None or alpha_sparse is None:
            dyn_dense, dyn_sparse = self.compute_weights(query)
            a_dense = alpha_dense if alpha_dense is not None else dyn_dense
            a_sparse = alpha_sparse if alpha_sparse is not None else dyn_sparse
        else:
            a_dense, a_sparse = alpha_dense, alpha_sparse

        # 2. Dense semantic search from ChromaDB
        dense_results = self.store.query(query, n_results=min(len(all_docs), max(top_k * 3, 10)))
        dense_ranks = {doc.id: rank for rank, doc in enumerate(dense_results)}

        # 3. Sparse lexical search using synchronized BM25 index
        bm25_index = getattr(self.store, "bm25", None)
        if bm25_index is None:
            bm25_index = BM25Index(all_docs)

        bm25_scores = bm25_index.score(query)
        sorted_bm25_ids = sorted(bm25_scores.keys(), key=lambda did: bm25_scores[did], reverse=True)
        sparse_ranks = {
            doc_id: rank for rank, doc_id in enumerate(sorted_bm25_ids) if bm25_scores[doc_id] > 0.0
        }

        # 4. Weighted Reciprocal Rank Fusion (Weighted RRF)
        # score(d) = (alpha_dense / (k + rank_dense)) + (alpha_sparse / (k + rank_sparse))
        rrf_scores: Dict[str, float] = {}
        all_candidate_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        if not all_candidate_ids:
            all_candidate_ids = set(doc_map.keys())

        for doc_id in all_candidate_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += a_dense / (k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += a_sparse / (k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        ranked_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        results: List[Document] = []
        for doc_id in ranked_ids[:top_k]:
            doc = doc_map.get(doc_id)
            if doc:
                # Create a copy with the calculated RRF score attached
                doc_copy = Document(
                    id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    url=doc.url,
                    published=doc.published,
                    score=round(rrf_scores[doc_id], 5),
                    content_hash=doc.content_hash,
                    metadata=doc.metadata.copy(),
                )
                results.append(doc_copy)

        return results
