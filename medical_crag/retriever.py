"""Hybrid retriever combining ChromaDB vector similarity and BM25 lexical scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List
from medical_crag.storage import ChromaStore
from medical_crag.types import Document


def tokenize(text: str) -> List[str]:
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
    """Merges ChromaDB dense search with BM25 sparse search using RRF."""

    def __init__(self, store: ChromaStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        all_docs = self.store.get_all_documents()
        if not all_docs:
            return []

        # 1. Dense search from ChromaDB
        dense_results = self.store.query(query, n_results=min(len(all_docs), top_k * 3))
        dense_ranks = {doc.id: rank for rank, doc in enumerate(dense_results)}

        # 2. Sparse BM25 search
        bm25 = BM25Index(all_docs)
        bm25_scores = bm25.score(query)
        sorted_bm25_ids = sorted(bm25_scores.keys(), key=lambda did: bm25_scores[did], reverse=True)
        sparse_ranks = {doc_id: rank for rank, doc_id in enumerate(sorted_bm25_ids) if bm25_scores[doc_id] > 0}

        # 3. Reciprocal Rank Fusion (RRF)
        # Score = sum(1 / (k + rank)) where k=60
        rrf_scores: Dict[str, float] = {}
        doc_map = {doc.id: doc for doc in all_docs}

        all_candidate_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        if not all_candidate_ids:
            # Fallback to dense if sparse was completely uninformative
            all_candidate_ids = set(doc_map.keys())

        for doc_id in all_candidate_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (60.0 + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (60.0 + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        ranked_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        results: List[Document] = []
        for doc_id in ranked_ids[:top_k]:
            doc = doc_map.get(doc_id)
            if doc:
                doc.score = round(rrf_scores[doc_id], 5)
                results.append(doc)

        return results
