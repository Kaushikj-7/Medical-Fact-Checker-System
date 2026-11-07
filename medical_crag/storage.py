"""Persistent synchronized ChromaDB + SQLite storage manager with claim caching and BM25 index synchronization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional, Set

import chromadb
from chromadb.config import Settings as ChromaSettings

from medical_crag.types import Document, MedicalCRAGResult, SourceCitation

logger = logging.getLogger(__name__)

STARTER_GUIDELINES = [
    Document(
        id="cdc-antibiotics-01",
        title="CDC Advisory: Antibiotic Use and Viral Illness",
        text="Antibiotics do not work against viruses like colds and flu. Taking antibiotics for viral infections can cause unnecessary side effects and contributes to antibiotic resistance.",
        source="CDC",
        url="https://www.cdc.gov/antibiotic-use/index.html",
        published="2024",
    ),
    Document(
        id="fda-statin-grapefruit-01",
        title="FDA Consumer Update: Grapefruit Juice and Certain Drugs",
        text="Grapefruit juice can interact with certain medications, including statins like atorvastatin and simvastatin, by blocking the CYP3A4 enzyme, leading to dangerously high drug levels in the blood.",
        source="FDA",
        url="https://www.fda.gov/consumers/consumer-updates/grapefruit-juice-and-some-drugs-dont-mix",
        published="2023",
    ),
    Document(
        id="who-vaccine-safety-01",
        title="WHO Statement: Vaccine Safety and Autism",
        text="Extensive scientific research across millions of children has conclusively established that vaccines, including MMR, do not cause autism. The original 1998 claim was retracted due to falsified data.",
        source="WHO",
        url="https://www.who.int/news-room/questions-and-answers/item/vaccine-safety",
        published="2023",
    ),
    Document(
        id="fda-bleach-warning-01",
        title="FDA Warning: Ingesting Bleach or Disinfectants",
        text="Ingesting bleach, chlorine dioxide, or industrial disinfectants does not cure COVID-19 or any illness and can cause severe chemical burns, respiratory failure, and fatal poisoning.",
        source="FDA",
        url="https://www.fda.gov/consumers/consumer-updates/danger-dont-drink-miracle-mineral-solution-or-similar-products",
        published="2020",
    ),
]


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SQLiteDocumentStore:
    """Lightweight SQLite database for canonical document persistence and claim caching with TTL."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Documents canonical store
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT,
                    published TEXT,
                    content_hash TEXT UNIQUE NOT NULL,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash)")

            # Claim verification cache with TTL
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS claim_cache (
                    claim_hash TEXT PRIMARY KEY,
                    claim TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_claim_expires ON claim_cache(expires_at)")
            conn.commit()

    def get_existing_hashes(self) -> Set[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM documents")
            return {row["content_hash"] for row in cursor.fetchall()}

    def upsert_documents(self, documents: List[Document]) -> List[Document]:
        """Insert or replace documents in SQLite. Returns documents successfully saved."""
        if not documents:
            return []

        saved: List[Document] = []
        now = time.time()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc in documents:
                h = doc.content_hash or compute_content_hash(doc.text)
                doc.content_hash = h
                meta_json = json.dumps(doc.metadata) if doc.metadata else "{}"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO documents (doc_id, title, content, source, url, published, content_hash, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc.id, doc.title, doc.text, doc.source, doc.url, doc.published, h, meta_json, now),
                )
                saved.append(doc)
            conn.commit()

        return saved

    def get_all_documents(self) -> List[Document]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, title, content, source, url, published, content_hash, metadata_json FROM documents")
            rows = cursor.fetchall()

            docs: List[Document] = []
            for row in rows:
                meta = {}
                try:
                    if row["metadata_json"]:
                        meta = json.loads(row["metadata_json"])
                except Exception:
                    pass
                docs.append(
                    Document(
                        id=row["doc_id"],
                        title=row["title"],
                        text=row["content"],
                        source=row["source"],
                        url=row["url"] or "",
                        published=row["published"] or "",
                        content_hash=row["content_hash"],
                        metadata=meta,
                    )
                )
            return docs

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM documents")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def get_cached_claim(self, claim: str) -> Optional[MedicalCRAGResult]:
        claim_clean = " ".join(claim.strip().lower().split())
        claim_h = hashlib.sha256(claim_clean.encode("utf-8")).hexdigest()
        now = time.time()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT result_json, expires_at FROM claim_cache WHERE claim_hash = ?",
                (claim_h,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            if row["expires_at"] < now:
                # Expired: clean it up
                cursor.execute("DELETE FROM claim_cache WHERE claim_hash = ?", (claim_h,))
                conn.commit()
                return None

            try:
                data = json.loads(row["result_json"])
                sources = [
                    SourceCitation(
                        id=s.get("id", ""),
                        title=s.get("title", ""),
                        url=s.get("url", ""),
                        source=s.get("source", ""),
                        snippet=s.get("snippet", ""),
                    )
                    for s in data.get("sources", [])
                ]
                return MedicalCRAGResult(
                    claim=data.get("claim", claim),
                    verdict=data.get("verdict", "INSUFFICIENT_EVIDENCE"),
                    confidence=data.get("confidence", 0.0),
                    consumer_summary=data.get("consumer_summary", ""),
                    clinical_explanation=data.get("clinical_explanation", ""),
                    sources=sources,
                    corrective_action_taken=data.get("corrective_action_taken", False),
                    execution_time_ms=data.get("execution_time_ms", 0),
                    llm_provider_used=data.get("llm_provider_used", "cache"),
                    citation_precision=data.get("citation_precision", 1.0),
                    verified_citations=data.get("verified_citations", []),
                    cached=True,
                    state=data.get("state", "CORRECT"),
                )
            except Exception as e:
                logger.warning("Failed to deserialize cached claim: %s", e)
                return None

    def cache_claim(self, claim: str, result: MedicalCRAGResult, ttl_seconds: float = 86400.0) -> None:
        claim_clean = " ".join(claim.strip().lower().split())
        claim_h = hashlib.sha256(claim_clean.encode("utf-8")).hexdigest()
        now = time.time()
        expires = now + ttl_seconds

        verdict_str = str(getattr(result, "verdict", "INSUFFICIENT_EVIDENCE"))
        try:
            conf_val = float(getattr(result, "confidence", 0.0))
        except (TypeError, ValueError):
            conf_val = 0.0

        try:
            if hasattr(result, "to_json"):
                json_str = result.to_json()
            elif hasattr(result, "to_dict"):
                json_str = json.dumps(result.to_dict())
            else:
                json_str = json.dumps(str(result))
        except Exception:
            json_str = "{}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO claim_cache (claim_hash, claim, verdict, confidence, result_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_hash) DO UPDATE SET
                    verdict = excluded.verdict,
                    confidence = excluded.confidence,
                    result_json = excluded.result_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (claim_h, claim, verdict_str, conf_val, json_str, now, expires),
            )
            conn.commit()


class ChromaStore:
    """Manages persistent synchronized ChromaDB vector store alongside SQLite and in-memory BM25 index."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "medical_evidence",
        auto_seed: bool = True,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize SQLite document & cache store
        sqlite_path = os.path.join(self.persist_dir, "metadata_store.sqlite")
        self.sqlite = SQLiteDocumentStore(sqlite_path)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory synchronized BM25 index holder
        self._bm25_index: Optional[Any] = None

        # Re-sync ChromaDB from SQLite if SQLite has docs but ChromaDB was empty/reset
        sqlite_count = self.sqlite.count()
        chroma_count = self.collection.count()

        if sqlite_count > 0 and chroma_count == 0:
            all_sqlite_docs = self.sqlite.get_all_documents()
            self._upsert_to_chroma(all_sqlite_docs)
        elif auto_seed and sqlite_count == 0 and chroma_count == 0:
            self.seed_default_guidelines()

        # Rebuild synchronized in-memory BM25 index
        self.sync_bm25_index()

    def count(self) -> int:
        return self.sqlite.count()

    def seed_default_guidelines(self) -> None:
        self.upsert_documents(STARTER_GUIDELINES)

    def sync_bm25_index(self) -> None:
        """Trigger an immediate rebuild/sync of in-memory BM25 index from SQLite."""
        from medical_crag.retriever import BM25Index

        docs = self.sqlite.get_all_documents()
        self._bm25_index = BM25Index(docs)
        logger.debug("Synchronized in-memory BM25 index with %d documents", len(docs))

    @property
    def bm25(self) -> Any:
        if self._bm25_index is None:
            self.sync_bm25_index()
        return self._bm25_index

    def _upsert_to_chroma(self, documents: List[Document]) -> None:
        if not documents:
            return
        ids = [doc.id for doc in documents]
        documents_text = [f"{doc.title}\n{doc.text}" for doc in documents]
        metadatas = [
            {
                "title": doc.title,
                "text": doc.text,
                "source": doc.source,
                "url": doc.url,
                "published": doc.published,
                "content_hash": doc.content_hash,
            }
            for doc in documents
        ]
        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
        )

    def upsert_documents(self, documents: List[Document]) -> None:
        """Synchronously upsert documents into SQLite, ChromaDB, and rebuild BM25 index.

        Prevents split-brain by computing SHA-256 content hashes, deduplicating within
        the batch and against existing SQLite hashes, persisting to both stores, and
        immediately updating the in-memory BM25 index.
        """
        if not documents:
            return

        existing_hashes = self.sqlite.get_existing_hashes()
        seen_batch_ids: Set[str] = set()
        seen_batch_hashes: Set[str] = set()
        unique_new_docs: List[Document] = []

        for doc in documents:
            if not doc.id:
                continue
            h = doc.content_hash or compute_content_hash(doc.text)
            doc.content_hash = h

            # Deduplicate by document ID within batch
            if doc.id in seen_batch_ids:
                continue
            seen_batch_ids.add(doc.id)

            # Deduplicate by content hash within batch and against existing store
            if h in seen_batch_hashes or h in existing_hashes:
                continue
            seen_batch_hashes.add(h)

            unique_new_docs.append(doc)

        if not unique_new_docs:
            return

        # 1. Persist to SQLite
        self.sqlite.upsert_documents(unique_new_docs)

        # 2. Persist to ChromaDB
        self._upsert_to_chroma(unique_new_docs)

        # 3. Immediately trigger incremental/full rebuild of the in-memory BM25 index
        self.sync_bm25_index()

    def query(self, query_text: str, n_results: int = 5) -> List[Document]:
        total = self.collection.count()
        if total == 0:
            return []
        limit = min(n_results, total)

        results = self.collection.query(
            query_texts=[query_text],
            n_results=limit,
        )

        documents: List[Document] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return documents

        ids = results["ids"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, doc_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            dist = distances[idx] if idx < len(distances) else 1.0
            # Cosine distance to similarity: 1 - dist
            score = max(0.0, 1.0 - dist)
            documents.append(
                Document(
                    id=doc_id,
                    title=meta.get("title", ""),
                    text=meta.get("text", ""),
                    source=meta.get("source", "local"),
                    url=meta.get("url", ""),
                    published=meta.get("published", ""),
                    score=round(score, 4),
                    content_hash=meta.get("content_hash", ""),
                )
            )

        return documents

    def get_all_documents(self) -> List[Document]:
        """Fetch all documents from SQLite for guaranteed consistency."""
        return self.sqlite.get_all_documents()

    def get_cached_claim(self, claim: str) -> Optional[MedicalCRAGResult]:
        return self.sqlite.get_cached_claim(claim)

    def cache_claim(self, claim: str, result: MedicalCRAGResult, ttl_seconds: float = 86400.0) -> None:
        self.sqlite.cache_claim(claim, result, ttl_seconds=ttl_seconds)
