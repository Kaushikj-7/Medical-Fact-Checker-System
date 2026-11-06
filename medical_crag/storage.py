"""Persistent ChromaDB storage manager for medical evidence."""

from __future__ import annotations

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from medical_crag.types import Document


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


class ChromaStore:
    """Manages persistent document storage and semantic vector retrieval."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "medical_evidence",
        auto_seed: bool = True,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if auto_seed and self.count() == 0:
            self.seed_default_guidelines()

    def count(self) -> int:
        return self.collection.count()

    def seed_default_guidelines(self) -> None:
        self.upsert_documents(STARTER_GUIDELINES)

    def upsert_documents(self, documents: List[Document]) -> None:
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
            }
            for doc in documents
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 5) -> List[Document]:
        total = self.count()
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
                )
            )

        return documents

    def get_all_documents(self) -> List[Document]:
        """Fetch all documents for sparse keyword indexing."""
        data = self.collection.get()
        docs: List[Document] = []
        if not data or not data.get("ids"):
            return docs

        ids = data["ids"]
        metas = data.get("metadatas", [])
        for idx, doc_id in enumerate(ids):
            meta = metas[idx] if metas and idx < len(metas) else {}
            docs.append(
                Document(
                    id=doc_id,
                    title=meta.get("title", ""),
                    text=meta.get("text", ""),
                    source=meta.get("source", "local"),
                    url=meta.get("url", ""),
                    published=meta.get("published", ""),
                )
            )
        return docs
