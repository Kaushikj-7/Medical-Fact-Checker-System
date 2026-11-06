"""Main pipeline orchestrating the 4-stage Medical Corrective RAG workflow."""

from __future__ import annotations

import logging
import time
from typing import Optional
from medical_crag.config import Settings
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.llm import LLMGateway
from medical_crag.mcp_tools import MCPClient
from medical_crag.retriever import HybridRetriever
from medical_crag.storage import ChromaStore
from medical_crag.synthesizer import MedicalSynthesizer
from medical_crag.types import MedicalCRAGResult, Verdict

logger = logging.getLogger(__name__)


class MedicalCRAGPipeline:
    """End-to-end Medical Corrective RAG pipeline.

    Execution Flow:
    1. RETRIEVE: Local ChromaDB + BM25 hybrid search.
    2. EVALUATE: Quick LLM check on evidence sufficiency.
    3. CORRECT: If insufficient, query MCP medical search (PubMed, FDA, WHO) and auto-index into ChromaDB.
    4. GENERATE: Grounded dual-layer verdict synthesis with multi-provider failover.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[ChromaStore] = None,
        retriever: Optional[HybridRetriever] = None,
        evaluator: Optional[MedicalEvaluator] = None,
        mcp: Optional[MCPClient] = None,
        synthesizer: Optional[MedicalSynthesizer] = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.store = store or ChromaStore(
            persist_dir=self.settings.chroma_persist_dir,
            collection_name=self.settings.chroma_collection_name,
        )
        self.retriever = retriever or HybridRetriever(self.store)
        self.llm = LLMGateway(self.settings)
        self.evaluator = evaluator or MedicalEvaluator(
            self.llm, confidence_threshold=self.settings.evaluator_confidence_threshold
        )
        self.mcp = mcp or MCPClient(timeout=self.settings.request_timeout)
        self.synthesizer = synthesizer or MedicalSynthesizer(self.llm)

    def verify(self, claim: str) -> MedicalCRAGResult:
        claim = claim.strip()
        if not claim:
            raise ValueError("Health claim must not be empty.")

        t0 = time.perf_counter()

        # Step 1: Retrieve candidate passages from local store
        retrieved_docs = self.retriever.retrieve(claim, top_k=self.settings.top_k)

        # Step 2: Evaluate sufficiency
        eval_res = self.evaluator.evaluate(claim, retrieved_docs)

        # Step 3: Corrective branch
        corrective_taken = False
        evidence_pool = retrieved_docs

        if not eval_res.is_sufficient:
            corrective_taken = True
            logger.info(
                "Local evidence insufficient (confidence=%.2f). Querying MCP source: %s with terms: %s",
                eval_res.confidence,
                eval_res.search_source,
                eval_res.search_keywords,
            )

            external_docs = self.mcp.search(
                eval_res.search_source,
                eval_res.search_keywords,
                max_results=3,
            )

            if external_docs:
                # Auto-ingest newly fetched external documents into ChromaDB
                try:
                    self.store.upsert_documents(external_docs)
                    logger.info("Successfully indexed %d external docs into ChromaDB", len(external_docs))
                except Exception as e:
                    logger.warning("Auto-ingestion into ChromaDB failed: %s", e)

                # Prioritize external docs, retaining at most 1 partially relevant local doc as background
                evidence_pool = external_docs + retrieved_docs[:1]
            else:
                # Deterministic short-circuit when external medical literature yields 0 results
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logger.info("External MCP search returned 0 results. Short-circuiting to INSUFFICIENT_EVIDENCE.")
                return MedicalCRAGResult(
                    claim=claim,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                    confidence=0.0,
                    consumer_summary="No authoritative medical evidence was found to support or debunk this claim.",
                    clinical_explanation=f"Evaluator flagged local evidence as insufficient ({eval_res.rationale}). Subsequent external search across {eval_res.search_source.upper()} for '{eval_res.search_keywords}' returned zero indexed literature.",
                    sources=[],
                    corrective_action_taken=True,
                    execution_time_ms=elapsed_ms,
                    llm_provider_used="none",
                )

        # Step 4: Synthesize grounded verdict
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return self.synthesizer.synthesize(
            claim=claim,
            evidence_docs=evidence_pool,
            corrective_taken=corrective_taken,
            execution_ms=elapsed_ms,
        )
