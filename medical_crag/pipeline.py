"""Autonomous Medical Corrective RAG (CRAG) pipeline orchestrator with caching and async streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, List, Optional

from medical_crag.config import Settings
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.llm import LLMGateway
from medical_crag.mcp_tools import MCPClient, gate_mcp_documents
from medical_crag.retriever import HybridRetriever
from medical_crag.storage import ChromaStore
from medical_crag.synthesizer import MedicalSynthesizer
from medical_crag.types import Document, MedicalCRAGResult, Verdict

logger = logging.getLogger(__name__)


class MedicalCRAGPipeline:
    """End-to-end Autonomous Medical Corrective RAG pipeline.

    Execution Flow:
    1. CACHE CHECK: Lookup SQLite TTL claim cache for identical query.
    2. RETRIEVE: Synchronized hybrid ChromaDB vector + BM25 keyword search using Weighted RRF.
    3. EVALUATE: Ternary CRAG state grading (CORRECT / AMBIGUOUS / INCORRECT) with per-passage pruning.
    4. CORRECT: If AMBIGUOUS or INCORRECT, fan-out parallel MCP searches (PubMed/FDA/WHO) with gated write-back.
    5. SYNTHESIZE: Dual-layer grounded clinical synthesis with deterministic citation verification.
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
        """Synchronously verify a biomedical claim through the 4-stage CRAG pipeline."""
        claim = claim.strip()
        if not claim:
            raise ValueError("Health claim must not be empty.")

        t0 = time.perf_counter()

        # Step 0: SQLite Claim Cache check with TTL
        cached_res = self.store.get_cached_claim(claim)
        if cached_res is not None:
            cached_res.execution_time_ms = int((time.perf_counter() - t0) * 1000)
            logger.info("Serving claim '%s' from SQLite cache", claim[:40])
            return cached_res

        # Step 1: Hybrid Retrieve (ChromaDB + BM25 via Weighted RRF)
        retrieved_docs = self.retriever.retrieve(claim, top_k=self.settings.top_k)

        # Step 2: Ternary CRAG Evaluation & Passage Grading
        eval_res = self.evaluator.evaluate(claim, retrieved_docs)
        passage_scores = getattr(eval_res, "passage_scores", {})
        surviving_local = self.evaluator.prune_passages(
            retrieved_docs, passage_scores, min_score=0.50
        )

        # Check status and is_sufficient with mock compatibility
        raw_status = getattr(eval_res, "status", None)
        is_suff = getattr(eval_res, "is_sufficient", None)
        if isinstance(raw_status, str):
            status = raw_status
        elif isinstance(is_suff, bool):
            status = "CORRECT" if is_suff else "AMBIGUOUS"
        else:
            status = "CORRECT" if bool(is_suff) else "AMBIGUOUS"

        target_mcps_raw = getattr(eval_res, "target_mcps", None)
        if isinstance(target_mcps_raw, list) and target_mcps_raw:
            target_mcps = target_mcps_raw
        else:
            src = getattr(eval_res, "search_source", "pubmed")
            target_mcps = [str(src)] if src and isinstance(src, str) else ["pubmed"]

        corrective_taken = False
        evidence_pool: List[Document] = []

        if status == "CORRECT":
            # Local evidence is sufficient; skip external search
            evidence_pool = surviving_local if surviving_local else retrieved_docs
        else:
            # AMBIGUOUS or INCORRECT: trigger corrective MCP external search
            corrective_taken = True
            logger.info(
                "CRAG State is %s (confidence=%.2f). Querying target MCPs %s for: '%s'",
                status,
                getattr(eval_res, "confidence", 0.0),
                target_mcps,
                eval_res.search_keywords,
            )

            if len(target_mcps) == 1:
                raw_external = self.mcp.search(
                    target_mcps[0],
                    eval_res.search_keywords,
                    max_results=3,
                )
            else:
                raw_external = self.mcp.search_all(
                    target_mcps,
                    eval_res.search_keywords,
                    max_results_per_source=3,
                )

            if raw_external:
                # Gated write-back: prevent index poisoning by filtering content and relevance
                gated_docs = gate_mcp_documents(raw_external, eval_res.search_keywords, min_relevance=0.50, min_length=15)
                docs_to_ingest = gated_docs if gated_docs else raw_external
                try:
                    self.store.upsert_documents(docs_to_ingest)
                    logger.info("Indexed %d external docs into ChromaDB & SQLite", len(docs_to_ingest))
                except Exception as e:
                    logger.warning("Gated write-back failed: %s", e)

                if status == "AMBIGUOUS":
                    evidence_pool = raw_external + surviving_local
                else:
                    # INCORRECT: discard all local passages to prevent evidence contamination
                    evidence_pool = raw_external
            else:
                # Deterministic short-circuit when external medical literature yields 0 results
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                sources_str = ", ".join(m.upper() for m in target_mcps)
                res = MedicalCRAGResult(
                    claim=claim,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                    confidence=0.0,
                    consumer_summary="No authoritative medical evidence was found to support or debunk this claim.",
                    clinical_explanation=f"Evaluator flagged evidence as {status} ({getattr(eval_res, 'rationale', '')}). Subsequent external search across {sources_str} for '{eval_res.search_keywords}' returned zero indexed literature.",
                    sources=[],
                    corrective_action_taken=True,
                    execution_time_ms=elapsed_ms,
                    llm_provider_used="none",
                    state=status,
                )
                self.store.cache_claim(claim, res)
                return res

        # Step 4: Grounded Dual-Layer Synthesis
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        final_result = self.synthesizer.synthesize(
            claim=claim,
            evidence_docs=evidence_pool,
            corrective_taken=corrective_taken,
            execution_ms=elapsed_ms,
        )
        final_result.state = status

        # Cache result in SQLite
        self.store.cache_claim(claim, final_result)
        return final_result

    async def verify_async(self, claim: str) -> MedicalCRAGResult:
        """Asynchronously verify a biomedical claim leveraging concurrent MCP searches."""
        return await asyncio.to_thread(self.verify, claim)

    async def stream_verify(self, claim: str) -> AsyncGenerator[dict, None]:
        """Stream real-time CRAG pipeline stage transitions via generator for SSE."""
        claim = claim.strip()
        if not claim:
            yield {"stage": "error", "message": "Health claim must not be empty."}
            return

        t0 = time.perf_counter()

        # Cache check
        cached = self.store.get_cached_claim(claim)
        if cached is not None:
            yield {"stage": "cache_hit", "message": "Result found in SQLite claim cache."}
            yield {"stage": "complete", "result": cached.to_dict()}
            return

        # Stage 1: Retrieve
        yield {"stage": "retrieve", "message": "Executing hybrid dense + lexical retrieval with Weighted RRF..."}
        retrieved_docs = await asyncio.to_thread(self.retriever.retrieve, claim, top_k=self.settings.top_k)

        # Stage 2: Evaluate
        yield {
            "stage": "evaluate",
            "message": f"Evaluating {len(retrieved_docs)} local evidence passages with ternary state grader...",
        }
        eval_res = await asyncio.to_thread(self.evaluator.evaluate, claim, retrieved_docs)
        passage_scores = getattr(eval_res, "passage_scores", {})
        surviving_local = self.evaluator.prune_passages(retrieved_docs, passage_scores, min_score=0.50)

        status = getattr(eval_res, "status", None)
        if status is None:
            status = "CORRECT" if getattr(eval_res, "is_sufficient", False) else "AMBIGUOUS"

        target_mcps = getattr(eval_res, "target_mcps", None)
        if not target_mcps:
            src = getattr(eval_res, "search_source", "pubmed")
            target_mcps = [src] if src else ["pubmed"]

        corrective_taken = False
        evidence_pool: List[Document] = []

        if status == "CORRECT":
            yield {"stage": "evaluate", "message": "Local evidence sufficient (CORRECT). Bypassing external search."}
            evidence_pool = surviving_local if surviving_local else retrieved_docs
        else:
            corrective_taken = True
            sources_label = ", ".join(target_mcps).upper()
            yield {
                "stage": "mcp_correct",
                "message": f"State is {status}. Querying {sources_label} for '{eval_res.search_keywords}'...",
            }

            if len(target_mcps) == 1:
                raw_external = await asyncio.to_thread(self.mcp.search, target_mcps[0], eval_res.search_keywords, 3)
            else:
                raw_external = await self.mcp.search_all_async(
                    target_mcps,
                    eval_res.search_keywords,
                    max_results_per_source=3,
                )

            if raw_external:
                gated_docs = gate_mcp_documents(raw_external, eval_res.search_keywords, min_relevance=0.50, min_length=15)
                docs_to_ingest = gated_docs if gated_docs else raw_external
                await asyncio.to_thread(self.store.upsert_documents, docs_to_ingest)
                yield {
                    "stage": "mcp_correct",
                    "message": f"Indexed {len(docs_to_ingest)} external documents into ChromaDB & SQLite.",
                }

                if status == "AMBIGUOUS":
                    evidence_pool = raw_external + surviving_local
                else:
                    evidence_pool = raw_external
            else:
                if status == "AMBIGUOUS" and surviving_local:
                    evidence_pool = surviving_local
                else:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    short_circ = MedicalCRAGResult(
                        claim=claim,
                        verdict=Verdict.INSUFFICIENT_EVIDENCE.value,
                        confidence=0.0,
                        consumer_summary="No authoritative medical evidence was found to verify this claim.",
                        clinical_explanation=f"Evaluator flagged evidence as {status}. External search returned 0 records.",
                        sources=[],
                        corrective_action_taken=True,
                        execution_time_ms=elapsed_ms,
                        llm_provider_used="none",
                        state=status,
                    )
                    self.store.cache_claim(claim, short_circ)
                    yield {"stage": "complete", "result": short_circ.to_dict()}
                    return

        # Stage 4: Synthesize
        yield {"stage": "synthesize", "message": "Synthesizing grounded verdict with deterministic citation verifier..."}
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        final_result = await asyncio.to_thread(
            self.synthesizer.synthesize,
            claim,
            evidence_pool,
            corrective_taken,
            elapsed_ms,
        )
        final_result.state = status
        self.store.cache_claim(claim, final_result)

        yield {"stage": "complete", "result": final_result.to_dict()}
