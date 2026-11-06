# Medical Corrective RAG (CRAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, non-complex medical Corrective RAG (CRAG) system with ChromaDB storage, multi-provider LLM failover (Gemini + OpenRouter), MCP medical tools (PubMed, FDA, WHO), and grounded dual-layer verification.

**Architecture:** A 4-stage sequential pipeline: (1) Hybrid Retrieve (ChromaDB + BM25), (2) Fast Evaluator check, (3) Corrective MCP retrieval (PubMed/FDA/WHO) with auto-indexing into ChromaDB, and (4) Grounded dual-layer synthesis with failover.

**Tech Stack:** Python 3.12, ChromaDB, Google Generative AI (Gemini), OpenRouter (HTTPX), aiohttp, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-medical-crag-design.md`

## Global Constraints
- Python 3.10+ compatible (running on Python 3.12).
- Zero knowledge graphs (no GraphRAG / Neo4j / graph databases).
- Hard budget: strictly 1–2 LLM calls per claim verification.
- Offline-resilient and deterministic fallback if external networks fail.
- All code covered by unit tests with mocked network/API boundaries.

## Review Focus
1. **Empty external search response**: When PubMed/FDA/WHO returns 0 hits, pipeline must cleanly output `INSUFFICIENT_EVIDENCE` without hallucinating.
2. **LLM provider failover**: When Gemini raises rate-limit (429) or network error, OpenRouter must seamlessly fulfill the request.
3. **ChromaDB deduplication**: Ingesting fetched PubMed papers multiple times must never duplicate records in the vector database.
4. **Keyword extraction robustness**: Evaluator query extraction must format valid boolean/keyword queries for PubMed e-utilities.
5. **Exact citation grounding**: Synthesizer must strictly map citations `[1]`, `[2]` to returned source items.

---

### Task 1: Core Data Models & Configuration

**Files:**
- Create: `medical_crag/__init__.py`
- Create: `medical_crag/types.py`
- Create: `medical_crag/config.py`
- Test: `tests/test_medical_crag/test_config_and_types.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_config_and_types.py` verifying data class schemas (`Document`, `SourceCitation`, `MedicalCRAGResult`, `EvaluationResult`) and `Settings.from_env()`.
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_config_and_types.py`
- [ ] Implement `medical_crag/types.py` and `medical_crag/config.py`.
- [ ] Run test and confirm it passes: `python -m pytest tests/test_medical_crag/test_config_and_types.py`
- [ ] Commit changes: `git add medical_crag/ tests/ && git commit -m "feat(crag): add core types and configuration"`

---

### Task 2: Storage & Hybrid Retrieval Layer

**Files:**
- Create: `medical_crag/storage.py`
- Create: `medical_crag/retriever.py`
- Test: `tests/test_medical_crag/test_storage_and_retrieval.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_storage_and_retrieval.py` testing ChromaDB collection creation, upserting with deduplication, BM25 scoring, and RRF merging.
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_storage_and_retrieval.py`
- [ ] Implement `medical_crag/storage.py` (persistent ChromaDB client and starter seed helper) and `medical_crag/retriever.py` (hybrid Chroma + BM25 with RRF).
- [ ] Run test and confirm it passes: `python -m pytest tests/test_medical_crag/test_storage_and_retrieval.py`
- [ ] Commit changes: `git add medical_crag/storage.py medical_crag/retriever.py tests/ && git commit -m "feat(crag): implement ChromaDB storage and hybrid retriever"`

---

### Task 3: Multi-Provider LLM Gateway

**Files:**
- Create: `medical_crag/llm.py`
- Test: `tests/test_medical_crag/test_llm_gateway.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_llm_gateway.py` mocking Gemini primary success, Gemini failure with OpenRouter fallback, and both failing with clear exception.
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_llm_gateway.py`
- [ ] Implement `medical_crag/llm.py` supporting Gemini and OpenRouter (with httpx client, structured JSON parsing, and retry/fallback logic).
- [ ] Run test and confirm it passes: `python -m pytest tests/test_medical_crag/test_llm_gateway.py`
- [ ] Commit changes: `git add medical_crag/llm.py tests/ && git commit -m "feat(crag): implement multi-provider LLM gateway with failover"`

---

### Task 4: Corrective MCP Tools (PubMed, FDA, WHO)

**Files:**
- Create: `medical_crag/mcp_tools.py`
- Test: `tests/test_medical_crag/test_mcp_tools.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_mcp_tools.py` testing PubMed e-utilities parsing, FDA drug label extraction, and WHO alert queries with mocked HTTP responses.
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_mcp_tools.py`
- [ ] Implement `medical_crag/mcp_tools.py` containing `search_pubmed()`, `search_fda()`, and `search_who()`.
- [ ] Run test and confirm it passes: `python -m pytest tests/test_medical_crag/test_mcp_tools.py`
- [ ] Commit changes: `git add medical_crag/mcp_tools.py tests/ && git commit -m "feat(crag): add MCP medical search tools"`

---

### Task 5: Evaluator & Synthesizer

**Files:**
- Create: `medical_crag/evaluator.py`
- Create: `medical_crag/synthesizer.py`
- Test: `tests/test_medical_crag/test_evaluator_and_synthesizer.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_evaluator_and_synthesizer.py` verifying evaluation JSON extraction (`is_sufficient`, `search_source`, `search_keywords`) and synthesis schema generation (`MedicalCRAGResult`).
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_evaluator_and_synthesizer.py`
- [ ] Implement `medical_crag/evaluator.py` and `medical_crag/synthesizer.py`.
- [ ] Run test and confirm it passes: `python -m pytest tests/test_medical_crag/test_evaluator_and_synthesizer.py`
- [ ] Commit changes: `git add medical_crag/evaluator.py medical_crag/synthesizer.py tests/ && git commit -m "feat(crag): implement evaluator and synthesizer"`

---

### Task 6: Pipeline Orchestration, CLI, & End-to-End Verification

**Files:**
- Create: `medical_crag/pipeline.py`
- Create: `medical_crag/cli.py`
- Test: `tests/test_medical_crag/test_pipeline_e2e.py`

**Steps:**
- [ ] Write failing test in `tests/test_medical_crag/test_pipeline_e2e.py` testing both sufficient local evidence path (skip external) and insufficient evidence path (triggers MCP + auto-ingest into ChromaDB).
- [ ] Run test to watch it fail: `python -m pytest tests/test_medical_crag/test_pipeline_e2e.py`
- [ ] Implement `medical_crag/pipeline.py` (`MedicalCRAGPipeline.verify()`) and `medical_crag/cli.py`.
- [ ] Run full test suite: `python -m pytest tests/test_medical_crag/`
- [ ] Commit changes: `git add medical_crag/pipeline.py medical_crag/cli.py tests/ && git commit -m "feat(crag): wire pipeline and CLI with end-to-end test"`
