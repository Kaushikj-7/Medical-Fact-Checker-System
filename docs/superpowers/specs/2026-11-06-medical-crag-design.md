# Medical Corrective RAG (CRAG) Architecture Design

**Date:** 2026-11-06  
**Status:** In Review  
**Topic:** Standalone, Lean Corrective RAG for Medical Claim Verification

---

## 1. Intent & The Why

Medical claims and health advice spread rapidly, often with dangerous consequences. Verifying health claims requires high accuracy, clear citations of authoritative medical literature (e.g., peer-reviewed research, CDC, WHO, FDA), and zero tolerance for hallucinations.

Standard RAG systems fail when the local knowledge base lacks relevant information, often hallucinating or providing misleading answers based on tangential passages.

This architecture defines a **standalone, lean Corrective RAG (CRAG)** system. When local evidence is adequate, it answers directly. When local evidence is insufficient, it automatically intervenes, executes an authoritative external search against the NCBI PubMed API, indexes the new evidence into a local ChromaDB store, and delivers a grounded verdict with transparent clinical citations.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Predictable 4-Step Pipeline**: Sequential and transparent execution: `Retrieve` → `Evaluate` → `Correct (if needed)` → `Generate`.
- **Lightweight Local Storage**: Use persistent **ChromaDB** with hybrid retrieval (semantic similarity + BM25 keyword matching).
- **Self-Expanding Medical Corpus**: Automatically index newly fetched PubMed abstracts into ChromaDB so the local knowledge base expands organically with usage.
- **Resilient Multi-Provider LLM Gateway**: Support **Google Gemini** as the primary provider with automatic failover to **OpenRouter** (e.g., Llama-3.3-70B or Claude) on rate limits, errors, or timeouts.
- **Dual-Layer Output**: Deliver a clear, empathetic plain-language verdict for everyday readers, accompanied by a structured clinical proof chain (PubMed IDs, URLs, and evidence snippets) for verification.
- **Zero Framework Bloat**: Pure Python implementation with minimal external dependencies; no heavy orchestration frameworks.

### Non-Goals
- **No Knowledge Graphs**: Explicitly avoids graph databases (Neo4j, GraphRAG, RDF triples), keeping storage strictly document- and vector-based.
- **No Complex Multi-Agent Swarms**: Avoids distributed task queues, background workers, or autonomous multi-agent debates that add latency and failure modes.
- **Not a Diagnostic Medical Device**: Strictly an informational and educational fact-checking verification tool, not a clinical diagnostic system.

### Anti-Goals
- **No Ungrounded Hallucinations**: If evidence is missing and external PubMed search yields no relevant results, the system must output `INSUFFICIENT_EVIDENCE` or `INCONCLUSIVE` rather than guessing.
- **No Opaque Decisions**: Every claim verdict must link directly to the source documents used in synthesis.

---

## 3. Constraints & Prerequisites

- **Environment**: Python 3.10+ on Windows / Linux / macOS.
- **API Keys**:
  - `GEMINI_API_KEY`: Primary generation and grading model (e.g., Gemini 1.5 Flash).
  - `OPENROUTER_API_KEY`: Automatic fallback model (e.g., `meta-llama/llama-3.3-70b-instruct`).
- **External Network Access**: NCBI PubMed E-utilities (`eutils.ncbi.nlm.nih.gov`) for live medical abstracts.

---

## 4. Approaches Considered

| Approach | Description | Trade-offs | Decision |
|---|---|---|---|
| **Approach 1: Lean 4-Stage Sequential CRAG** | Deterministic pipeline: ChromaDB lookup → Gemini evaluation check → PubMed fallback & auto-index → Final grounded synthesis. | Clean, fast (1-2 LLM calls), zero framework baggage, trivial to test and debug. | **Selected** (User approved) |
| **Approach 2: Tool-Calling Agent RAG** | Single LLM prompt with `search_chroma` and `search_pubmed` tools. | Fewer pipeline stages, but prone to tool-calling loops, higher latency, and inconsistent behavior across different LLM providers. | Rejected |
| **Approach 3: Complex Multi-Agent / Graph RAG** | Swarm of agents debating claims, integrated with biomedical Knowledge Graphs. | Extreme complexity, slow response times, fragile dependencies; rejected by user preference. | Rejected |

---

## 5. Detailed Architecture & Component Specifications

```
                     [User Health Claim]
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │ 1. RETRIEVE                               │
        │    - ChromaDB Semantic Search             │
        │    - BM25 Keyword Search                  │
        │    - Reciprocal Rank Fusion (RRF) Merge   │
        └─────────────────────┬─────────────────────┘
                              │ Top 3-5 Passages
                              ▼
        ┌───────────────────────────────────────────┐
        │ 2. EVALUATE (Corrective Core)             │
        │    - Gemini evaluates sufficiency:        │
        │      "Does evidence clearly address       │
        │       and resolve the claim?"             │
        └──────────────┬─────────────┬──────────────┘
                       │             │
             [SUFFICIENT]           [INSUFFICIENT]
                       │             │
                       │             ▼
                       │    ┌───────────────────────────────────────────┐
                       │    │ 3. CORRECT (MCP External Tools)           │
                       │    │    - PubMed MCP (Clinical research)       │
                       │    │    - FDA MCP (Drug labels/warnings)       │
                       │    │    - WHO MCP (Outbreaks/Fact sheets)      │
                       │    │    - Upsert newly found docs to ChromaDB  │
                       │    │    - Append to candidate evidence pool    │
                       │    └─────────────┬─────────────────────────────┘
                       │                  │
                       ▼                  ▼
        ┌───────────────────────────────────────────┐
        │ 4. GENERATE                               │
        │    - Multi-Provider LLM Gateway:          │
        │      Primary: Google Gemini               │
        │      Fallback: OpenRouter                 │
        │    - Dual-Layer Structured Output:        │
        │      * Patient-friendly Verdict & Summary │
        │      * Clinical Proof Chain & Citations   │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                      [Pipeline Result]
```

### 5.1 Storage Layer (`storage.py`)
- **Technology**: `chromadb.PersistentClient` stored in `./data/chroma_db`.
- **Collection**: `medical_evidence`
- **Document Metadata**: `{"pmid": str, "title": str, "source": str, "url": str, "published": str}`
- **Pre-seeding**: Bundled starter corpus with common medical advisories (e.g., CDC/WHO guidelines, FDA alerts) loaded on initialization if the collection is empty.
- **Deduplication**: Ingestion uses `pmid` or unique document hashes as IDs to prevent duplicate entries.

### 5.2 Retrieval Layer (`retriever.py`)
- **Hybrid Search**:
  1. Dense vector query against ChromaDB.
  2. Sparse keyword matching via an in-memory BM25 index over the local corpus.
  3. Merge using Reciprocal Rank Fusion ($RRF = \sum \frac{1}{60 + \text{rank}}$) to maintain exact drug names and chemical dosages.

### 5.3 Evaluator Layer (`evaluator.py`)
- Sends the claim and top candidate passages to the fast LLM (Gemini 1.5 Flash).
- Prompt contract: Strict JSON response:
  ```json
  {
    "is_sufficient": true | false,
    "confidence": 0.0 - 1.0,
    "rationale": "Brief reason why evidence is adequate or inadequate",
    "search_source": "pubmed" | "fda" | "who",
    "search_keywords": "Optimized search terms (e.g., 'metformin lactic acidosis')"
  }
  ```
- **Builder Call**: The evaluator identifies the ideal authoritative medical source domain (`pubmed` for biomedical studies, `fda` for drug indications/warnings, `who` for vaccines/outbreaks) and extracts search keywords in this single call.
- If `is_sufficient == false` or `confidence < 0.65`, triggers the **Correct** stage targeting the recommended MCP tool.

### 5.4 Corrective Search Layer via MCP Tools (`mcp_tools/`)
- Encapsulates authoritative medical external search into Model Context Protocol (MCP) compliant tools:
  - **`mcp_pubmed`**: NCBI PubMed E-utilities (`esearch` + `efetch`) for peer-reviewed trial abstracts.
  - **`mcp_fda`**: OpenFDA endpoint for official FDA drug labels, contraindications, and recalls.
  - **`mcp_who`**: World Health Organization public health API for outbreak news and disease fact sheets.
- Formats retrieved results into standard `Document` objects.
- **Auto-Ingestion**: Passes fetched documents to `storage.py` to be saved into ChromaDB with unique source IDs.
- **Short-Circuiting**: If the selected MCP tool returns 0 results or encounters an error, the pipeline immediately short-circuits to an `INSUFFICIENT_EVIDENCE` verdict without calling the Synthesizer.
- **Evidence Pool Assembly**: Fetched MCP documents become the primary evidence (top 2–3 passages), retaining at most the top-1 local passage as background (capped at 4 total passages).

### 5.5 Multi-Provider LLM Gateway (`llm.py`)
- Unified interface: `generate_completion(prompt, system_prompt=..., json_mode=False)`
- **Execution Flow**:
  1. Attempt primary call using Google Gemini API (`gemini-1.5-flash` or `gemini-1.5-pro`).
  2. If Gemini raises rate limit (`429`), network error, or timeout, catch the exception, log a warning, and immediately retry using the OpenRouter API (`https://openrouter.ai/api/v1` with `meta-llama/llama-3.3-70b-instruct` or configured model).
  3. If both fail, raise a descriptive runtime error.

### 5.6 Synthesis Layer (`synthesizer.py`)
- Generates a grounded verdict strictly constrained to the assembled evidence.
- Prompt enforces:
  - **Verdict**: One of `SUPPORTED`, `REFUTED`, `MISLEADING`, or `INSUFFICIENT_EVIDENCE`.
  - **Confidence**: 0.0 to 1.0 assessing certainty in the final verdict based on the retrieved evidence.
  - **Consumer Summary**: 2-3 empathetic, jargon-free sentences answering the user's concern.
  - **Clinical Proof Chain**: Bullet points quoting the exact evidence with explicit citations `[1]`, `[2]`.
  - **Sources**: List of sources with title, URL (PubMed link), and PMID.
- **Builder Call**: `confidence` in `MedicalCRAGResult` originates directly from the Synthesizer's certainty in the claim verdict. The Evaluator's local sufficiency score is preserved separately in the execution trace for debugging.

---

## 6. Output Data Contract

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SourceCitation:
    id: str
    title: str
    url: str
    source: str
    snippet: str

@dataclass
class MedicalCRAGResult:
    claim: str
    verdict: str                  # SUPPORTED | REFUTED | MISLEADING | INSUFFICIENT_EVIDENCE
    confidence: float             # 0.0 to 1.0
    consumer_summary: str         # Plain English for patients
    clinical_explanation: str     # Biomedical details with citations
    sources: List[SourceCitation] # Authoritative sources used
    corrective_action_taken: bool # True if external PubMed search was triggered
    execution_time_ms: int
    llm_provider_used: str        # "gemini" or "openrouter"
```

---

## 7. What's Left to the Builder

- Selection of default local embedding function for ChromaDB (e.g., standard SentenceTransformers `all-MiniLM-L6-v2` or Chroma's default onnx embedder).
- Packaging as a clean installable package (`medical_crag`) with a simple CLI entrypoint (`python -m medical_crag.cli "vitamin C cures cancer"`).
- Providing an optional lightweight FastAPI wrapper (`GET /health`, `POST /verify`) for integration into external dashboards or browser extensions.
