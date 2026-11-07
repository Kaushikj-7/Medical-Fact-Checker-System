# Medical Corrective RAG (CRAG)

> **High-Precision Biomedical Claim Verification System with Dynamic Corrective Search & Grounded Proof Chains**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-green.svg)](https://www.trychroma.com/)
[![NCBI PubMed](https://img.shields.io/badge/External%20Source-NCBI%20PubMed-red.svg)](https://pubmed.ncbi.nlm.nih.gov/)
[![openFDA](https://img.shields.io/badge/External%20Source-openFDA-orange.svg)](https://open.fda.gov/)
[![WHO DON](https://img.shields.io/badge/External%20Source-WHO%20DON-teal.svg)](https://www.who.int/emergencies/disease-outbreak-news)
[![Test Suite](https://img.shields.io/badge/Tests-233%20Passed-brightgreen.svg)]()

---

## Executive Summary

Standard Retrieval-Augmented Generation (RAG) fails in clinical healthcare because it operates on a closed, static corpus and blindly trusts retrieved documents—even when they are irrelevant, partial, or out-of-date. Conversely, complex GraphRAG and multi-agent swarms introduce excessive latency, non-deterministic loops, and fragile ontology schemas.

**Medical Corrective RAG (CRAG)** solves this by introducing an autonomous, high-speed, self-correcting 4-stage pipeline that:
1. **Performs Synchronized Hybrid Retrieval** (BM25 Lexical + ChromaDB Cosine Embeddings via Dynamic Weighted Reciprocal Rank Fusion).
2. **Dynamically Evaluates Retrieved Documents** using a Ternary CRAG state machine (CORRECT / AMBIGUOUS / INCORRECT) with per-passage relevance grading and pruning.
3. **Triggers Concurrent External Medical Search** via authoritative MCP tools (**NCBI PubMed E-Utilities XML**, **openFDA Drug Labels**, **WHO Disease Outbreak News**) when local store knowledge is insufficient, executing gated anti-poisoning write-back to ChromaDB and SQLite.
4. **Synthesizes Grounded Verifications** using a strict dual-layer output (Patient Summary in plain language and an evidence-backed Clinical Proof Chain) validated by a deterministic post-processing citation verifier.

---

## ByteByteGo-Style System Architecture

![Medical Corrective RAG ByteByteGo Architecture](docs/images/medical_crag_bytebytego_architecture.png)

### End-to-End Sequence & Decision Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Doctor / Patient
    participant CLI as CLI / API Gateway
    participant Ret as Hybrid Retriever (ChromaDB + BM25)
    participant Eval as Dynamic Evaluator (LLM)
    participant MCP as MCP Medical Search (PubMed / openFDA / WHO)
    participant Syn as Grounded Synthesizer (LLM Gateway)

    User->>CLI: Submit Medical Claim
    CLI->>Ret: Step 1: Synchronized Hybrid Search (Weighted RRF)
    Ret-->>Eval: Ranked Document Candidates
    Eval->>Eval: Step 2: Per-Passage Grading & Ternary State Routing

    alt State == CORRECT (Score >= 0.80, >= 2 surviving passages)
        Eval-->>Syn: Sufficient Local Context
    else State == AMBIGUOUS (0.50 <= Score < 0.80)
        Eval->>MCP: Step 3: Trigger Multi-MCP Search (search_keywords)
        MCP->>Ret: Gated Ingestion to ChromaDB & SQLite
        MCP-->>Syn: Combined External + Surviving Local Evidence
    else State == INCORRECT (Score < 0.50)
        Eval->>MCP: Step 3: Trigger Multi-MCP Search (search_keywords)
        MCP->>Ret: Gated Ingestion to ChromaDB & SQLite
        MCP-->>Syn: Uncontaminated External Evidence (Local Discarded)
    end

    Syn->>Syn: Step 4: Strict Deterministic Citation Verification
    Syn-->>CLI: Return MedicalCRAGResult
    CLI-->>User: Patient Summary + Clinical Proof Chain + Verified Citations
```

---

## Core Architectural Pillars

### 1. Synchronized Storage & SQLite Cache
- **Dual-Storage Consistency**: Operates ChromaDB persistent vector database alongside an atomic SQLite store (`documents` and `claim_cache`).
- **SHA-256 Content Deduplication**: Computes deterministic content hashes to reject duplicate insertions across batches and ingest runs.
- **Immediate BM25 Synchronization**: Every call to `store.upsert_documents()` persists to SQLite, ChromaDB, and triggers an incremental rebuild of the in-memory BM25 index, guaranteeing zero split-brain drift between vector and lexical representations.
- **TTL Claim Cache**: Caches claim verifications with configurable TTL (default: 24h) to eliminate redundant LLM calls.

### 2. Dynamic Weighted Reciprocal Rank Fusion (Weighted RRF)
- Combines dense and sparse ranks with clinical token detection:
  $$\text{Score}(d) = \frac{\alpha_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{\alpha_{\text{sparse}}}{60 + \text{rank}_{\text{sparse}}(d)}$$
- **Lexical Entity Boosting**: Detects exact clinical tokens (dosages like `mg`/`mcg`, `PMID`, `NCT\d+`, regulatory agencies `FDA`/`WHO`, and active drug suffixes like `-statin`, `-mab`, `-nib`).
- Dynamically shifts $(\alpha_{\text{dense}}, \alpha_{\text{sparse}})$ from $(0.5, 0.5)$ to $(0.3, 0.7)$ so exact pharmacological identifiers are never drowned out by dense cosine approximations.

### 3. Ternary CRAG State Machine & Passage Grading
- Evaluates candidate passages independently with relevance scores $[0.0, 1.0]$.
- **Pruning**: Discards local chunks with score $< 0.50$.
- **State Machine Transitions**:
  - `CORRECT`: $\max(\text{score}) \ge 0.80$ and $\ge 2$ passages survive pruning. Synthesizes directly from local evidence.
  - `AMBIGUOUS`: $0.50 \le \max(\text{score}) < 0.80$. Retains surviving passages and triggers concurrent MCP search.
  - `INCORRECT`: $\max(\text{score}) < 0.50$. Discards all local passages to prevent evidence contamination; triggers MCP search.
- **Multi-MCP Target Selection**: Evaluates claim context to select `target_mcps` from `["pubmed", "fda", "who"]`.

### 4. Parallel Async MCP Fan-Out & Gated Write-Back
- **Concurrent Concurrency**: Dispatches tool searches concurrently using `asyncio.gather(..., return_exceptions=True)` with per-tool timeouts and exponential backoff retry.
- **Gated Ingestion**: `gate_mcp_documents()` validates external documents for length ($\ge 15$ characters) and keyword overlap ($\ge 0.50$) before admitting them to the ChromaDB/SQLite store, preventing vector index poisoning.

### 5. Multi-Provider LLM Gateway with Instant Failover
- **OpenRouter Primary**: Powered by `meta-llama/llama-3.3-70b-instruct` with JSON schema enforcement.
- **Google Gemini Fallback**: Native REST client for `gemini-1.5-flash` with zero third-party SDK lock-in.
- **Zero-Failure JSON Scanner**: Balanced-bracket parser extracts JSON objects even when preambles or LaTeX math delimiters are present.
- **Strict Budget Gate**: Maximum 1 to 2 LLM calls per claim verification.

### 6. Grounded Dual-Layer Verification & Citation Verifier
- **Patient Summary**: Plain-language assessment for patients and health consumers.
- **Clinical Proof Chain**: Biomedical explanation detailing pharmacodynamics, hazard ratios, and clinical endpoints.
- **Deterministic Citation Verifier**: Scans bracket citations `[1]`, `[2]`, or direct IDs `[pmid-38785209]`, validates existence against `valid_doc_ids`, flags hallucinations, and exposes `citation_precision` and `verified_citations`.

---

## Live Verification Output Examples

### Example 1: Local Knowledge Base Sufficient
```bash
python -m medical_crag "Turmeric reduces inflammation in arthritis"
```
```
Processing claim through Corrective RAG: 'Turmeric reduces inflammation in arthritis'...

=================================================================
MEDICAL CORRECTIVE RAG VERIFICATION
=================================================================
Claim: "Turmeric reduces inflammation in arthritis"
Verdict: SUPPORTED (Confidence: 80%)
Execution Time: 2640 ms | Provider: openrouter
Corrective Action: Not Needed (Local Store Sufficient)
-----------------------------------------------------------------
PATIENT SUMMARY (Plain Language):
  Turmeric has been shown to reduce inflammation in arthritis, particularly 
  in knee osteoarthritis, by significantly reducing pain. This natural remedy 
  may be a helpful addition to traditional treatments. However, it is essential 
  to consult with a healthcare professional before using turmeric for medicinal purposes.

CLINICAL PROOF CHAIN (Biomedical Evidence):
  The systematic review and network meta-analysis of randomized controlled 
  trials [1] demonstrated that turmeric preparations significantly reduced pain 
  in knee osteoarthritis, with a mean difference of -4.01 in WOMAC pain reduction. 
  This suggests that turmeric may be an effective adjunctive treatment for arthritis.

AUTHORITATIVE SOURCES CITED:
  [1] Effect of turmeric products on knee osteoarthritis: a systematic review and network meta-analysis.
      Source: PubMed | URL: https://pubmed.ncbi.nlm.nih.gov/40731001/
      Quote: "Turmeric has traditionally been used to treat various inflammatory conditions, including knee osteoarthritis (OA)..."
=================================================================
```

### Example 2: Corrective External Search Triggered
```bash
python -m medical_crag "Semaglutide reduces major cardiovascular events in patients with type 2 diabetes"
```
```
Processing claim through Corrective RAG: 'Semaglutide reduces major cardiovascular events in patients with type 2 diabetes'...

=================================================================
MEDICAL CORRECTIVE RAG VERIFICATION
=================================================================
Claim: "Semaglutide reduces major cardiovascular events in patients with type 2 diabetes"
Verdict: SUPPORTED (Confidence: 90%)
Execution Time: 10609 ms | Provider: openrouter
Corrective Action: Triggered (External Search & Ingested)
-----------------------------------------------------------------
PATIENT SUMMARY (Plain Language):
  Semaglutide has been shown to reduce the risk of major cardiovascular events 
  in patients with type 2 diabetes, providing a beneficial effect on heart health. 
  This medication can be an important part of managing diabetes and reducing 
  the risk of heart problems.

CLINICAL PROOF CHAIN (Biomedical Evidence):
  The evidence from clinical trials [1, 2] demonstrates that semaglutide reduces 
  the risk of major cardiovascular events, including cardiovascular death, nonfatal 
  myocardial infarction, or nonfatal stroke, in patients with type 2 diabetes. 
  The hazard ratio for these events was significantly lower in the semaglutide group.

AUTHORITATIVE SOURCES CITED:
  [1] Effects of Semaglutide on Chronic Kidney Disease in Patients with Type 2 Diabetes.
      Source: PubMed | URL: https://pubmed.ncbi.nlm.nih.gov/38785209/
  [2] Semaglutide and Cardiovascular Outcomes in Patients with Type 2 Diabetes.
      Source: PubMed | URL: https://pubmed.ncbi.nlm.nih.gov/27633186/
=================================================================
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Internet access for live PubMed/FDA/WHO API calls

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Kaushikj-7/Medical-Fact-Checker-System.git
cd Medical-Fact-Checker-System
pip install -r requirements.txt
```

*(Or install editable package directly)*:
```bash
pip install -e .
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
PRIMARY_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

# Optional Fallback:
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
GEMINI_MODEL=gemini-1.5-flash
```

---

## Usage

### Command Line Interface (CLI)
Verify any biomedical or health claim directly from your terminal:
```bash
python -m medical_crag "Grapefruit juice interacts dangerously with statin medications"
```

### Programmatic Python API
```python
from medical_crag.config import Settings
from medical_crag.pipeline import MedicalCRAGPipeline

settings = Settings.from_env()
pipeline = MedicalCRAGPipeline(settings=settings)

result = pipeline.verify("Metformin carries an FDA boxed warning for lactic acidosis.")

print(f"Verdict: {result.verdict} (Confidence: {result.confidence:.0%})")
print(f"Patient Summary: {result.patient_summary}")
print(f"Clinical Proof Chain: {result.clinical_proof_chain}")
print(f"Citation Precision: {result.citation_precision:.1%}")

for source in result.sources:
    print(f"- [{source.source}] {source.title}: {source.url}")
```

### FastAPI & Server-Sent Events (SSE) Streaming
Run the production server with auto-reload:
```bash
uvicorn medical_crag.api:app --host 0.0.0.0 --port 8000 --reload
```
Available Endpoints:
- `POST /api/v1/verify`: JSON verification payload `{"claim": "..."}`
- `GET /api/v1/verify/stream`: Real-time stage transition SSE stream (`retrieve -> evaluate -> mcp_correct -> synthesize -> complete`)
- `POST /api/v1/verify/image`: Multimodal claim extraction and verification from uploaded screenshot/image
- `GET /health`: Service health and readiness check

---

## Testing & Verification

The test suite enforces clean architecture, TDD, zero mock cheating, and full end-to-end reliability.

### Run Full Test Suite (233 tests)
```bash
python -m pytest tests/ -v
```

---

## Repository Structure

```
Medical-Fact-Checker-System/
├── docs/
│   ├── ARCHITECTURE.md      # Comprehensive technical architecture specification
│   └── images/
│       └── medical_crag_bytebytego_architecture.png  # ByteByteGo architecture diagram
├── medical_crag/
│   ├── __init__.py          # Package initialization and exports
│   ├── __main__.py          # Entrypoint for python -m medical_crag
│   ├── api.py               # FastAPI REST, SSE streaming, and image upload endpoints
│   ├── cli.py               # Formatted CLI presentation & banner printer
│   ├── config.py            # Environment validation, masking & multi-path loader
│   ├── evaluator.py         # Ternary state machine, per-passage grading & multi-MCP router
│   ├── llm.py               # Multi-provider LLM gateway (OpenRouter + Gemini Vision)
│   ├── mcp_tools.py         # Authoritative medical search clients (PubMed, openFDA, WHO)
│   ├── pipeline.py          # 4-stage Corrective RAG pipeline orchestrator with caching
│   ├── retriever.py         # Dynamic Weighted BM25 + ChromaDB Reciprocal Rank Fusion
│   ├── storage.py           # Synchronized ChromaDB + SQLite store & claim TTL cache
│   ├── synthesizer.py       # Dual-layer grounded clinical synthesis & citation verifier
│   └── types.py             # Strongly typed Pydantic/dataclass models
├── tests/
│   ├── test_medical_crag/   # Isolated core component tests
│   ├── test_crag_state_machine.py # Ternary CRAG state machine & passage grading
│   ├── test_citation_verifier.py  # Deterministic citation grounding & verification
│   ├── test_hybrid_rrf_sync.py    # Synchronized SQLite/ChromaDB & Weighted RRF
│   ├── test_api_endpoints.py      # FastAPI REST and SSE stream tests
│   ├── test_live_crag.py          # 42 live CRAG validation tests
│   └── test_adversarial_llm.py    # Stress testing and JSON resilience
├── .env.example             # Template environment configuration
├── .gitignore               # Ignored secrets, database & bytecode files
├── Dockerfile               # Containerized production runtime
├── pyproject.toml           # Modern package configuration (PEP 517/621)
├── requirements.txt         # Pinned production dependencies
└── README.md                # Project documentation
```

---

## License & Medical Disclaimer

This project is open-source under the MIT License.

> **Disclaimer**: *Medical Corrective RAG is a clinical decision-support and health research tool designed for research and educational purposes. It does not provide formal medical advice, diagnosis, or treatment plans. Clinical decisions should always be made by a licensed healthcare provider.*
