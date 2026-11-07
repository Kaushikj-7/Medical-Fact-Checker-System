# Project: Medical Corrective RAG (CRAG) Capability Showcase

## Architecture
Medical Corrective RAG (CRAG) evaluates health claims using a multi-stage retrieval, evaluation, corrective search, and synthesis pipeline:
1. **Query Processing & Retrieval**: Retrieves local documents from ChromaDB vector store.
2. **Sufficiency Evaluation**: Evaluates local evidence using LLMGateway. If sufficient, proceeds to synthesis. If ambiguous or insufficient, triggers corrective search.
3. **Corrective Search (External MCP Tools)**: Queries live medical APIs (PubMed E-utilities, openFDA Drug Labels, WHO Outbreak News / Guidelines) using dynamically generated keywords.
4. **Evidence Synthesis**: Synthesizes verified clinical evidence, assesses verdict (`SUPPORTED`, `REFUTED`, `INSUFFICIENT_EVIDENCE`), computes confidence, provides consumer summary, clinical explanation, and grounded citations.
5. **LLM Gateway**: Dynamic multi-provider integration (OpenRouter with `meta-llama/llama-3.3-70b-instruct`, Google Gemini) with resilient JSON parsing.

```
User Claim
   │
   ▼
[Local Retriever] ──(documents)──► [Evaluator]
                                        │
                                   is_sufficient?
                                  /              \
                             YES /                \ NO / AMBIGUOUS
                                /                  \
                               ▼                    ▼
                        [Synthesizer] ◄── [Corrective Search]
                                          (PubMed / FDA / WHO)
                               │
                               ▼
                      MedicalCRAGResult
```

## Feature Inventory
Every feature from user requirements and the survey phase is enumerated here with its assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Multi-Variable Env Key Extraction | Support `OPENROUTER_API_KEY` and `OPEN_ROUTER_API_KEY`, `GEMINI_API_KEY` and `GOOGLE_API_KEY`, stripping whitespace | M1 | R1, Survey 1, Survey 2 |
| 2 | Robust Configuration Loading | Ensure `.env` is loaded reliably whether executed from root or subpackage; clear error reporting | M1 | R1, Survey 1 |
| 3 | OpenRouter Live API Compliance | Compliant headers (`Authorization`, `HTTP-Referer`, `X-Title`), payload, and `response_format={"type": "json_object"}` | M2 | R2, Survey 2 |
| 4 | Google Gemini API Compliance | REST payload structure, header/query key auth, generationConfig json mode, 400/403/429 error handling | M2 | R2, Survey 2 |
| 5 | Resilient Multi-Stage JSON Parser | `clean_json_text` with markdown code block stripping, outermost `{...}` regex extraction, trailing comma cleanup | M2 | R2, R3, Survey 3 |
| 6 | PubMed E-utilities Live Search | Live `esearch.fcgi` & `efetch.fcgi` integration, NCBI tool/email compliance, full abstract extraction without 600-char cutoff | M3 | R4, Survey 1, Survey 2 |
| 7 | openFDA Live Drug Search | Live `api.fda.gov/drug/label.json` integration, clean HTTP 404 NOT_FOUND handling, boolean query syntax, direct URLs | M3 | R4, Survey 1, Survey 2 |
| 8 | WHO Outbreak & Guidelines Search | Live WHO DON OData endpoint with `$filter` and `$orderby`, `ItemDefaultUrl` URL fixing, HTML stripping for Overview/Advice | M3 | R4, Survey 1, Survey 2 |
| 9 | Full Context Medical Evaluator | Evaluator prompt and context delivery without 400-char truncation, dynamic search source and keyword generation | M4 | R3, Survey 1, Survey 3 |
| 10 | Dynamic Grounded Synthesizer | Context window with full abstracts (1200+ chars), compound citation parsing `[1, 2]`, eliminating static fallback disclaimers | M4 | R3, Survey 1, Survey 3 |
| 11 | Uncontaminated Corrective Evidence Flow | Filter irrelevant local documents before assembling corrective evidence pool, preventing citation leakage | M4 | R3, Survey 3 |
| 12 | End-to-End Live Verification Suite | Test suite testing live APIs, PubMed/FDA/WHO retrieval, OpenRouter reasoning, and CLI showcase queries | M5 | R5, Survey 1, Survey 3 |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Config | `medical_crag/config.py` multi-key detection, whitespace stripping, and robust `.env` loading | none | DONE |
| M2 | LLM Gateway & JSON Resilience | `medical_crag/llm.py` OpenRouter compliance, Gemini integration, and multi-stage robust JSON parser | M1 | DONE |
| M3 | Live Medical Search Tools | `medical_crag/mcp_tools.py` PubMed, openFDA, and WHO live API search fixes and schema alignment | none | IN_PROGRESS |
| M4 | Dynamic Evaluator & Synthesizer | `medical_crag/evaluator.py`, `medical_crag/synthesizer.py`, and `medical_crag/pipeline.py` dynamic reasoning & citation parsing | M2, M3 | PLANNED |
| M5 | Final Verification & Showcase | `tests/test_live_crag.py`, `medical_crag/cli.py`, and 100% E2E test suite pass against live APIs | M4 | PLANNED |

## Interface Contracts

### Configuration ↔ LLM Gateway
- `Settings.openrouter_api_key: str`: non-empty string stripped of whitespace if either `OPENROUTER_API_KEY` or `OPEN_ROUTER_API_KEY` is present.
- `Settings.gemini_api_key: str`: non-empty string stripped of whitespace if either `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present.
- `Settings.from_env(env_file: Optional[Path] = None) -> Settings`: loads `.env` from workspace root `E:\Rag-medical\.env` or current directory.

### LLM Gateway ↔ Evaluator / Synthesizer
- `LLMGateway.generate_json(prompt: str, system_prompt: Optional[str] = None, schema_name: Optional[str] = None) -> Tuple[Dict[str, Any], str]`:
  - Returns parsed dictionary guaranteed to be valid JSON.
  - Second tuple element is provider name (`openrouter` or `gemini`).
  - Raises exception on failure instead of silently returning empty dict or fake fallback.
- `clean_json_text(text: str) -> str`:
  - Handles markdown fences ` ```json ... ``` `
  - Extracts outermost `{ ... }`
  - Cleans trailing commas before closing braces/brackets

### Medical Search Tools ↔ Pipeline / Synthesizer
- `search_pubmed(query: str, max_results: int = 3) -> List[Document]`
  - `doc.source == "pubmed"`
  - `doc.url`: `https://pubmed.ncbi.nlm.nih.gov/<pmid>/`
  - `doc.text`: Full abstract text containing trial results
- `search_fda(query: str, max_results: int = 3) -> List[Document]`
  - `doc.source == "fda"`
  - `doc.url`: `https://www.fda.gov/drugs` or direct drug label URL
  - Returns empty list on 404 NOT_FOUND without throwing exceptions
- `search_who(query: str, max_results: int = 3) -> List[Document]`
  - `doc.source == "who"`
  - `doc.url`: `https://www.who.int/emergencies/disease-outbreak-news/item/<ItemDefaultUrl>`
  - `doc.text`: Cleaned text from `Overview`, `Assessment`, or `Advice` (HTML stripped)

### Pipeline ↔ End User / CLI
- `MedicalCRAGPipeline.verify(claim: str) -> MedicalCRAGResult`:
  - `result.verdict`: `"SUPPORTED"` | `"REFUTED"` | `"INSUFFICIENT_EVIDENCE"`
  - `result.confidence`: `float` (0.0 to 1.0)
  - `result.consumer_summary`: Clear, dynamic, reasoned summary (never static canned disclaimer)
  - `result.clinical_explanation`: Grounded clinical rationale with citation markers
  - `result.sources`: List of cited `Document` objects matching citation markers
  - `result.llm_provider_used`: `"openrouter"` or `"gemini"` (never `"fallback"`)

## Code Layout
- `medical_crag/config.py`: Environment and settings configuration (Owned by M1)
- `medical_crag/llm.py`: Multi-provider LLM gateway and JSON parser (Owned by M2)
- `medical_crag/mcp_tools.py`: External live search tools (Owned by M3)
- `medical_crag/evaluator.py`: Retrieval sufficiency evaluator (Owned by M4)
- `medical_crag/synthesizer.py`: Dynamic clinical synthesizer and citation parser (Owned by M4)
- `medical_crag/pipeline.py`: Corrective RAG pipeline coordinator (Owned by M4)
- `medical_crag/cli.py`: Interactive and showcase CLI (Owned by M5)
- `tests/test_live_crag.py`: Live API automated test suite (Owned by M5 & E2E Track)
- `tests/test_medical_crag/`: Existing unit tests (Regression suite)
