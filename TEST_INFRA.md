# Medical Corrective RAG (CRAG) — Test Infrastructure Specification

**Document Version**: 1.0.0  
**Target Root**: `E:\Rag-medical\`  
**Test Suite Path**: `tests/test_live_crag.py`  
**Test Methodology**: 4-Tier Hierarchical Testing (Category-Partition, Boundary Value Analysis, Pairwise Combinatorial, Real-World Workloads)

---

## 1. Executive Summary & Testing Philosophy

Medical Corrective RAG (CRAG) is a mission-critical biomedical claim verification pipeline. Clinical assertions made to consumers and clinicians demand rigorous validation across four distinct test tiers:
1. **Tier 1: Feature Coverage (Category-Partition Testing)**: Exhaustively partitions inputs and behaviors for Requirements R1 through R4, guaranteeing $\ge 5$ targeted test cases per feature.
2. **Tier 2: Boundary Value Analysis & Edge Cases**: Exercises extreme boundaries (empty inputs, missing keys, HTTP 404 NOT_FOUND, invalid models, malformed JSON, and citation compound ranges).
3. **Tier 3: Cross-Feature Combinations (Pairwise & Integration)**: Validates state handoffs across architectural seams (e.g. `Settings` $\to$ `LLMGateway` $\to$ `clean_json_text` $\to$ `Evaluator` $\to$ `MCPClient` $\to$ `ChromaStore` $\to$ `Synthesizer`).
4. **Tier 4: Real-World Medical Claim Scenarios**: Validates the end-to-end pipeline against realistic medical claims across viral respiratory infections, boxed warnings, herbal supplements, cardiovascular therapies, epidemiological alerts, and unsubstantiated myths.

The testing infrastructure supports **dual-mode execution**:
- **Live Mode**: When valid API credentials (`OPENROUTER_API_KEY` or `GEMINI_API_KEY`) and live internet connections are detected, tests interact with live endpoints (OpenRouter, PubMed, openFDA, WHO) to verify real clinical reasoning and citations.
- **Hermetic / Safe Fallback Mode**: When keys are absent, isolated or running in continuous integration without secrets, tests verify component contracts, graceful fallback paths, clean exception handling, and expected structural degradation without throwing unhandled exceptions.

---

## 2. Four-Tier Test Architecture Matrix

```
+---------------------------------------------------------------------------------------+
|                                    TIER 4                                             |
|                     Real-World Medical Claim Scenarios                                |
|  - Viral Cold (CDC/PubMed)      - Metformin Lactic Acidosis (FDA)                     |
|  - Ashwagandha Sleep (PubMed)   - Turmeric Osteoarthritis (PubMed)                    |
|  - Aspirin Heart Attack (FDA)   - Avian Flu Outbreak (WHO)  - Moonstone Myth (Clean)  |
+---------------------------------------------------------------------------------------+
                                           ▲
+---------------------------------------------------------------------------------------+
|                                    TIER 3                                             |
|                     Cross-Feature Combinations (Pairwise)                             |
|  - Config -> LLMGateway -> JSON Parser -> Evaluator                                   |
|  - Insufficient Local Retrieval -> MCP Multi-Tool Search -> Auto-Ingest -> Synthesis  |
|  - Clean Evidence Filtering (Zero Citation Leakage from Irrelevant Local Context)     |
+---------------------------------------------------------------------------------------+
                                           ▲
+---------------------------------------------------------------------------------------+
|                                    TIER 2                                             |
|                     Boundary Value & Corner Case Analysis                             |
|  - Empty / Whitespace Claims    - 404 NOT_FOUND on openFDA  - Empty Choice Responses  |
|  - Whitespace & Alias Keys      - Trailing Commas in JSON   - Compound Bracket [1, 2] |
+---------------------------------------------------------------------------------------+
                                           ▲
+---------------------------------------------------------------------------------------+
|                                    TIER 1                                             |
|                     Core Feature Coverage (Category-Partition)                        |
|  - R1: Config & Multi-Key Extraction (>=5 Test Cases)                                 |
|  - R2: Live LLM Provider Integration & JSON Resilience (>=5 Test Cases)               |
|  - R3: Dynamic Evaluator & Grounded Synthesizer (>=5 Test Cases)                      |
|  - R4: Medical Evidence Search Tools: PubMed, FDA, WHO (>=5 Test Cases)               |
+---------------------------------------------------------------------------------------+
```

---

## 3. Tier 1: Core Feature Coverage (Category-Partition)

### 3.1 Requirement R1: Environment Configuration & API Key Extraction

| Test ID | Partition / Scenario | Input Condition | Expected Behavior / Output | Authoritative Reference |
|---|---|---|---|---|
| `TC-R1-01` | OpenRouter Key Aliasing | `OPEN_ROUTER_API_KEY` set in env, `OPENROUTER_API_KEY` unset | `Settings.from_env().openrouter_api_key` extracts key correctly | PROJECT.md § M1, Survey 1 § 2.2 |
| `TC-R1-02` | Standard OpenRouter Key | `OPENROUTER_API_KEY` set, `OPEN_ROUTER_API_KEY` unset | Key extracted correctly without errors | PROJECT.md § M1 |
| `TC-R1-03` | Gemini Key Aliasing | `GOOGLE_API_KEY` set, `GEMINI_API_KEY` unset | `Settings.from_env().gemini_api_key` extracts key correctly | PROJECT.md § M1 |
| `TC-R1-04` | Key Whitespace Stripping | Key has leading/trailing spaces (`"  sk-or-v1-... \t\n"`) | Extracted key is cleanly stripped to `"sk-or-v1-..."` | PROJECT.md § Interface Contracts |
| `TC-R1-05` | Custom `.env` File Loading | Specific `.env` file path provided to `Settings.from_env(path)` | Settings loaded from target path | PROJECT.md § Interface Contracts |
| `TC-R1-06` | Missing Keys Fallback | Neither OpenRouter nor Gemini keys set | Settings initializes with empty strings `""`; no crash | Survey 1 § 2.2 |

### 3.2 Requirement R2: Live LLM Provider Integration & JSON Resilience

| Test ID | Partition / Scenario | Input Condition | Expected Behavior / Output | Authoritative Reference |
|---|---|---|---|---|
| `TC-R2-01` | OpenRouter Headers & Payload | Live or mocked call to OpenRouter | Headers include `Authorization: Bearer <key>`, `HTTP-Referer`, `X-Title`; payload contains model and messages | OpenRouter Live Spec § 1 |
| `TC-R2-02` | OpenRouter JSON Mode Enforcement | `generate_json` executed | Payload includes `response_format={"type": "json_object"}` or returns parseable JSON | OpenRouter Spec § 1 |
| `TC-R2-03` | Multi-Stage JSON Markdown Strip | LLM returns ```` ```json {"key": "val"} ``` ```` | `clean_json_text` extracts raw `{"key": "val"}` | PROJECT.md § M2 |
| `TC-R2-04` | Multi-Stage Outermost JSON Extraction | LLM returns `Preamble text... {"key": "val"} Postscript commentary` | `clean_json_text` strips conversational text and parses JSON object | Pipeline Analysis § 2 |
| `TC-R2-05` | Trailing Comma Cleanup | LLM returns `{"a": 1, "b": [1, 2, ], }` | Trailing commas cleaned; parses successfully with `json.loads` | Pipeline Analysis § 8.1 |
| `TC-R2-06` | Provider Failover | Gemini key invalid or missing; OpenRouter key valid | LLMGateway seamlessly fails over to OpenRouter; provider returned is `"openrouter"` | PROJECT.md § Architecture |

### 3.3 Requirement R3: Dynamic Evaluator & Synthesizer Execution

| Test ID | Partition / Scenario | Input Condition | Expected Behavior / Output | Authoritative Reference |
|---|---|---|---|---|
| `TC-R3-01` | Evaluator Full Context Delivery | Retrieved document has text $>400$ characters | Evaluator prompt includes full text without 400-char clipping | Survey 1 § 4.1, Pipeline Analysis § 2 |
| `TC-R3-02` | Evaluator Dynamic Routing | Claim concerning drug label warning (e.g. Metformin) | Evaluator dynamically outputs `search_source="fda"` and relevant keywords | Pipeline Analysis § 7 |
| `TC-R3-03` | Evaluator Keyword Sanitization | Model outputs list `["metformin", "lactic acidosis"]` or `None` | Keywords converted to clean search query string (not stringified python list or `"None"`) | Pipeline Analysis § 2 |
| `TC-R3-04` | Synthesizer Full Abstract Context | Document text $>1000$ characters containing conclusions | Prompt receives full clinical conclusion; verdict is informed by clinical results | Survey 1 § 4.2 |
| `TC-R3-05` | Compound Citation Parsing | Explanation contains compound markers `[1, 2]` or `[1-3]` | Citation parser resolves both document 1 and document 2 into `result.sources` | Pipeline Analysis § 3 |
| `TC-R3-06` | Canned Fallback Elimination | Valid LLM generation on claim | Result contains dynamically reasoned `consumer_summary` and `clinical_explanation`, never static default disclaimer | PROJECT.md § R3, Survey 1 § 4.2 |

### 3.4 Requirement R4: Medical Evidence Search Tools

| Test ID | Partition / Scenario | Input Condition | Expected Behavior / Output | Authoritative Reference |
|---|---|---|---|---|
| `TC-R4-01` | PubMed Live Search & Compliance | Query `"ashwagandha sleep quality"` | Returns `Document` objects with `pubmed-<pmid>`, `source="PubMed"`, URL `https://pubmed.ncbi.nlm.nih.gov/<pmid>/`, includes `tool` and `email` | Spec Report § 3 |
| `TC-R4-02` | PubMed XML Abstract Extraction | Complex XML abstract with nested tags (`<i>`, `<b>`, `<inf>`) | Extracts complete text using `itertext()` without dropping text | Spec Report § 3 |
| `TC-R4-03` | openFDA Live Drug Search | Query `"metformin"` | Returns official FDA drug label with boxed warning / indications | Spec Report § 4 |
| `TC-R4-04` | openFDA 404 Clean Return | Nonexistent drug query `"nonexistentdrugxyz999"` | Returns empty list `[]` without raising unhandled HTTP 404 exception | Spec Report § 4 |
| `TC-R4-05` | WHO DON Live Endpoint & URL Fix | Query `"avian influenza"` | Returns WHO alert with `ItemDefaultUrl` correctly mapped to full URL and HTML stripped from text | Spec Report § 5 |
| `TC-R4-06` | MCPClient Routing Dispatch | Dispatch to `pubmed`, `fda`, and `who` | Queries routed to corresponding API with correct source tag | PROJECT.md § M3 |

---

## 4. Tier 2: Boundary Value Analysis & Edge Cases

| Test ID | Boundary / Edge Category | Input / Stress Condition | Invariant / Expected Handling |
|---|---|---|---|
| `TC-BVA-01` | Empty Input Claims | Empty string `""` or whitespace `"   "` | Raises `ValueError("Health claim must not be empty.")` immediately |
| `TC-BVA-02` | Zero Search Results Short-Circuit | External search returns `[]` for unknown/nonsense claim | Pipeline returns `INSUFFICIENT_EVIDENCE`, `confidence=0.0`, `llm_provider_used="none"`, synthesizer bypassed |
| `TC-BVA-03` | Malformed LLM Output / Single Quotes | LLM emits Python dict string `{'is_sufficient': True, 'confidence': 0.8}` | Handled gracefully via multi-stage JSON cleaner / literal evaluation fallback |
| `TC-BVA-04` | openFDA Multi-Word Lucene Query | Multi-term query `"metformin lactic acidosis"` | Handled cleanly via fallback query syntax without 404 crash |
| `TC-BVA-05` | Missing Optional Fields in WHO | WHO item has `"Summary": ""` and raw HTML in `Overview` | Strips HTML tags, extracts text from `Overview` or `Advice` |
| `TC-BVA-06` | Citation Out of Bounds | Clinical explanation references `[99]` when only 2 documents exist | Ignores out-of-range index; does not throw `IndexError` |
| `TC-BVA-07` | Missing API Keys at Runtime | No keys in environment or settings | Raises clean `RuntimeError("All LLM providers failed or no keys configured...")` without uncaught crashes |

---

## 5. Tier 3: Cross-Feature Combinations (Pairwise & Integration Chains)

| Test ID | Cross-Feature Seam | Verification Flow | Expected Result |
|---|---|---|---|
| `TC-XF-01` | Config $\to$ LLM Gateway $\to$ JSON Parser | Load keys from config $\to$ invoke `generate_json` with messy prompt $\to$ clean JSON returned | Structured dict parsed with accurate provider attribution |
| `TC-XF-02` | Insufficient Local $\to$ Evaluator $\to$ PubMed $\to$ Chroma Ingest $\to$ Synthesizer | Query without local docs $\to$ Evaluator selects `pubmed` $\to$ MCP fetches PubMed doc $\to$ ChromaDB store count increments $\to$ Synthesizer generates verdict with PubMed citation | Verdict generated, `corrective_action_taken=True`, ChromaDB auto-ingest confirmed |
| `TC-XF-03` | Insufficient Local $\to$ Evaluator $\to$ openFDA $\to$ Chroma Ingest $\to$ Synthesizer | Query on drug boxed warning $\to$ Evaluator selects `fda` $\to$ MCP queries openFDA $\to$ Synthesizer produces clinical explanation with FDA warning citation | Boxed warning cited, `corrective_action_taken=True` |
| `TC-XF-04` | Sufficient Local $\to$ Evaluator $\to$ Bypass MCP $\to$ Synthesizer | Query with pre-seeded local match $\to$ Evaluator marks `is_sufficient=True` $\to$ MCP search is skipped $\to$ Synthesizer runs immediately | `corrective_action_taken=False`, MCP search never invoked |
| `TC-XF-05` | Uncontaminated Evidence Flow | Corrective search triggered $\to$ Local ChromaDB contains irrelevant background documents | Irrelevant local background docs are filtered; `result.sources` only contains relevant external docs |

---

## 6. Tier 4: Real-World Medical Claim Scenarios

### 6.1 Clinical Scenario Portfolio

| Scenario ID | Medical Claim | Expected Verdict | Target Evidence Source | Clinical Grounding & Rationale |
|---|---|---|---|---|
| `TC-RW-01` | *"Antibiotics are effective for curing viral infections like the common cold."* | `REFUTED` | CDC / PubMed | Antibiotics only target bacterial cell walls/ribosomes and have zero efficacy against rhinovirus/influenza viruses. |
| `TC-RW-02` | *"Metformin carries an FDA boxed warning for lactic acidosis."* | `SUPPORTED` | openFDA | Official FDA package insert includes black box warning for metformin-associated lactic acidosis (MALA), especially in renal impairment. |
| `TC-RW-03` | *"Ashwagandha root extract improves sleep quality and reduces insomnia in clinical trials."* | `SUPPORTED` | PubMed | Randomized, double-blind, placebo-controlled trials show Withania somnifera improves sleep latency and sleep quality via GABAergic modulation. |
| `TC-RW-04` | *"Turmeric and curcumin products reduce knee osteoarthritis pain."* | `SUPPORTED` | PubMed | Systematic reviews and network meta-analyses demonstrate significant WOMAC pain score reductions compared to placebo. |
| `TC-RW-05` | *"Daily aspirin is recommended for all adults to prevent heart attacks."* | `REFUTED` / `MISLEADING` | openFDA / PubMed | Current USPSTF and FDA guidance does not recommend universal daily aspirin for primary prevention due to gastrointestinal/cerebral bleeding risks. |
| `TC-RW-06` | *"Avian influenza A(H5N1) has caused reported human infections and outbreaks in poultry."* | `SUPPORTED` | WHO Disease Outbreak News | WHO DON alerts document confirmed human infections and widespread poultry outbreaks with epidemiological characterization. |
| `TC-RW-07` | *"Crushed blue moonstone dissolved in mineral water cures type 1 diabetes."* | `INSUFFICIENT_EVIDENCE` | Pure Negative / Nonsense | Zero peer-reviewed or regulatory evidence exists; pipeline cleanly short-circuits with confidence 0.0. |

---

## 7. Test Suite Execution & Automation Guidelines

### 7.1 Running the Suite
The test suite in `tests/test_live_crag.py` is executed via pytest:

```powershell
# Run the live verification suite with Python 3.12
$env:PYTHONPATH="E:\Rag-medical"; py -3.12 -m pytest tests/test_live_crag.py -v

# Run the complete regression and verification test suite
$env:PYTHONPATH="E:\Rag-medical"; py -3.12 -m pytest tests/ -v
```

### 7.2 Expected Exit Codes & Assertions
- `0 (Success)`: All tests passed or cleanly skipped according to key availability.
- Non-zero indicates genuine regression or defect in configuration, gateway, parser, tools, evaluator, or synthesizer.
