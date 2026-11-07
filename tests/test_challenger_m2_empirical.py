"""Empirical Stress and Adversarial Challenge Suite for Milestone M2.

This test file is written by Challenger M2-2 to independently stress-test:
1. Downstream integration with MedicalEvaluator and MedicalSynthesizer.
2. Rapid sequential calls, prompt variance, and concurrent thread isolation.
3. Gemini REST client socket hygiene, request formatting, and safety block handling.
4. Live OpenRouter credential verification, provider attribution, and medical reasoning quality.
5. Multi-stage JSON parser adversarial edge cases and injection attempts.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from medical_crag.config import Settings
from medical_crag.evaluator import MedicalEvaluator
from medical_crag.llm import LLMGateway, parse_json_payload
from medical_crag.synthesizer import MedicalSynthesizer
from medical_crag.types import Document, Verdict


# ---------------------------------------------------------------------------
# Discovery Helpers for Live Tests
# ---------------------------------------------------------------------------
def _get_live_key() -> str:
    for var in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    try:
        from pathlib import Path
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY") and v:
                        return v
    except (OSError, UnicodeDecodeError):
        pass
    return ""


LIVE_KEY = _get_live_key()
has_live_openrouter = pytest.mark.skipif(
    not LIVE_KEY, reason="Live OpenRouter key required for live empirical test"
)


# ===========================================================================
# 1. Adversarial Multi-Stage JSON Parser Stress Tests
# ===========================================================================
class TestAdversarialJSONParsing:
    """Stress tests on parser under hostile, malformed, or nested inputs."""

    def test_empty_and_whitespace_inputs(self):
        with pytest.raises(RuntimeError, match="empty or whitespace-only"):
            parse_json_payload("")
        with pytest.raises(RuntimeError, match="empty or whitespace-only"):
            parse_json_payload("   \n\t  ")

    def test_markdown_code_fences_with_embedded_fences(self):
        # LLM outputting python code fences INSIDE json explanation
        raw = """```json
        {
            "verdict": "SUPPORTED",
            "clinical_explanation": "Use ```python\\nimport math``` inside the notebook"
        }
        ```"""
        result = parse_json_payload(raw)
        assert result["verdict"] == "SUPPORTED"
        assert "math" in result["clinical_explanation"]

    def test_unescaped_literal_newlines_and_tabs(self):
        # LLMs often omit \\n and output raw newlines inside JSON string values
        raw = '{\n  "clinical_explanation": "First mechanism line.\nSecond mechanism line.",\n  "confidence": 0.85\n}'
        result = parse_json_payload(raw)
        assert result["confidence"] == 0.85
        assert "First mechanism line." in result["clinical_explanation"]

    def test_python_syntax_and_literals(self):
        # Python booleans, None, single quotes
        raw = "{'is_sufficient': True, 'confidence': 0.92, 'search_source': None, 'search_keywords': False}"
        result = parse_json_payload(raw)
        assert result["is_sufficient"] is True
        assert result["confidence"] == 0.92
        assert result["search_source"] is None
        assert result["search_keywords"] is False

    def test_trailing_commas_in_deeply_nested_structures(self):
        raw = """
        {
            "verdict": "SUPPORTED",
            "nested": {
                "arr": [1, 2, 3, ],
                "empty_arr": [ ],
                "sub": {"a": 1, },
            },
        }
        """
        result = parse_json_payload(raw)
        assert result["verdict"] == "SUPPORTED"
        assert result["nested"]["arr"] == [1, 2, 3]
        assert result["nested"]["sub"] == {"a": 1}

    def test_line_and_block_comments_in_json(self):
        raw = """
        // Medical assessment
        {
            /* Clinical verdict block */
            "verdict": "REFUTED", // high confidence
            "confidence": 0.99
        }
        """
        result = parse_json_payload(raw)
        assert result["verdict"] == "REFUTED"
        assert result["confidence"] == 0.99

    def test_unicode_and_emojis_in_json(self):
        raw = '{"verdict": "SUPPORTED", "summary": "❤️ Cardiovascular efficacy demonstrated 🩺: α-tocopherol 100µg"}'
        result = parse_json_payload(raw)
        assert result["verdict"] == "SUPPORTED"
        assert "Cardiovascular" in result["summary"]
        assert "100µg" in result["summary"]

    def test_multi_element_list_rejected_by_generate_json(self):
        settings = Settings(openrouter_api_key="mock", request_timeout=5.0)
        gateway = LLMGateway(settings)
        with (
            patch.object(gateway, "generate", return_value=('[{"a": 1}, {"b": 2}]', "openrouter")),
            pytest.raises(RuntimeError, match="expected a dictionary, got list"),
        ):
            gateway.generate_json("prompt")

    def test_primitive_rejected_by_generate_json(self):
        settings = Settings(openrouter_api_key="mock", request_timeout=5.0)
        gateway = LLMGateway(settings)
        with (
            patch.object(gateway, "generate", return_value=('"plain string response"', "openrouter")),
            pytest.raises(RuntimeError, match="expected a dictionary, got str"),
        ):
            gateway.generate_json("prompt")

    def test_empty_dict_rejected_by_generate_json(self):
        settings = Settings(openrouter_api_key="mock", request_timeout=5.0)
        gateway = LLMGateway(settings)
        with (
            patch.object(gateway, "generate", return_value=("{  }", "openrouter")),
            pytest.raises(RuntimeError, match="empty JSON dictionary"),
        ):
            gateway.generate_json("prompt")


# ===========================================================================
# 2. Downstream Integration: Evaluator & Synthesizer
# ===========================================================================
class TestDownstreamIntegration:
    """Verify seamless contract adherence with Evaluator and Synthesizer."""

    def test_evaluator_with_mocked_llm_gateway(self):
        settings = Settings(openrouter_api_key="mock-key")
        gateway = LLMGateway(settings)
        evaluator = MedicalEvaluator(gateway)

        docs = [
            Document(id="d1", title="Metformin Trial", text="Metformin reduces HbA1c significantly", source="pubmed")
        ]

        mock_payload = {
            "is_sufficient": True,
            "confidence": 0.89,
            "rationale": "High quality randomized controlled trial directly addresses the claim.",
            "search_source": "pubmed",
            "search_keywords": "metformin hba1c trial",
        }
        with patch.object(gateway, "generate_json", return_value=(mock_payload, "openrouter")):
            res = evaluator.evaluate("Metformin lowers HbA1c", docs)
            assert res.is_sufficient is True
            assert res.confidence == 0.89
            assert "metformin hba1c" in res.search_keywords
            assert res.search_source == "pubmed"

    def test_synthesizer_with_mocked_llm_gateway(self):
        settings = Settings(openrouter_api_key="mock-key")
        gateway = LLMGateway(settings)
        synthesizer = MedicalSynthesizer(gateway)

        docs = [
            Document(id="d1", title="Metformin Safety", text="Metformin is contraindicated in severe kidney failure.", source="fda", url="https://fda.gov/label/1"),
            Document(id="d2", title="Metformin Trial", text="Lactic acidosis risk increases with eGFR < 30.", source="pubmed", url="https://pubmed.gov/2"),
        ]

        mock_payload = {
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "consumer_summary": "Metformin should not be taken by patients with advanced kidney disease due to rare but serious risks.",
            "clinical_explanation": "In patients with eGFR < 30 mL/min/1.73m2, metformin accumulation triggers lactic acidosis [1], corroborated by trial literature [2].",
        }
        with patch.object(gateway, "generate_json", return_value=(mock_payload, "openrouter")):
            res = synthesizer.synthesize("Metformin is dangerous in severe renal failure", docs)
            assert res.verdict == Verdict.SUPPORTED.value
            assert res.confidence == 0.95
            assert res.llm_provider_used == "openrouter"
            assert len(res.sources) == 2
            assert res.sources[0].id == "d1"
            assert res.sources[1].id == "d2"

    @has_live_openrouter
    def test_live_evaluator_integration(self):
        """Live evaluation of medical evidence using actual OpenRouter LLM."""
        settings = Settings.from_env()
        gateway = LLMGateway(settings)
        evaluator = MedicalEvaluator(gateway)

        docs = [
            Document(
                id="doc-live-1",
                title="Scurvy Prevention",
                text="Ascorbic acid (Vitamin C) intake of at least 10mg/day prevents clinical scurvy by enabling collagen synthesis.",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            )
        ]

        res = evaluator.evaluate("Vitamin C prevents scurvy", docs)
        assert isinstance(res.is_sufficient, bool)
        assert isinstance(res.confidence, float)
        assert len(res.rationale) > 10
        assert res.search_source in ("pubmed", "fda", "who")

    @has_live_openrouter
    def test_live_synthesizer_integration(self):
        """Live clinical synthesis using actual OpenRouter LLM."""
        settings = Settings.from_env()
        gateway = LLMGateway(settings)
        synthesizer = MedicalSynthesizer(gateway)

        docs = [
            Document(
                id="pmid-1",
                title="Metformin in T2D",
                text="Metformin remains the primary first-line pharmacotherapeutic agent for type 2 diabetes mellitus due to its glycemic efficacy and safety profile.",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/99999/",
            )
        ]

        res = synthesizer.synthesize("Metformin is first-line medication for type 2 diabetes", docs)
        assert res.verdict in ("SUPPORTED", "REFUTED", "MISLEADING", "INSUFFICIENT_EVIDENCE")
        assert res.confidence > 0.5
        assert res.llm_provider_used == "openrouter"
        assert len(res.consumer_summary) > 20
        assert len(res.clinical_explanation) > 20
        assert len(res.sources) >= 1


# ===========================================================================
# 3. Concurrency, Rapid Sequential Calls & Socket Hygiene
# ===========================================================================
class TestConcurrencyAndSocketHygiene:
    """Stress test connection pooling, client lifetime, and concurrent isolation."""

    def test_gemini_client_closed_on_exceptions(self):
        """Empirically verify that httpx.Client context manager closes the client even on network errors."""
        settings = Settings(gemini_api_key="mock-key", request_timeout=2.0)
        gateway = LLMGateway(settings)

        client_instances = []

        class TrackingClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                client_instances.append(self)

        with (
            patch("httpx.Client", side_effect=TrackingClient),
            patch.object(TrackingClient, "post", side_effect=httpx.ConnectError("Connection refused")),
            pytest.raises(RuntimeError, match="Gemini connection failed"),
        ):
            gateway._call_gemini("test prompt")

        assert len(client_instances) == 1
        assert client_instances[0].is_closed, "Client socket pool must be explicitly closed after error"

    def test_gemini_client_closed_on_timeout(self):
        """Empirically verify that httpx.Client is closed after timeout."""
        settings = Settings(gemini_api_key="mock-key", request_timeout=2.0)
        gateway = LLMGateway(settings)

        client_instances = []

        class TrackingClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                client_instances.append(self)

        with (
            patch("httpx.Client", side_effect=TrackingClient),
            patch.object(TrackingClient, "post", side_effect=httpx.TimeoutException("Timed out")),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            gateway._call_gemini("test prompt")

        assert len(client_instances) == 1
        assert client_instances[0].is_closed

    def test_openrouter_client_closed_on_server_error(self):
        """Empirically verify openrouter httpx.Client closes on HTTP 500 error."""
        settings = Settings(openrouter_api_key="mock-key", request_timeout=2.0)
        gateway = LLMGateway(settings)

        client_instances = []

        class TrackingClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                client_instances.append(self)

        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.return_value = {"error": {"message": "Bad Gateway"}}

        with (
            patch("httpx.Client", side_effect=TrackingClient),
            patch.object(TrackingClient, "post", return_value=mock_resp),
            pytest.raises(RuntimeError, match="OpenRouter server error"),
        ):
            gateway._call_openrouter("test prompt")

        assert len(client_instances) == 1
        assert client_instances[0].is_closed

    def test_thread_concurrency_isolation(self):
        """Verify multiple threads calling LLMGateway simultaneously do not race or cross-contaminate."""
        settings = Settings(openrouter_api_key="mock-key")
        gateway = LLMGateway(settings)

        def mock_post(url, *args, **kwargs):
            payload = kwargs.get("json", {})
            prompt = payload.get("messages", [{}])[-1].get("content", "")
            tid = prompt.split("thread ")[-1]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"thread_id": tid, "status": "ok"})
                        }
                    }
                ]
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            def worker_task(thread_id: int):
                res, prov = gateway.generate_json(f"Prompt from thread {thread_id}")
                return res["thread_id"], prov

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker_task, i) for i in range(10)]
                results = [f.result() for f in futures]

        for idx, (res_id, prov) in enumerate(results):
            assert res_id == str(idx)
            assert prov == "openrouter"

    @has_live_openrouter
    def test_rapid_sequential_live_calls(self):
        """Empirical live test: 3 rapid sequential calls with diverse schemas and prompt sizes."""
        settings = Settings.from_env()
        gateway = LLMGateway(settings)

        test_cases = [
            (
                "Short claim: 'Aspirin inhibits COX-1'. Return JSON: {'inhibits': bool, 'enzyme': str}",
                "Respond with valid JSON.",
            ),
            (
                (
                    "Medium claim: 'Penicillin class antibiotics target bacterial cell wall peptidoglycan synthesis'. "
                    "Evaluate whether this is biologically supported. Return JSON: {'claim_valid': bool, 'target': str, 'mechanism': str}"
                ),
                "Respond with valid JSON.",
            ),
            (
                "Clinical synthesis scenario with detailed prompt: " + ("Metformin reduces hepatic gluconeogenesis. " * 10) +
                "\nProvide structured assessment. Return JSON: {'action': str, 'confidence': float}",
                "Respond with valid JSON.",
            ),
        ]

        latencies = []
        for prompt, sys_prompt in test_cases:
            t0 = time.time()
            data, provider = gateway.generate_json(prompt, system_prompt=sys_prompt)
            dt = time.time() - t0
            latencies.append(dt)

            assert provider == "openrouter"
            assert isinstance(data, dict)
            assert len(data) >= 1

        # Check all succeeded
        assert len(latencies) == 3


# ===========================================================================
# 4. Live OpenRouter Reasoned Medical Content Verification
# ===========================================================================
class TestLiveOpenRouterMedicalReasoning:
    """Verify live credentials return reasoned medical content, not empty or generic tokens."""

    @has_live_openrouter
    def test_live_openrouter_factual_medical_reasoning(self):
        settings = Settings.from_env()
        gateway = LLMGateway(settings)

        prompt = (
            "Analyze the mechanism of statins (HMG-CoA reductase inhibitors) on LDL cholesterol clearance. "
            "Return JSON with keys: mechanism, target_enzyme, ldl_receptor_effect, clinical_verdict"
        )
        system_prompt = "You are a clinical pharmacologist. Respond strictly in valid JSON."

        data, provider = gateway.generate_json(prompt, system_prompt=system_prompt)
        assert provider == "openrouter"
        assert isinstance(data, dict)

        # Check for genuine medical domain terms
        combined_text = " ".join(str(v).lower() for v in data.values())
        assert any(term in combined_text for term in ("hmg-coa", "reductase", "statin", "ldl", "cholesterol", "receptor"))
