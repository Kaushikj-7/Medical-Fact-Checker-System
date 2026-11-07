"""Adversarial Verification Suite for Milestone M2 Iteration 2 Gate Verification.

Written by Challenger M2-1 Gen 2 to stress-test:
1. Settings primary_provider configuration, case normalization, and fallback.
2. Novel edge cases in clean_json_text and _find_balanced_json_substring:
   - Nested arrays with curly braces in strings
   - Escaped quotes and escaped backslashes inside strings
   - Unicode braces and mathematical delimiters in preambles
   - Multiline LaTeX equations with curly braces
   - Preambles with unclosed or mismatched braces
   - Preambles and postscripts simultaneously
   - Comments containing braces inside JSON structures
   - Provider ordering and failover in LLMGateway
"""

from unittest.mock import patch

from medical_crag.config import DEFAULT_PRIMARY_PROVIDER, Settings
from medical_crag.llm import (
    LLMGateway,
    clean_json_text,
    parse_json_payload,
)


class TestSettingsPrimaryProvider:
    """Empirical verification of Settings.primary_provider."""

    def test_default_primary_provider(self):
        s = Settings()
        assert s.primary_provider == "gemini"

    def test_explicit_valid_providers(self):
        s_gemini = Settings(primary_provider="gemini")
        assert s_gemini.primary_provider == "gemini"

        s_or = Settings(primary_provider="openrouter")
        assert s_or.primary_provider == "openrouter"

    def test_case_insensitivity_and_whitespace(self):
        assert Settings(primary_provider="OPENROUTER").primary_provider == "openrouter"
        assert Settings(primary_provider="OpenRouter").primary_provider == "openrouter"
        assert Settings(primary_provider="  openrouter  ").primary_provider == "openrouter"
        assert Settings(primary_provider="GEMINI").primary_provider == "gemini"
        assert Settings(primary_provider=" Gemini\t").primary_provider == "gemini"

    def test_invalid_provider_falls_back_to_default(self):
        for invalid in ["claude", "gpt-4", "anthropic", "unknown", "", "   "]:
            s = Settings(primary_provider=invalid)
            assert s.primary_provider == DEFAULT_PRIMARY_PROVIDER

    def test_non_string_provider_falls_back_to_default(self):
        for invalid in [None, 123, 45.6, [], {}]:
            s = Settings(primary_provider=invalid)  # type: ignore[arg-type]
            assert s.primary_provider == DEFAULT_PRIMARY_PROVIDER

    def test_from_env_primary_provider(self, monkeypatch):
        monkeypatch.setenv("PRIMARY_PROVIDER", "openrouter")
        s = Settings.from_env()
        assert s.primary_provider == "openrouter"

    def test_from_env_llm_primary_provider_alias(self, monkeypatch):
        monkeypatch.delenv("PRIMARY_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "OPENROUTER")
        s = Settings.from_env()
        assert s.primary_provider == "openrouter"

    def test_from_env_invalid_fallback(self, monkeypatch):
        monkeypatch.setenv("PRIMARY_PROVIDER", "invalid_llm")
        s = Settings.from_env()
        assert s.primary_provider == "gemini"


class TestLLMGatewayProviderDispatch:
    """Empirically verify provider dispatch order and failover in LLMGateway."""

    def test_openrouter_primary_dispatch_order(self):
        s = Settings(
            gemini_api_key="mock_gemini_key",
            openrouter_api_key="mock_openrouter_key",
            primary_provider="openrouter",
        )
        gw = LLMGateway(settings=s)

        call_order = []

        def mock_or(*args, **kwargs):
            call_order.append("openrouter")
            return '{"res": "from_openrouter"}'

        def mock_gemini(*args, **kwargs):
            call_order.append("gemini")
            return '{"res": "from_gemini"}'

        with (
            patch.object(gw, "_call_openrouter", side_effect=mock_or),
            patch.object(gw, "_call_gemini", side_effect=mock_gemini),
        ):
            text, prov = gw.generate("test prompt")
            assert text == '{"res": "from_openrouter"}'
            assert prov == "openrouter"
            assert call_order == ["openrouter"]

    def test_openrouter_failover_to_gemini(self):
        s = Settings(
            gemini_api_key="mock_gemini_key",
            openrouter_api_key="mock_openrouter_key",
            primary_provider="openrouter",
        )
        gw = LLMGateway(settings=s)

        call_order = []

        def mock_or(*args, **kwargs):
            call_order.append("openrouter")
            raise RuntimeError("OpenRouter 503 unavailable")

        def mock_gemini(*args, **kwargs):
            call_order.append("gemini")
            return '{"res": "from_gemini"}'

        with (
            patch.object(gw, "_call_openrouter", side_effect=mock_or),
            patch.object(gw, "_call_gemini", side_effect=mock_gemini),
        ):
            text, prov = gw.generate("test prompt")
            assert text == '{"res": "from_gemini"}'
            assert prov == "gemini"
            assert call_order == ["openrouter", "gemini"]

    def test_gemini_primary_dispatch_order(self):
        s = Settings(
            gemini_api_key="mock_gemini_key",
            openrouter_api_key="mock_openrouter_key",
            primary_provider="gemini",
        )
        gw = LLMGateway(settings=s)

        call_order = []

        def mock_or(*args, **kwargs):
            call_order.append("openrouter")
            return '{"res": "from_openrouter"}'

        def mock_gemini(*args, **kwargs):
            call_order.append("gemini")
            return '{"res": "from_gemini"}'

        with (
            patch.object(gw, "_call_openrouter", side_effect=mock_or),
            patch.object(gw, "_call_gemini", side_effect=mock_gemini),
        ):
            text, prov = gw.generate("test prompt")
            assert text == '{"res": "from_gemini"}'
            assert prov == "gemini"
            assert call_order == ["gemini"]

    def test_gemini_failover_to_openrouter(self):
        s = Settings(
            gemini_api_key="mock_gemini_key",
            openrouter_api_key="mock_openrouter_key",
            primary_provider="gemini",
        )
        gw = LLMGateway(settings=s)

        call_order = []

        def mock_gemini(*args, **kwargs):
            call_order.append("gemini")
            raise RuntimeError("Gemini safety filter triggered")

        def mock_or(*args, **kwargs):
            call_order.append("openrouter")
            return '{"res": "from_openrouter"}'

        with (
            patch.object(gw, "_call_gemini", side_effect=mock_gemini),
            patch.object(gw, "_call_openrouter", side_effect=mock_or),
        ):
            text, prov = gw.generate("test prompt")
            assert text == '{"res": "from_openrouter"}'
            assert prov == "openrouter"
            assert call_order == ["gemini", "openrouter"]


class TestNovelAdversarialJSONScenarios:
    """Stress-test clean_json_text and parse_json_payload against novel adversarial patterns."""

    def test_nested_arrays_with_braces_in_strings(self):
        """Array of objects with curly braces inside nested string values."""
        raw = """
        Analysis output:
        [
            {"tag": "SET_{A}", "subtags": ["{x}", "{y, z}"]},
            {"tag": "SET_{B}", "subtags": ["{1}", "{2}"]}
        ]
        End of report.
        """
        cleaned = clean_json_text(raw)
        assert cleaned.startswith("[") and cleaned.endswith("]")
        data = parse_json_payload(raw)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["tag"] == "SET_{A}"
        assert data[0]["subtags"] == ["{x}", "{y, z}"]

    def test_escaped_quotes_and_braces_in_strings(self):
        """String values with escaped quotes and braces."""
        raw = r'''
        ```json
        {
            "quote": "Doctor said: \"Take {2} tablets daily\", then paused.",
            "path": "C:\\Program Files\\{App}\\data.json",
            "regex": "\\d+\\{[a-z]+\\}"
        }
        ```
        '''
        data = parse_json_payload(raw)
        assert 'Doctor said: "Take {2} tablets daily"' in data["quote"]
        assert data["path"] == r"C:\Program Files\{App}\data.json"

    def test_unicode_and_mathematical_braces_in_preamble(self):
        """Preamble with Unicode fullwidth braces ｛ ｝ and math white braces ⦃ ⦄."""
        raw = (
            "Mathematical definition: ⦃x ∈ ℝ : x ≥ 0⦄ and Japanese notation ｛注釈｝\n"
            "```json\n"
            '{"status": "CONFIRMED", "score": 0.99}\n'
            "```"
        )
        data = parse_json_payload(raw)
        assert data["status"] == "CONFIRMED"
        assert data["score"] == 0.99

    def test_multiline_latex_equation_in_preamble(self):
        """LaTeX equation with multiple nested curly braces in conversational preamble."""
        raw = r"""
        Here is the relevant clinical formula:
        \begin{equation}
        GFR = 141 \times \min\left(\frac{S_{cr}}{\kappa}, 1\right)^{\alpha} \times \max\left(\frac{S_{cr}}{\kappa}, 1\right)^{-1.209} \times 0.993^{Age}
        \end{equation}

        Below is the computed JSON assessment:
        {
            "eGFR": 82.5,
            "stage": "Stage 2 mildly decreased",
            "clinical_guideline": "{KDIGO 2024}"
        }
        """
        cleaned = clean_json_text(raw)
        assert "eGFR" in cleaned
        data = parse_json_payload(raw)
        assert data["eGFR"] == 82.5
        assert data["stage"] == "Stage 2 mildly decreased"
        assert data["clinical_guideline"] == "{KDIGO 2024}"

    def test_unmatched_braces_in_preamble(self):
        """Preamble contains an unmatched opening brace '{' or stray closing brace '}'."""
        raw = "Notice: unclosed section { without end.\nAlso stray } brace.\n{\"verdict\": \"SUPPORTED\"}"
        data = parse_json_payload(raw)
        assert data["verdict"] == "SUPPORTED"

    def test_preamble_and_postscript_simultaneous_braces(self):
        """Both preamble and postscript contain curly braces."""
        raw = (
            "Equation: {x + y = z}\n"
            "```json\n"
            '{"verdict": "SUPPORTED", "reasoning": "Direct match"}\n'
            "```\n"
            "Citation link: {doi:10.1001/jama.2024.1234}"
        )
        data = parse_json_payload(raw)
        assert data["verdict"] == "SUPPORTED"
        assert data["reasoning"] == "Direct match"

    def test_comments_with_braces_inside_json(self):
        """Single-line and multi-line comments containing braces inside the JSON payload."""
        raw = """
        {
            // Comment 1: {ignore_this_brace}
            "study_id": "ST-991",
            /* Multi-line comment:
               function foo() { return {nested: true}; }
            */
            "evidence_grade": "HIGH", // trailing comment {with_brace}
        }
        """
        data = parse_json_payload(raw)
        assert data["study_id"] == "ST-991"
        assert data["evidence_grade"] == "HIGH"

    def test_deeply_nested_mixed_dict_list_trailing_commas(self):
        """Deeply nested dicts and lists with trailing commas at all levels."""
        raw = """
        {
            "level1": [
                {
                    "level2": {
                        "level3": [
                            "item1",
                            "item2",
                        ],
                    },
                },
            ],
            "valid": true,
        }
        """
        data = parse_json_payload(raw)
        assert data["valid"] is True
        assert data["level1"][0]["level2"]["level3"] == ["item1", "item2"]

    def test_python_repr_with_contractions_and_escaped_quotes(self):
        """Python dict repr with contractions and mixed quotes."""
        raw = "{'summary': \"The patient hasn't exhibited symptoms.\", 'is_safe': True, 'score': None, 'sub': {'flag': False,}}"
        data = parse_json_payload(raw)
        assert data["summary"] == "The patient hasn't exhibited symptoms."
        assert data["is_safe"] is True
        assert data["score"] is None
        assert data["sub"]["flag"] is False

    def test_markdown_fence_without_newline_after_fence(self):
        """Edge case: fence closed without newline before ```."""
        raw = "```json\n{\"direct\": true}```"
        data = parse_json_payload(raw)
        assert data["direct"] is True

    def test_multiple_valid_json_blocks_selects_longest(self):
        """When multiple valid JSON candidates exist, the largest valid payload is selected."""
        raw = (
            'Draft: {"note": "short"}\n'
            'Final full analysis: {"verdict": "SUPPORTED", "confidence": 0.95, "detail": "Comprehensive evidence"}'
        )
        data = parse_json_payload(raw)
        assert data["verdict"] == "SUPPORTED"
        assert "Comprehensive evidence" in data["detail"]
