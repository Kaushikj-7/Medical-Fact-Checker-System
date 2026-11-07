"""Adversarial and Stress Testing Suite for LLMGateway and Resilient JSON Parsing.

Invoked by Challenger M2-1 to stress test medical_crag/llm.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from medical_crag.config import Settings
from medical_crag.llm import (
    LLMGateway,
    clean_json_text,
    parse_json_payload,
)

# ==============================================================================
# 1. Preamble & Bracket Edge Cases
# ==============================================================================


def test_preamble_with_curly_braces_equation():
    """Adversarial: Preamble contains curly braces, e.g. equation {x + y = z} before JSON."""
    raw = 'Note: equation {x + y = z} explains the following JSON:\n{"supported": true}'
    cleaned = clean_json_text(raw)
    assert cleaned == '{"supported": true}', (
        f"clean_json_text greedily matched preamble curly braces: {cleaned!r}"
    )
    data = parse_json_payload(raw)
    assert data["supported"] is True


def test_postscript_with_curly_braces():
    """Adversarial: Postscript contains curly braces, e.g. template tag {footer}."""
    raw = (
        '{"verdict": "SUPPORTED", "confidence": 0.88}\n'
        "Template reference: {footer_note_1}"
    )
    cleaned = clean_json_text(raw)
    assert cleaned == '{"verdict": "SUPPORTED", "confidence": 0.88}', (
        f"clean_json_text greedily matched postscript curly braces: {cleaned!r}"
    )
    data = parse_json_payload(raw)
    assert data["verdict"] == "SUPPORTED"


def test_preamble_with_code_block_and_braces():
    """Preamble contains code block with braces before the actual JSON fence."""
    raw = (
        "Here is a code snippet with braces: `function() { return 1; }`\n"
        "And here is the JSON:\n"
        "```json\n"
        '{"is_sufficient": true, "confidence": 0.95}\n'
        "```\n"
        "End of response."
    )
    data = parse_json_payload(raw)
    assert data["is_sufficient"] is True
    assert data["confidence"] == 0.95


def test_multiple_json_objects_greedy_spanning():
    """Adversarial: LLM outputs two JSON objects separated by explanation."""
    raw = (
        'Initial guess: {"verdict": "REFUTED"}\n'
        'Refined assessment: {"verdict": "SUPPORTED", "confidence": 0.95}'
    )
    cleaned = clean_json_text(raw)
    assert cleaned.startswith("{") and cleaned.endswith("}")


# ==============================================================================
# 2. Markdown Fences & Nested Backticks
# ==============================================================================


def test_markdown_fence_with_internal_backticks():
    """Adversarial: JSON text containing markdown backticks inside JSON string values."""
    raw = (
        "```json\n"
        '{\n'
        '  "consumer_summary": "The study uses `p < 0.05` as significance.",\n'
        '  "clinical_explanation": "Code was run via ```python test()``` format.",\n'
        '  "confidence": 0.9\n'
        '}\n'
        "```"
    )
    cleaned = clean_json_text(raw)
    assert "confidence" in cleaned, (
        f"clean_json_text truncated at internal markdown fence: {cleaned!r}"
    )
    data = parse_json_payload(raw)
    assert data["confidence"] == 0.9


def test_markdown_fence_case_variants_and_whitespace():
    """Fence with mixed casing and arbitrary spacing."""
    raw = "```   JSON   \r\n{\"status\": \"OK\"}\r\n```"
    data = parse_json_payload(raw)
    assert data["status"] == "OK"


def test_markdown_fence_unclosed_with_trailing_text():
    """Unclosed markdown fence."""
    raw = "```json\n{\"status\": \"active\", \"score\": 42}"
    data = parse_json_payload(raw)
    assert data["status"] == "active"
    assert data["score"] == 42


def test_markdown_fence_quadruple_backticks():
    """Adversarial: Outer markdown fence uses 4 backticks ````json ... ````."""
    raw = "````json\n{\"valid\": true}\n````"
    data = parse_json_payload(raw)
    assert data["valid"] is True


# ==============================================================================
# 3. Unicode, Surrogate Pairs, Emojis, and Non-ASCII Characters
# ==============================================================================

def test_unicode_and_emojis():
    """JSON containing emojis, Greek medical letters, and escaped unicode."""
    raw = (
        '{\n'
        '  "condition": "Type 2 Diabetes Mellitus \U0001f3e5 \u2695\ufe0f",\n'
        '  "biomarker": "\u03b1-\u03b2 receptor (\u03b1\u03b2\u03b3)",\n'
        '  "dosage": "\u00b5g/mL",\n'
        '  "temperature_range": "37\u00b0C \u00b1 0.5\u00b0C",\n'
        '  "escaped_quote": "Patient said \\"feeling better\\""\n'
        '}'
    )
    data = parse_json_payload(raw)
    assert "Diabetes" in data["condition"]
    assert "\u03b1-\u03b2" in data["biomarker"]
    assert "\u00b5g/mL" == data["dosage"]
    assert 'Patient said "feeling better"' == data["escaped_quote"]


def test_cjk_and_arabic_characters():
    """JSON containing CJK, Arabic, and Cyrillic medical text."""
    raw = (
        '{\n'
        '  "zh": "\u4e8c\u578b\u7cd6\u5c3f\u75c5",\n'
        '  "ar": "\u0645\u0631\u0636 \u0627\u0644\u0633\u0643\u0631\u064a",\n'
        '  "ru": "\u0421\u0430\u0445\u0430\u0440\u043d\u044b\u0439 \u0434\u0438\u0430\u0431\u0435\u0442"\n'
        '}'
    )
    data = parse_json_payload(raw)
    assert data["zh"] == "\u4e8c\u578b\u7cd6\u5c3f\u75c5"
    assert data["ar"] == "\u0645\u0631\u0636 \u0627\u0644\u0633\u0643\u0631\u064a"
    assert data["ru"] == "\u0421\u0430\u0445\u0430\u0440\u043d\u044b\u0439 \u0434\u0438\u0430\u0431\u0435\u0442"


# ==============================================================================
# 4. Multiline Strings with Literal Unescaped Control Chars
# ==============================================================================

def test_multiline_strings_with_literal_newlines_and_tabs():
    """LLMs frequently return literal unescaped newlines in JSON strings."""
    raw = """{
  "summary": "Line 1: Patient presented with fever.
Line 2: Administered 500mg paracetamol.
	Tabbed note: observation ongoing.",
  "confidence": 0.88
}"""
    data = parse_json_payload(raw)
    assert "Line 1:" in data["summary"]
    assert "Line 2:" in data["summary"]
    assert "Tabbed note:" in data["summary"]
    assert data["confidence"] == 0.88


# ==============================================================================
# 5. Complex Nested Trailing Commas
# ==============================================================================

def test_complex_nested_trailing_commas():
    """Deeply nested objects and arrays with multiple trailing commas."""
    raw = """{
  "study": {
    "cohorts": [
      {"id": 1, "size": 100, },
      {"id": 2, "size": 150,, },
    ],
    "endpoints": [
      "hba1c",
      "fasting_glucose",
    ],
    "meta": {
      "active": true,
      "nested": {
        "val": 123,
      },
    },
  },
}"""
    data = parse_json_payload(raw)
    assert data["study"]["cohorts"][0]["size"] == 100
    assert data["study"]["cohorts"][1]["size"] == 150
    assert data["study"]["endpoints"] == ["hba1c", "fasting_glucose"]
    assert data["study"]["meta"]["nested"]["val"] == 123


# ==============================================================================
# 6. Python Single-Quoted Representation with True, False, None
# ==============================================================================

def test_python_repr_dict_single_quotes():
    """Python dict string with single quotes and True, False, None."""
    raw = "{'is_sufficient': True, 'has_warning': False, 'extra_notes': None, 'sub_list': ['a', 'b', None]}"
    data = parse_json_payload(raw)
    assert data["is_sufficient"] is True
    assert data["has_warning"] is False
    assert data["extra_notes"] is None
    assert data["sub_list"] == ["a", "b", None]


def test_python_repr_with_inner_double_quotes():
    """Python dict representation where values contain inner double quotes."""
    raw = "{'summary': 'The doctor said \"rest is recommended\"', 'valid': True}"
    data = parse_json_payload(raw)
    assert 'The doctor said "rest is recommended"' in data["summary"]
    assert data["valid"] is True


def test_python_repr_with_contractions():
    """Adversarial: Python single-quoted dict containing escaped contraction, e.g. didn\'t."""
    raw = "{'summary': 'Patient didn\\'t exhibit symptoms', 'active': True}"
    data = parse_json_payload(raw)
    assert "didn't" in data["summary"]
    assert data["active"] is True


# ==============================================================================
# 7. Comments in JSON
# ==============================================================================

def test_trailing_and_block_comments():
    """JSON with single-line (//) and multi-line (/* */) comments."""
    raw = """{
  // Primary evaluation verdict
  "verdict": "SUPPORTED", /* High quality RCT */
  "confidence": 0.95, // Verified by Cochrane review
  "details": {
    /* Multi-line
       explanation
    */
    "source_count": 3,
  }
}"""
    data = parse_json_payload(raw)
    assert data["verdict"] == "SUPPORTED"
    assert data["confidence"] == 0.95
    assert data["details"]["source_count"] == 3


# ==============================================================================
# 8. Unparseable & Invalid Inputs Must Raise RuntimeError
# ==============================================================================

def test_empty_string_raises_runtime_error():
    with pytest.raises(RuntimeError, match="cannot parse JSON"):
        parse_json_payload("")
    with pytest.raises(RuntimeError, match="cannot parse JSON"):
        parse_json_payload("   \n\t  ")


def test_html_error_page_raises_runtime_error():
    raw = "<html><body><h1>502 Bad Gateway</h1>Cloudflare outage</body></html>"
    with pytest.raises(RuntimeError):
        parse_json_payload(raw)


def test_model_chatty_refusal_raises_runtime_error():
    raw = "I am an AI assistant and I cannot provide medical advice. Please consult your physician."
    with pytest.raises(RuntimeError, match="Model (response contained no JSON structure|failed to emit valid JSON)"):
        parse_json_payload(raw)


def test_corrupted_json_syntax_raises_runtime_error():
    raw = '{"verdict": "SUPPORTED", "confidence": '
    with pytest.raises(RuntimeError, match="Model failed to emit valid JSON"):
        parse_json_payload(raw)


def test_generate_json_empty_dict_raises():
    settings = Settings(openrouter_api_key="mock_key")
    gateway = LLMGateway(settings)
    with (
        patch.object(gateway, "generate", return_value=("{}", "openrouter")),
        pytest.raises(RuntimeError, match="Model emitted an empty JSON dictionary"),
    ):
        gateway.generate_json("prompt")


def test_generate_json_primitive_type_raises():
    settings = Settings(openrouter_api_key="mock_key")
    gateway = LLMGateway(settings)
    with (
        patch.object(gateway, "generate", return_value=("[1, 2, 3]", "openrouter")),
        pytest.raises(RuntimeError, match="expected a dictionary, got list"),
    ):
        gateway.generate_json("prompt")


def test_generate_json_unwraps_single_dict_in_list():
    settings = Settings(openrouter_api_key="mock_key")
    gateway = LLMGateway(settings)
    with patch.object(gateway, "generate", return_value=('[{"is_sufficient": true, "confidence": 0.9}]', "openrouter")):
        data, provider = gateway.generate_json("prompt")
        assert provider == "openrouter"
        assert data["is_sufficient"] is True
        assert data["confidence"] == 0.9


# ==============================================================================
# 9. Provider Fallback Mechanics Stress Testing
# ==============================================================================

def test_fallback_under_primary_gemini_500():
    """Simulate Gemini returning HTTP 500, verifies clean fallback to OpenRouter."""
    settings = Settings(
        gemini_api_key="gemini_key",
        openrouter_api_key="openrouter_key",
    )
    settings.primary_provider = "gemini"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=RuntimeError("Gemini API error (HTTP 500): Internal Server Error")),
        patch.object(gateway, "_call_openrouter", return_value='{"verdict": "SUPPORTED"}') as mock_or,
    ):
        data, provider = gateway.generate_json("prompt")
        assert provider == "openrouter"
        assert data["verdict"] == "SUPPORTED"
        assert mock_or.called


def test_fallback_under_gemini_safety_block():
    """Simulate Gemini finishReason SAFETY, verifies fallback to OpenRouter."""
    settings = Settings(
        gemini_api_key="gemini_key",
        openrouter_api_key="openrouter_key",
    )
    settings.primary_provider = "gemini"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=RuntimeError("Gemini response blocked due to SAFETY: [{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT'}]")),
        patch.object(gateway, "_call_openrouter", return_value='{"verdict": "REFUTED"}') as mock_or,
    ):
        data, provider = gateway.generate_json("prompt")
        assert provider == "openrouter"
        assert data["verdict"] == "REFUTED"
        assert mock_or.called


def test_fallback_under_gemini_timeout():
    """Simulate Gemini request timeout, verifies fallback to OpenRouter."""
    settings = Settings(
        gemini_api_key="gemini_key",
        openrouter_api_key="openrouter_key",
    )
    settings.primary_provider = "gemini"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=RuntimeError("Gemini request timed out after 10.0s")),
        patch.object(gateway, "_call_openrouter", return_value='{"verdict": "INSUFFICIENT_EVIDENCE"}') as mock_or,
    ):
        data, provider = gateway.generate_json("prompt")
        assert provider == "openrouter"
        assert data["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert mock_or.called


def test_fallback_under_primary_openrouter_rate_limit():
    """Simulate OpenRouter primary failing with 429 rate limit, falling back to Gemini."""
    settings = Settings(
        gemini_api_key="gemini_key",
        openrouter_api_key="openrouter_key",
    )
    settings.primary_provider = "openrouter"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_openrouter", side_effect=RuntimeError("OpenRouter rate limit exceeded (HTTP 429)")),
        patch.object(gateway, "_call_gemini", return_value='{"verdict": "SUPPORTED"}') as mock_gemini,
    ):
        data, provider = gateway.generate_json("prompt")
        assert provider == "gemini"
        assert data["verdict"] == "SUPPORTED"
        assert mock_gemini.called


def test_exhaustion_when_all_fail_consolidates_error_reasons():
    """When all providers fail, consolidated error message details each provider's error."""
    settings = Settings(
        gemini_api_key="gemini_key",
        openrouter_api_key="openrouter_key",
    )
    settings.primary_provider = "gemini"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=RuntimeError("Gemini quota 429")),
        patch.object(gateway, "_call_openrouter", side_effect=RuntimeError("OpenRouter balance 402")),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            gateway.generate_json("prompt")
        msg = str(exc_info.value)
        assert "All LLM providers failed or no keys configured" in msg
        assert "gemini: Gemini quota 429" in msg
        assert "openrouter: OpenRouter balance 402" in msg


# ==============================================================================
# 10. Live OpenRouter Integration Stress
# ==============================================================================

def test_live_openrouter_structured_json():
    """Live call to OpenRouter verifying attribution, headers, and parsed dict."""
    settings = Settings.from_env()
    if not settings.openrouter_api_key:
        pytest.skip("No OpenRouter API key found in environment or .env")
    settings.primary_provider = "openrouter"
    gateway = LLMGateway(settings)

    prompt = (
        "Respond with a JSON object evaluating this medical claim:\n"
        "'Metformin is a first-line medication for type 2 diabetes.'\n"
        "Return keys: 'claim', 'supported', 'first_line', 'target_condition'."
    )
    system_prompt = "You are a clinical pharmacologist. Output ONLY valid JSON."
    data, provider = gateway.generate_json(prompt, system_prompt=system_prompt)

    assert provider == "openrouter"
    assert isinstance(data, dict)
    assert any(k in data for k in ("supported", "first_line", "claim"))
