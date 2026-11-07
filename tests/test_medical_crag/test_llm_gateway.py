from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from medical_crag.config import Settings
from medical_crag.llm import (
    LLMGateway,
    clean_json_text,
    parse_json_payload,
)

# ==============================================================================
# Existing Provider Fallback Tests (Retained)
# ==============================================================================


def test_gemini_success():
    settings = Settings(
        gemini_api_key="mock_gemini_key",
        openrouter_api_key="mock_openrouter_key",
        gemini_model="gemini-1.5-flash",
    )
    gateway = LLMGateway(settings)

    with patch.object(gateway, "_call_gemini", return_value="Gemini response"):
        resp, provider = gateway.generate("test prompt")
        assert resp == "Gemini response"
        assert provider == "gemini"


def test_gemini_failure_fallback_to_openrouter():
    settings = Settings(
        gemini_api_key="mock_gemini_key",
        openrouter_api_key="mock_openrouter_key",
        openrouter_model="meta-llama/llama-3.3-70b-instruct",
    )
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=Exception("Gemini 429 Rate Limited")),
        patch.object(gateway, "_call_openrouter", return_value="OpenRouter response"),
    ):
        resp, provider = gateway.generate("test prompt")
        assert resp == "OpenRouter response"
        assert provider == "openrouter"


def test_both_providers_fail_raises_runtime_error():
    settings = Settings(
        gemini_api_key="mock_gemini_key",
        openrouter_api_key="mock_openrouter_key",
    )
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=Exception("Gemini error")),
        patch.object(gateway, "_call_openrouter", side_effect=Exception("OpenRouter error")),
        pytest.raises(RuntimeError, match="All LLM providers failed"),
    ):
        gateway.generate("test prompt")


def test_json_mode_parsing():
    settings = Settings(gemini_api_key="mock_gemini_key")
    gateway = LLMGateway(settings)

    raw_json = '```json\n{"is_sufficient": true, "confidence": 0.9}\n```'
    with patch.object(gateway, "_call_gemini", return_value=raw_json):
        data, provider = gateway.generate_json("test prompt")
        assert data["is_sufficient"] is True
        assert data["confidence"] == 0.9
        assert provider == "gemini"


# ==============================================================================
# Multi-Stage Resilient Parser Unit Tests
# ==============================================================================


def test_clean_json_text_markdown_fence_variants():
    """Verify clean_json_text strips ```json, ```JSON, plain ```, and unclosed fences."""
    # Standard lowercase fence
    assert clean_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
    # Uppercase JSON fence
    assert clean_json_text('```JSON\n{"a": 1}\n```') == '{"a": 1}'
    # Plain fence without tag
    assert clean_json_text('```\n{"a": 1}\n```') == '{"a": 1}'
    # Unclosed / truncated fence
    assert clean_json_text('```json\n{"a": 1}') == '{"a": 1}'


def test_clean_json_text_multiple_markdown_code_blocks():
    """Verify clean_json_text handles output with markdown blocks and surrounding text."""
    raw = (
        "Here is the result:\n"
        "```json\n"
        '{"status": "success", "count": 2}\n'
        "```\n"
        "Additional notes follow."
    )
    cleaned = clean_json_text(raw)
    parsed = json.loads(cleaned)
    assert parsed["status"] == "success"
    assert parsed["count"] == 2


def test_clean_json_text_conversational_preamble_and_postscript():
    """Verify clean_json_text isolates outermost JSON object from preambles and postscripts."""
    raw = (
        "Based on the medical literature, here is the clinical assessment:\n"
        '{"verdict": "SUPPORTED", "confidence": 0.92, "rationale": "Strong RCT evidence."}\n'
        "Hope this explanation assists your diagnostic evaluation!"
    )
    cleaned = clean_json_text(raw)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")
    parsed = json.loads(cleaned)
    assert parsed["verdict"] == "SUPPORTED"
    assert parsed["confidence"] == 0.92


def test_clean_json_text_trailing_commas():
    """Verify clean_json_text strips trailing commas in objects and arrays."""
    raw = '{"verdict": "REFUTED", "citations": [1, 2, ], "extra": {"nested": true, }, }'
    cleaned = clean_json_text(raw)
    parsed = json.loads(cleaned)
    assert parsed["verdict"] == "REFUTED"
    assert parsed["citations"] == [1, 2]
    assert parsed["extra"]["nested"] is True


def test_clean_json_text_truncated_markdown_fences():
    """Verify clean_json_text handles truncated markdown fences at start or end."""
    raw = '```json\n{"verdict": "SUPPORTED", "confidence": 0.85}'
    cleaned = clean_json_text(raw)
    parsed = json.loads(cleaned)
    assert parsed["verdict"] == "SUPPORTED"
    assert parsed["confidence"] == 0.85


def test_parse_json_payload_strict_success():
    """Verify standard strict JSON parsing works correctly."""
    raw = '{"verdict": "SUPPORTED", "confidence": 0.99}'
    data = parse_json_payload(raw)
    assert data["verdict"] == "SUPPORTED"
    assert data["confidence"] == 0.99


def test_parse_json_payload_unescaped_control_characters():
    """Verify parser handles literal unescaped newlines and tabs inside strings."""
    raw = '{\n  "clinical_explanation": "Point 1: First finding.\nPoint 2: Second finding.",\n  "confidence": 0.85\n}'
    data = parse_json_payload(raw)
    assert isinstance(data, dict)
    assert "Point 1: First finding.\nPoint 2: Second finding." in data["clinical_explanation"]
    assert data["confidence"] == 0.85


def test_parse_json_payload_comments_stripping():
    """Verify parser removes inline and block comments from JSON."""
    raw = """{
        // Clinical verdict
        "verdict": "SUPPORTED",
        /* Confidence assessment */
        "confidence": 0.91
    }"""
    data = parse_json_payload(raw)
    assert data["verdict"] == "SUPPORTED"
    assert data["confidence"] == 0.91


def test_parse_json_payload_python_literal_eval_fallback():
    """Verify parser falls back to ast.literal_eval for Python single-quoted dicts with True/False/None."""
    raw = "{'is_sufficient': True, 'confidence': 0.88, 'rationale': 'Direct evidence', 'search_keywords': None}"
    data = parse_json_payload(raw)
    assert data["is_sufficient"] is True
    assert data["confidence"] == 0.88
    assert data["search_keywords"] is None


def test_parse_json_payload_escaped_quotes_and_nested():
    """Verify parser handles nested structures with escaped quotes."""
    raw = '{"title": "Study \\"Alpha\\"", "findings": ["Evidence 1", "Evidence 2"]}'
    data = parse_json_payload(raw)
    assert data["title"] == 'Study "Alpha"'
    assert len(data["findings"]) == 2


def test_generate_json_list_wrapped_object_unwrapping():
    """Verify generate_json unwraps a single-element list containing a dictionary."""
    gateway = LLMGateway(Settings(openrouter_api_key="mock_key"))
    with patch.object(
        gateway,
        "generate",
        return_value=('[{"verdict": "SUPPORTED", "confidence": 0.95}]', "openrouter"),
    ):
        data, provider = gateway.generate_json("Assess claim")
        assert isinstance(data, dict)
        assert data["verdict"] == "SUPPORTED"
        assert provider == "openrouter"


def test_generate_json_empty_or_whitespace_raises_runtime_error():
    """Verify generate_json raises RuntimeError when LLM returns empty or whitespace response."""
    gateway = LLMGateway(Settings(openrouter_api_key="mock_key"))
    with (
        patch.object(gateway, "generate", return_value=("", "openrouter")),
        pytest.raises(RuntimeError, match="empty or whitespace-only response"),
    ):
        gateway.generate_json("Assess claim")

    with (
        patch.object(gateway, "generate", return_value=("   \t  \n  ", "openrouter")),
        pytest.raises(RuntimeError, match="empty or whitespace-only response"),
    ):
        gateway.generate_json("Assess claim")


def test_generate_json_empty_dict_raises_runtime_error():
    """Verify generate_json raises RuntimeError when LLM returns an empty JSON object."""
    gateway = LLMGateway(Settings(openrouter_api_key="mock_key"))
    with (
        patch.object(gateway, "generate", return_value=("{}", "openrouter")),
        pytest.raises(RuntimeError, match="empty JSON dictionary"),
    ):
        gateway.generate_json("Assess claim")


def test_generate_json_non_json_refusal_raises_runtime_error():
    """Verify generate_json raises actionable RuntimeError when LLM returns conversational refusal without JSON."""
    gateway = LLMGateway(Settings(openrouter_api_key="mock_key"))
    refusal = "I am an AI assistant and cannot provide medical evaluations."
    with (
        patch.object(gateway, "generate", return_value=(refusal, "openrouter")),
        pytest.raises(RuntimeError, match="Model failed to emit valid JSON after multi-stage extraction"),
    ):
        gateway.generate_json("Assess claim")


def test_generate_json_primitive_type_raises_runtime_error():
    """Verify generate_json raises RuntimeError when model emits a JSON primitive rather than a dict."""
    gateway = LLMGateway(Settings(openrouter_api_key="mock_key"))
    with (
        patch.object(gateway, "generate", return_value=("42", "openrouter")),
        pytest.raises(RuntimeError, match="expected a dictionary, got int"),
    ):
        gateway.generate_json("Assess claim")


def test_generate_json_provider_attribution():
    """Verify provider attribution is accurately preserved for both gemini and openrouter."""
    gateway = LLMGateway(Settings(gemini_api_key="mock_gemini", openrouter_api_key="mock_openrouter"))

    with patch.object(gateway, "generate", return_value=('{"verdict": "SUPPORTED"}', "gemini")):
        _, provider = gateway.generate_json("Test")
        assert provider == "gemini"

    with patch.object(gateway, "generate", return_value=('{"verdict": "SUPPORTED"}', "openrouter")):
        _, provider = gateway.generate_json("Test")
        assert provider == "openrouter"


# ==============================================================================
# OpenRouter Compliance & Error Parsing Tests
# ==============================================================================


def test_openrouter_payload_headers_and_token_budget():
    """Verify headers, max_tokens, and temperature in standard OpenRouter generation."""
    settings = Settings(openrouter_api_key="sk-or-test-key", request_timeout=12.0)
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Sample content"}, "finish_reason": "stop"}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        resp, provider = gateway.generate("Hello user", system_prompt="System rules")
        assert provider == "openrouter"
        assert resp == "Sample content"
        assert mock_post.called

        call_kwargs = mock_post.call_args[1]
        headers = call_kwargs["headers"]
        payload = call_kwargs["json"]

        assert headers["Authorization"] == "Bearer sk-or-test-key"
        assert headers["HTTP-Referer"] == "https://github.com/medical-crag"
        assert headers["X-Title"] == "Medical-CRAG"
        assert headers["Accept"] == "application/json"
        assert payload["model"] == settings.openrouter_model
        assert payload["temperature"] == 0.1
        assert payload["max_tokens"] == 2048
        assert "response_format" not in payload


def test_openrouter_json_mode_response_format():
    """Verify generate_json includes response_format={'type': 'json_object'}."""
    settings = Settings(openrouter_api_key="sk-or-test-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"is_valid": true}'}, "finish_reason": "stop"}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        data, provider = gateway.generate_json("Prompt asking for JSON")
        assert provider == "openrouter"
        assert data == {"is_valid": True}

        payload = mock_post.call_args[1]["json"]
        assert payload.get("response_format") == {"type": "json_object"}
        assert payload.get("max_tokens") == 2048


def test_openrouter_http_401_auth_error():
    """Verify clean RuntimeError with 401 message extraction."""
    settings = Settings(openrouter_api_key="sk-or-bad-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": {"message": "User not found.", "code": 401}}

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="HTTP 401"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_http_402_insufficient_credits():
    """Verify clean RuntimeError with 402 payment required extraction."""
    settings = Settings(openrouter_api_key="sk-or-valid-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.json.return_value = {"error": {"message": "Insufficient credits", "code": 402}}

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="HTTP 402.*credits"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_http_429_rate_limit():
    """Verify clean RuntimeError with 429 rate limit extraction."""
    settings = Settings(openrouter_api_key="sk-or-valid-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded", "code": 429}}

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match=r"(?i)HTTP 429.*rate limit"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_http_500_server_error():
    """Verify clean RuntimeError with 500 server error extraction."""
    settings = Settings(openrouter_api_key="sk-or-valid-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": {"message": "Provider failed upstream", "code": 500}}

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match=r"(?i)server error \(HTTP 500\)"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_timeout_exception():
    """Verify httpx.TimeoutException is converted to descriptive RuntimeError."""
    settings = Settings(openrouter_api_key="sk-or-valid-key", request_timeout=5.0)
    gateway = LLMGateway(settings)

    with (
        patch("httpx.Client.post", side_effect=httpx.ReadTimeout("Read timed out")),
        pytest.raises(RuntimeError, match="timed out after 5.0s"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_connection_error():
    """Verify httpx.ConnectError is converted to descriptive RuntimeError."""
    settings = Settings(openrouter_api_key="sk-or-valid-key")
    gateway = LLMGateway(settings)

    with (
        patch("httpx.Client.post", side_effect=httpx.ConnectError("Failed to resolve host")),
        pytest.raises(RuntimeError, match="connection failed"),
    ):
        gateway._call_openrouter("Test")


def test_openrouter_content_safety_filter():
    """Verify content_filter finish reason raises descriptive error."""
    settings = Settings(openrouter_api_key="sk-or-valid-key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]
    }

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="blocked by content safety"),
    ):
        gateway._call_openrouter("Unsafe query")


# ==============================================================================
# Google Gemini REST & Safety Block Tests
# ==============================================================================


def test_gemini_rest_headers_payload_and_url():
    """Verify Gemini REST URL, x-goog-api-key header, query parameter, and payload."""
    settings = Settings(gemini_api_key="test_gemini_key", gemini_model="models/gemini-1.5-flash")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": '{"result": "ok"}'}]},
                "finishReason": "STOP",
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        data, provider = gateway.generate_json("Medical prompt", system_prompt="System instructions")
        assert provider == "gemini"
        assert data == {"result": "ok"}
        assert mock_post.called

        url = mock_post.call_args[0][0]
        # URL must NOT contain duplicate models/ prefix
        assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        headers = mock_post.call_args[1]["headers"]
        params = mock_post.call_args[1]["params"]
        payload = mock_post.call_args[1]["json"]

        assert headers["x-goog-api-key"] == "test_gemini_key"
        assert params["key"] == "test_gemini_key"
        assert payload["contents"][0]["parts"][0]["text"] == "Medical prompt"
        assert payload["systemInstruction"]["parts"][0]["text"] == "System instructions"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_rest_safety_block_candidate():
    """Verify finishReason == 'SAFETY' raises RuntimeError containing SAFETY."""
    settings = Settings(gemini_api_key="test_gemini_key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "finishReason": "SAFETY",
                "safetyRatings": [{"category": "HARM_DANGEROUS", "blocked": True}],
            }
        ]
    }

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="SAFETY"),
    ):
        gateway._call_gemini("Hazardous medical prompt")


def test_gemini_rest_safety_block_prompt_feedback():
    """Verify promptFeedback.blockReason raises RuntimeError."""
    settings = Settings(gemini_api_key="test_gemini_key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [],
    }

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="prompt blocked due to SAFETY"),
    ):
        gateway._call_gemini("Hazardous prompt")


def test_gemini_rest_http_403_permission_denied():
    """Verify Gemini HTTP 403 raises descriptive RuntimeError."""
    settings = Settings(gemini_api_key="test_gemini_key")
    gateway = LLMGateway(settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"error": {"message": "Method doesn't allow unregistered callers"}}

    with (
        patch("httpx.Client.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="HTTP 403"),
    ):
        gateway._call_gemini("Prompt")


def test_gemini_rest_timeout_exception():
    """Verify Gemini httpx.TimeoutException is wrapped into descriptive RuntimeError."""
    settings = Settings(gemini_api_key="test_gemini_key", request_timeout=3.0)
    gateway = LLMGateway(settings)

    with (
        patch("httpx.Client.post", side_effect=httpx.ReadTimeout("Timeout")),
        pytest.raises(RuntimeError, match="timed out after 3.0s"),
    ):
        gateway._call_gemini("Prompt")


# ==============================================================================
# Multi-Provider Fallback Chain Tests
# ==============================================================================


def test_fallback_chain_primary_openrouter():
    """Verify primary_provider='openrouter' attempts OpenRouter first."""
    settings = Settings(
        gemini_api_key="mock_gemini",
        openrouter_api_key="mock_openrouter",
    )
    # Dynamically assign primary_provider
    settings.primary_provider = "openrouter"
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_openrouter", return_value="OpenRouter text") as mock_openrouter,
        patch.object(gateway, "_call_gemini", return_value="Gemini text") as mock_gemini,
    ):
        text, provider = gateway.generate("Test prompt")
        assert text == "OpenRouter text"
        assert provider == "openrouter"
        assert mock_openrouter.called
        assert not mock_gemini.called


def test_fallback_chain_primary_gemini_fails_then_openrouter_succeeds():
    """Verify Gemini failure automatically triggers OpenRouter fallback."""
    settings = Settings(
        gemini_api_key="mock_gemini",
        openrouter_api_key="mock_openrouter",
    )
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini", side_effect=RuntimeError("Gemini API error (HTTP 429)")),
        patch.object(gateway, "_call_openrouter", return_value="OpenRouter fallback text") as mock_openrouter,
    ):
        text, provider = gateway.generate("Test prompt")
        assert text == "OpenRouter fallback text"
        assert provider == "openrouter"
        assert mock_openrouter.called


def test_single_key_configured_only_calls_configured_provider():
    """Verify that when only OpenRouter key is present, Gemini is skipped without making network calls."""
    settings = Settings(
        gemini_api_key="",
        openrouter_api_key="mock_openrouter",
    )
    gateway = LLMGateway(settings)

    with (
        patch.object(gateway, "_call_gemini") as mock_gemini,
        patch.object(gateway, "_call_openrouter", return_value="OpenRouter direct") as mock_openrouter,
    ):
        text, provider = gateway.generate("Test prompt")
        assert text == "OpenRouter direct"
        assert provider == "openrouter"
        assert not mock_gemini.called
        assert mock_openrouter.called
