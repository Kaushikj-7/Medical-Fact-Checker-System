import json
import pytest
from unittest.mock import MagicMock, patch
from medical_crag.config import Settings
from medical_crag.llm import LLMGateway


def test_gemini_success(monkeypatch):
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

    with patch.object(gateway, "_call_gemini", side_effect=Exception("Gemini 429 Rate Limited")):
        with patch.object(gateway, "_call_openrouter", return_value="OpenRouter response"):
            resp, provider = gateway.generate("test prompt")
            assert resp == "OpenRouter response"
            assert provider == "openrouter"


def test_both_providers_fail_raises_runtime_error():
    settings = Settings(
        gemini_api_key="mock_gemini_key",
        openrouter_api_key="mock_openrouter_key",
    )
    gateway = LLMGateway(settings)

    with patch.object(gateway, "_call_gemini", side_effect=Exception("Gemini error")):
        with patch.object(gateway, "_call_openrouter", side_effect=Exception("OpenRouter error")):
            with pytest.raises(RuntimeError, match="All LLM providers failed"):
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
