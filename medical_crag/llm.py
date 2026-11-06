"""Multi-provider LLM gateway supporting Google Gemini with OpenRouter fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple
import httpx
from medical_crag.config import Settings

logger = logging.getLogger(__name__)


def clean_json_text(text: str) -> str:
    """Strip markdown code fence if model wraps JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


class LLMGateway:
    """Unified LLM client providing automatic failover from Gemini to OpenRouter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _call_gemini(self, prompt: str, system_prompt: str = "") -> str:
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=system_prompt if system_prompt else None,
        )
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise RuntimeError("Gemini returned empty response")
        return response.text

    def _call_openrouter(self, prompt: str, system_prompt: str = "") -> str:
        if not self.settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/medical-crag",
            "X-Title": "Medical-CRAG",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=self.settings.request_timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices or not choices[0].get("message", {}).get("content"):
                raise RuntimeError("OpenRouter returned empty choices")
            return choices[0]["message"]["content"]

    def generate(self, prompt: str, system_prompt: str = "") -> Tuple[str, str]:
        """Generate response trying Gemini first, falling back to OpenRouter."""
        errors = []

        # 1. Try Gemini
        if self.settings.gemini_api_key:
            try:
                text = self._call_gemini(prompt, system_prompt)
                return text, "gemini"
            except Exception as e:
                logger.warning("Gemini generation failed: %s; falling back to OpenRouter", e)
                errors.append(f"Gemini: {e}")

        # 2. Try OpenRouter
        if self.settings.openrouter_api_key:
            try:
                text = self._call_openrouter(prompt, system_prompt)
                return text, "openrouter"
            except Exception as e:
                logger.warning("OpenRouter generation failed: %s", e)
                errors.append(f"OpenRouter: {e}")

        raise RuntimeError(f"All LLM providers failed or no keys configured: {'; '.join(errors)}")

    def generate_json(self, prompt: str, system_prompt: str = "") -> Tuple[Dict[str, Any], str]:
        """Generate text and parse as JSON."""
        raw_text, provider = self.generate(prompt, system_prompt)
        cleaned = clean_json_text(raw_text)
        try:
            data = json.loads(cleaned)
            return data, provider
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Model failed to emit valid JSON: {cleaned}") from err
