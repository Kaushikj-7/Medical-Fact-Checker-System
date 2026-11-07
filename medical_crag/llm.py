"""Multi-provider LLM gateway supporting Google Gemini REST with OpenRouter fallback."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

import httpx

from medical_crag.config import Settings

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1


def _is_valid_json_or_dict(candidate: str) -> bool:
    """Check if candidate text parses as a JSON object/array or Python dict/list."""
    if not candidate:
        return False
    # Strip trailing commas
    s = re.sub(r",+\s*([}\]])", r"\1", candidate.strip())
    # 1. Strict json
    try:
        val = json.loads(s)
        if isinstance(val, (dict, list)):
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # 2. Relaxed json (control characters like literal newlines/tabs)
    try:
        val = json.loads(s, strict=False)
        if isinstance(val, (dict, list)):
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # 3. Comments stripped
    no_comments = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", s)
    no_comments = re.sub(r",+\s*([}\]])", r"\1", no_comments)
    try:
        val = json.loads(no_comments, strict=False)
        if isinstance(val, (dict, list)):
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # 4. ast.literal_eval for Python single-quoted dicts
    try:
        val = ast.literal_eval(no_comments)
        if isinstance(val, (dict, list)):
            return True
    except (ValueError, SyntaxError, MemoryError, TypeError):
        pass
    # 5. Python keyword normalization fallback
    try:
        py_to_json = re.sub(r"\bTrue\b", "true", no_comments)
        py_to_json = re.sub(r"\bFalse\b", "false", py_to_json)
        py_to_json = re.sub(r"\bNone\b", "null", py_to_json)
        py_to_json = re.sub(
            r"(?<=[\{\s,\[])'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[:,\}\]])",
            r'"\1"',
            py_to_json,
        )
        val = json.loads(py_to_json, strict=False)
        if isinstance(val, (dict, list)):
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return False


def _find_balanced_json_substring(text: str) -> str:
    """Locate the outermost balanced JSON object {...} or array [...] avoiding preambles/postscripts."""
    n = len(text)
    candidates: list[str] = []

    i = 0
    while i < n:
        ch = text[i]
        if ch in ("{", "["):
            open_char = ch
            stack = [open_char]
            in_string: str | None = None
            escape = False
            j = i + 1
            while j < n and stack:
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\":
                    if in_string:
                        escape = True
                elif in_string:
                    if c == in_string:
                        in_string = None
                else:
                    # Skip single-line and multi-line comments outside strings
                    if c == "/" and j + 1 < n:
                        next_c = text[j + 1]
                        if next_c == "/":
                            nl = text.find("\n", j + 2)
                            j = n if nl == -1 else nl
                            continue
                        if next_c == "*":
                            end_c = text.find("*/", j + 2)
                            j = n if end_c == -1 else end_c + 1
                            continue
                    if c == '"':
                        in_string = '"'
                    elif c == "'":
                        in_string = "'"
                    elif c in ("{", "["):
                        stack.append(c)
                    elif c in ("}", "]"):
                        expected = "}" if stack[-1] == "{" else "]"
                        if c == expected:
                            stack.pop()
                        else:
                            break
                j += 1

            if not stack:
                cand = text[i:j].strip()
                candidates.append(cand)
                i = j - 1
        i += 1

    if not candidates:
        return text

    # Select valid candidates
    valid_candidates = [c for c in candidates if _is_valid_json_or_dict(c)]
    if valid_candidates:
        return max(valid_candidates, key=len)

    # Fallback: candidates formatted with key-value or non-empty array
    kv_candidates = [
        c for c in candidates
        if (c.strip().startswith("{") and ":" in c) or (c.strip().startswith("[") and len(c.strip()) > 2)
    ]
    if kv_candidates:
        return max(kv_candidates, key=len)

    # Fallback to largest balanced candidate if none passed validation
    return max(candidates, key=len)


def clean_json_text(text: str) -> str:
    """Clean LLM output by removing markdown fences, conversational banter, and trailing commas.

    Multi-stage extraction:
    1. Strip markdown code fences (```json ... ``` or ``` ... ```, case-insensitive),
       preserving internal backticks in JSON strings and handling unclosed fences.
    2. Extract outermost balanced JSON object {...} or array [...], skipping non-JSON
       braces in conversational preambles (e.g. equations) and postscripts (e.g. template tags).
    3. Strip trailing commas before closing braces '}' and brackets ']'.
    """
    if not text:
        return ""
    text = text.strip()

    # Stage 1: Strip markdown code blocks (boundary-aware)
    m = re.search(
        r"(?:^|\n)[ \t]*(`{3,})[ \t]*json[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*\1[ \t]*(?:\r?\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        text = m.group(2).strip()
    else:
        m = re.search(
            r"^\s*(`{3,})(?:[ \t]*[a-zA-Z0-9_-]+)?\s*[\r\n]+([\s\S]*?)\s*[\r\n]+\s*\1\s*$",
            text,
            re.IGNORECASE,
        )
        if m:
            text = m.group(2).strip()
        else:
            m = re.search(
                r"(?:^|\n)[ \t]*(`{3,})(?:[ \t]*[a-zA-Z0-9_-]+)?[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*\1[ \t]*(?:\r?\n|$)",
                text,
                re.IGNORECASE,
            )
            if m:
                text = m.group(2).strip()
            elif text.startswith("```"):
                text = re.sub(r"^`{3,}(?:[ \t]*[a-zA-Z0-9_-]+)?\s*", "", text, flags=re.IGNORECASE).strip()
                text = re.sub(r"`{3,}$", "", text).strip()

    # Stage 2: Outermost enclosing balanced JSON object or array
    text = _find_balanced_json_substring(text)

    # Stage 3: Trailing comma stripping (commas immediately preceding '}' or ']')
    text = re.sub(r",+\s*([}\]])", r"\1", text)

    return text


def parse_json_payload(raw_text: str) -> Any:
    """Parse raw text from LLM into Python data structures using a multi-stage resilient strategy.

    Stages:
    1. Preprocessing via clean_json_text (fences, preambles, trailing commas).
    2. Strict json.loads (fast path).
    3. Relaxed json.loads(strict=False) for unescaped control chars (e.g. newlines inside strings).
    4. Comment stripping (// and /* */).
    5. Python literal fallback via ast.literal_eval (single quotes, True/False/None).
    6. Keyword and quote normalization fallback.
    7. Clear, actionable error raising if all stages fail.
    """
    if not raw_text or not raw_text.strip():
        raise RuntimeError("Model emitted empty or whitespace-only response; cannot parse JSON")

    cleaned = clean_json_text(raw_text)
    if not cleaned:
        raise RuntimeError(f"Model response contained no JSON structure after cleaning: {raw_text!r}")

    # Stage 4a: Strict json.loads
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Stage 4b: Relaxed json.loads
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Stage 4c: Comment stripping
    no_comments = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", cleaned)
    no_comments = re.sub(r",+\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_comments, strict=False)
    except json.JSONDecodeError:
        pass

    # Stage 4d: Safe Python literal eval (for single-quoted dicts/booleans/None)
    try:
        val = ast.literal_eval(no_comments)
        if isinstance(val, (dict, list)):
            return val
    except (ValueError, SyntaxError, MemoryError, TypeError):
        pass

    # Stage 4e: Keyword and quote normalization fallback
    try:
        py_to_json = re.sub(r"\bTrue\b", "true", no_comments)
        py_to_json = re.sub(r"\bFalse\b", "false", py_to_json)
        py_to_json = re.sub(r"\bNone\b", "null", py_to_json)
        py_to_json = re.sub(
            r"(?<=[\{\s,\[])'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[:,\}\]])",
            r'"\1"',
            py_to_json,
        )
        return json.loads(py_to_json, strict=False)
    except json.JSONDecodeError:
        pass

    # Stage 5: Actionable error raising
    raise RuntimeError(
        f"Model failed to emit valid JSON after multi-stage extraction. Sanitized payload was: {cleaned!r}"
    )


def _parse_openrouter_error(resp: httpx.Response) -> str:
    """Extract actionable error message from OpenRouter response body."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and "message" in err:
                return str(err["message"])
            if isinstance(err, str):
                return err
            if "message" in data:
                return str(data["message"])
    except (ValueError, json.JSONDecodeError):
        pass
    text_snippet = resp.text[:200].strip() if resp.text else ""
    return text_snippet or resp.reason_phrase or "Unknown HTTP error"


class LLMGateway:
    """Unified LLM client providing automatic failover between Gemini and OpenRouter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Execute Google Gemini completion via direct REST API with JSON mode support."""
        api_key = (self.settings.gemini_api_key or "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        model_name = self.settings.gemini_model.strip().removeprefix("models/")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        params = {"key": api_key}

        effective_temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
        generation_config: dict[str, Any] = {
            "temperature": effective_temp,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

        if system_prompt and system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt.strip()}]
            }

        timeout = self.settings.request_timeout

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, params=params, json=payload)
        except httpx.TimeoutException as exc:
            logger.error("Gemini request timed out after %.1fs", timeout)
            raise RuntimeError(f"Gemini request timed out after {timeout:.1f}s: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Gemini network connection error: %s", exc)
            raise RuntimeError(
                f"Gemini connection failed: {exc}. Verify network and DNS reachability."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Gemini transport error: %s", exc)
            raise RuntimeError(f"Gemini transport error: {exc}") from exc

        if resp.status_code != 200:
            err_msg = ""
            try:
                err_data = resp.json()
                if isinstance(err_data, dict):
                    err_obj = err_data.get("error", {})
                    if isinstance(err_obj, dict):
                        err_msg = err_obj.get("message", "")
                    elif isinstance(err_obj, str):
                        err_msg = err_obj
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                err_msg = resp.text[:200].strip() if resp.text else ""
            raise RuntimeError(
                f"Gemini API error (HTTP {resp.status_code}): {err_msg or resp.reason_phrase or 'Unknown error'}"
            )

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            snippet = resp.text[:200].strip() if resp.text else ""
            raise RuntimeError(f"Gemini returned invalid JSON (HTTP 200): {snippet}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Gemini returned non-dict JSON: {type(data).__name__}")  # noqa: TRY004

        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini prompt blocked due to {block_reason}")

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini returned no response candidates")

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise RuntimeError("Gemini candidate is not a dictionary")  # noqa: TRY004

        finish_reason = candidate.get("finishReason")
        if finish_reason == "SAFETY":
            safety_ratings = candidate.get("safetyRatings", [])
            raise RuntimeError(f"Gemini response blocked due to SAFETY: {safety_ratings}")
        if finish_reason in ("RECITATION", "OTHER"):
            raise RuntimeError(f"Gemini generation halted with finishReason: {finish_reason}")

        content = candidate.get("content")
        if not isinstance(content, dict):
            raise RuntimeError("Gemini candidate content is missing or not a dictionary")  # noqa: TRY004

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], dict) or "text" not in parts[0]:
            raise RuntimeError("Gemini candidate contains no text parts")

        text = str(parts[0]["text"]).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")

        return text

    def _call_openrouter(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Call OpenRouter Chat Completions API with compliant headers and resilient error handling."""
        api_key = (self.settings.openrouter_api_key or "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/medical-crag",
            "X-Title": "Medical-CRAG",
        }

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        effective_temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
        effective_max_tokens = max_tokens if max_tokens is not None else DEFAULT_OPENROUTER_MAX_TOKENS

        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "temperature": effective_temp,
            "max_tokens": effective_max_tokens,
        }

        if json_mode or response_format:
            payload["response_format"] = response_format or {"type": "json_object"}

        timeout = self.settings.request_timeout

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            logger.error("OpenRouter request timed out after %.1fs", timeout)
            raise RuntimeError(
                f"OpenRouter request timed out after {timeout:.1f}s: {exc}"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("OpenRouter network connection error: %s", exc)
            raise RuntimeError(
                f"OpenRouter connection failed: {exc}. Verify network and DNS reachability."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("OpenRouter HTTP request error: %s", exc)
            raise RuntimeError(f"OpenRouter transport error: {exc}") from exc

        if resp.status_code != 200:
            err_msg = _parse_openrouter_error(resp)
            status = resp.status_code

            if status == 401:
                logger.error("OpenRouter authentication failed (401): %s", err_msg)
                raise RuntimeError(
                    f"OpenRouter authentication failed (HTTP 401): {err_msg}. "
                    "Please verify OPENROUTER_API_KEY in .env."
                )
            if status == 402:
                logger.error("OpenRouter insufficient credits (402): %s", err_msg)
                raise RuntimeError(
                    f"OpenRouter payment/credits required (HTTP 402): {err_msg}. "
                    "Account balance insufficient."
                )
            if status == 429:
                logger.warning("OpenRouter rate limit exceeded (429): %s", err_msg)
                raise RuntimeError(
                    f"OpenRouter rate limit exceeded (HTTP 429): {err_msg}. "
                    "Rate limit reached for provider/model."
                )
            if status in (500, 502, 503, 504):
                logger.error("OpenRouter server error (%d): %s", status, err_msg)
                raise RuntimeError(
                    f"OpenRouter server error (HTTP {status}): {err_msg}. "
                    "Upstream model or provider temporarily unavailable."
                )
            logger.error("OpenRouter request error (%d): %s", status, err_msg)
            raise RuntimeError(
                f"OpenRouter request failed (HTTP {status}): {err_msg}"
            )

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            snippet = resp.text[:200].strip() if resp.text else ""
            raise RuntimeError(
                f"OpenRouter returned invalid JSON (HTTP 200): {snippet}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"OpenRouter returned non-dict JSON: {type(data).__name__}")  # noqa: TRY004

        if "error" in data:
            err_obj = data["error"]
            err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
            raise RuntimeError(f"OpenRouter returned API error: {err_msg}")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter returned empty choices list")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenRouter choice item is not a dictionary")  # noqa: TRY004

        finish_reason = first_choice.get("finish_reason")
        if finish_reason == "content_filter":
            raise RuntimeError("OpenRouter response was blocked by content safety filters")
        if finish_reason == "length":
            logger.warning(
                "OpenRouter response reached max_tokens (%d); output may be truncated",
                effective_max_tokens,
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenRouter message object missing in choice")  # noqa: TRY004

        content = message.get("content")
        if content is None or not str(content).strip():
            raise RuntimeError("OpenRouter returned empty message content")

        return str(content).strip()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str]:
        """Generate response attempting primary provider first, falling back to secondary."""
        primary = getattr(self.settings, "primary_provider", "gemini")
        if isinstance(primary, str) and primary.lower() == "openrouter":
            providers = [
                ("openrouter", self._call_openrouter, self.settings.openrouter_api_key),
                ("gemini", self._call_gemini, self.settings.gemini_api_key),
            ]
        else:
            providers = [
                ("gemini", self._call_gemini, self.settings.gemini_api_key),
                ("openrouter", self._call_openrouter, self.settings.openrouter_api_key),
            ]

        errors: list[str] = []
        for name, call_fn, key in providers:
            if not key:
                continue
            try:
                try:
                    text = call_fn(
                        prompt,
                        system_prompt=system_prompt,
                        json_mode=json_mode,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except TypeError:
                    try:
                        text = call_fn(prompt, system_prompt=system_prompt, json_mode=json_mode)
                    except TypeError:
                        text = call_fn(prompt, system_prompt=system_prompt)
                return text, name
            except Exception as e:  # noqa: BLE001
                logger.warning("%s generation failed: %s; attempting fallback", name.capitalize(), e)
                errors.append(f"{name}: {e}")

        raise RuntimeError(f"All LLM providers failed or no keys configured: {'; '.join(errors)}")

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_name: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Generate structured text in JSON mode and parse safely with multi-stage resilience.

        Returns:
            tuple[dict[str, Any], str]: (parsed_json_dict, provider_name)

        Raises:
            RuntimeError: If model response cannot be parsed into a valid JSON dictionary or is empty.
        """
        raw_text, provider = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            max_tokens=max_tokens,
        )
        data = parse_json_payload(raw_text)

        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]

        if not isinstance(data, dict):
            raise RuntimeError(  # noqa: TRY004
                f"Model emitted valid JSON but expected a dictionary, got {type(data).__name__:s}: {raw_text}"
            )

        if not data:
            raise RuntimeError(f"Model emitted an empty JSON dictionary: {raw_text}")

        return data, provider
