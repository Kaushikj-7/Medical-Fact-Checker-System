"""Configuration settings for Medical CRAG."""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

OPENROUTER_KEY_REGEX = re.compile(r"^sk-or-v1-[0-9a-fA-F]{64}$")
GEMINI_KEY_REGEX = re.compile(
    r"^(AIzaSy[A-Za-z0-9_-]{33}|AIza[A-Za-z0-9_-]{35}|AQ\.[A-Za-z0-9_.-]{20,})$"
)

VALID_PRIMARY_PROVIDERS: tuple[str, ...] = ("gemini", "openrouter")
DEFAULT_PRIMARY_PROVIDER: str = "gemini"


def clean_env_value(val: str | None) -> str:
    """Trim leading and trailing whitespace and surrounding quotes."""
    if val is None:
        return ""
    s = str(val).strip()
    while (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def mask_key(key: str | None) -> str:
    """Safely mask API key for logging without leaking short secret tokens."""
    if not key:
        return "<unset>"
    cleaned = str(key).strip()
    if not cleaned:
        return "<unset>"
    length = len(cleaned)
    if length <= 16:
        return "***"
    if length <= 24:
        return f"{cleaned[:2]}...{cleaned[-2:]}"
    if length < 39:
        return f"{cleaned[:4]}...{cleaned[-4:]}"
    return f"{cleaned[:8]}...{cleaned[-4:]}"


def validate_openrouter_key(key: str) -> tuple[bool, str | None]:
    """Validate OpenRouter API key format without raising exceptions."""
    if not key:
        return False, "OpenRouter API key is empty or unset."
    if not key.startswith("sk-or-v1-"):
        return False, f"OpenRouter key must start with 'sk-or-v1-' (found: {key[:10]!r})."
    if len(key) != 73:
        return False, f"OpenRouter key must be 73 characters long (found length: {len(key)})."
    if not OPENROUTER_KEY_REGEX.match(key):
        return False, "OpenRouter key suffix contains invalid characters (expected 64 hex characters)."
    return True, None


def validate_gemini_key(key: str) -> tuple[bool, str | None]:
    """Validate Google Gemini API key format without raising exceptions."""
    if not key:
        return False, "Gemini API key is empty or unset."
    if not GEMINI_KEY_REGEX.match(key):
        return False, (
            "Gemini key does not match recognized patterns "
            f"('AIzaSy...', 'AIza...', or 'AQ...'; found: {key[:8]!r})."
        )
    return True, None


_DEFAULT_ENV_LOADED: bool = False


def load_environment(
    env_file: str | Path | None = None,
    force_reload: bool = False,
) -> dict[str, str]:
    """Load environment variables from .env files safely and idempotently.

    Args:
        env_file: Optional explicit file path. If provided, parses file values
            without mutating global os.environ. If None, performs ambient discovery
            once per process.
        force_reload: If True, forces re-running ambient discovery even if previously run.

    Returns:
        Dict of values loaded if an explicit env_file was given, else empty dict.
    """
    global _DEFAULT_ENV_LOADED

    try:
        from dotenv import dotenv_values, find_dotenv, load_dotenv
    except ImportError:
        return {}

    if env_file is not None:
        target_path = Path(env_file).resolve()
        if not target_path.is_file():
            logger.warning("Specified env file does not exist: %s", target_path)
            return {}
        try:
            values = dotenv_values(dotenv_path=target_path)
            return {k: str(v) for k, v in values.items() if v is not None}
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(
                "Failed to parse env file %s: %s; falling back to ambient environment",
                target_path,
                e,
            )
            return {}

    # Default discovery: run once per process to preserve ambient and monkeypatched deletions
    if _DEFAULT_ENV_LOADED and not force_reload:
        return {}

    try:
        # Default discovery: cwd / ".env", then package root parent / ".env", then find_dotenv
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file():
            load_dotenv(dotenv_path=cwd_env, override=False)

        root_env = Path(__file__).resolve().parent.parent / ".env"
        if root_env.is_file() and root_env.resolve() != cwd_env.resolve():
            load_dotenv(dotenv_path=root_env, override=False)

        pkg_env = Path(__file__).resolve().parent / ".env"
        if pkg_env.is_file() and pkg_env.resolve() not in (cwd_env.resolve(), root_env.resolve()):
            load_dotenv(dotenv_path=pkg_env, override=False)

        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(dotenv_path=found, override=False)
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Error loading default .env files: %s", e)
    finally:
        _DEFAULT_ENV_LOADED = True

    return {}


# Run loader on import for backwards compatibility
load_environment()


def _extract_env_var(
    names: Sequence[str],
    default: str = "",
    source_dict: dict[str, str] | None = None,
) -> str:
    """Extract first non-empty cleaned environment variable matching names."""
    if source_dict:
        for name in names:
            val = source_dict.get(name)
            if val is not None:
                cleaned = clean_env_value(val)
                if cleaned:
                    return cleaned

    for name in names:
        val = os.getenv(name)
        if val is not None:
            cleaned = clean_env_value(val)
            if cleaned:
                return cleaned
    return default


def _extract_int_env(
    name: str,
    default: int,
    source_dict: dict[str, str] | None = None,
    min_val: int | None = 1,
    max_val: int | None = None,
) -> int:
    """Safely parse integer environment variable with boundary validation and fallback."""
    raw = None
    if source_dict and name in source_dict:
        raw = source_dict.get(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = clean_env_value(raw)
    if not cleaned:
        return default
    try:
        val = int(cleaned)
    except (ValueError, TypeError):
        logger.warning("Invalid integer for %s: %r; defaulting to %d", name, raw, default)
        return default

    if min_val is not None and val < min_val:
        logger.warning(
            "Integer for %s (%d) is below minimum %d; defaulting to %d",
            name,
            val,
            min_val,
            default,
        )
        return default
    if max_val is not None and val > max_val:
        logger.warning(
            "Integer for %s (%d) is above maximum %d; defaulting to %d",
            name,
            val,
            max_val,
            default,
        )
        return default
    return val


def _extract_float_env(
    name: str,
    default: float,
    source_dict: dict[str, str] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    must_be_positive: bool = False,
) -> float:
    """Safely parse float environment variable with NaN/Inf and boundary checks."""
    raw = None
    if source_dict and name in source_dict:
        raw = source_dict.get(name)
    if raw is None:
        raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = clean_env_value(raw)
    if not cleaned:
        return default
    try:
        val = float(cleaned)
    except (ValueError, TypeError):
        logger.warning("Invalid float for %s: %r; defaulting to %f", name, raw, default)
        return default

    if math.isnan(val) or math.isinf(val):
        logger.warning("Float for %s is NaN or Inf: %r; defaulting to %f", name, raw, default)
        return default

    if must_be_positive and val <= 0.0:
        logger.warning(
            "Float for %s (%f) must be positive; defaulting to %f",
            name,
            val,
            default,
        )
        return default

    if min_val is not None and val < min_val:
        logger.warning(
            "Float for %s (%f) is below minimum %f; defaulting to %f",
            name,
            val,
            min_val,
            default,
        )
        return default

    if max_val is not None and val > max_val:
        logger.warning(
            "Float for %s (%f) is above maximum %f; defaulting to %f",
            name,
            val,
            max_val,
            default,
        )
        return default

    return val


@dataclass
class Settings:
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    primary_provider: str = "gemini"
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "medical_evidence"
    top_k: int = 4
    evaluator_confidence_threshold: float = 0.65
    request_timeout: float = 10.0

    def __post_init__(self) -> None:
        """Sanitize strings and guarantee field type invariants."""
        self.gemini_api_key = clean_env_value(self.gemini_api_key)
        self.openrouter_api_key = clean_env_value(self.openrouter_api_key)
        self.gemini_model = clean_env_value(self.gemini_model) or "gemini-1.5-flash"
        self.openrouter_model = (
            clean_env_value(self.openrouter_model) or "meta-llama/llama-3.3-70b-instruct"
        )
        cleaned_provider = clean_env_value(self.primary_provider).lower()
        if cleaned_provider in VALID_PRIMARY_PROVIDERS:
            self.primary_provider = cleaned_provider
        else:
            logger.warning(
                "Invalid primary_provider %r; defaulting to 'gemini'",
                self.primary_provider,
            )
            self.primary_provider = DEFAULT_PRIMARY_PROVIDER
        self.chroma_persist_dir = (
            clean_env_value(self.chroma_persist_dir) or "./data/chroma_db"
        )
        self.chroma_collection_name = (
            clean_env_value(self.chroma_collection_name) or "medical_evidence"
        )
        try:
            val_int = int(self.top_k)
            self.top_k = val_int if val_int >= 1 else 4
        except (ValueError, TypeError):
            self.top_k = 4
        try:
            val_thresh = float(self.evaluator_confidence_threshold)
            if math.isnan(val_thresh) or math.isinf(val_thresh) or not (0.0 <= val_thresh <= 1.0):
                self.evaluator_confidence_threshold = 0.65
            else:
                self.evaluator_confidence_threshold = val_thresh
        except (ValueError, TypeError):
            self.evaluator_confidence_threshold = 0.65
        try:
            val_to = float(self.request_timeout)
            if math.isnan(val_to) or math.isinf(val_to) or val_to <= 0.0:
                self.request_timeout = 10.0
            else:
                self.request_timeout = val_to
        except (ValueError, TypeError):
            self.request_timeout = 10.0

    @property
    def has_openrouter_key(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_any_llm_key(self) -> bool:
        return self.has_openrouter_key or self.has_gemini_key

    @property
    def is_openrouter_key_valid(self) -> bool:
        valid, _ = validate_openrouter_key(self.openrouter_api_key)
        return valid

    @property
    def is_gemini_key_valid(self) -> bool:
        valid, _ = validate_gemini_key(self.gemini_api_key)
        return valid

    @property
    def masked_openrouter_key(self) -> str:
        return mask_key(self.openrouter_api_key)

    @property
    def masked_gemini_key(self) -> str:
        return mask_key(self.gemini_api_key)

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        """Load settings from environment variables and .env file.

        Args:
            env_file: Optional path to an explicit .env file. If provided,
                it is loaded into a local dictionary without mutating os.environ.
                Otherwise, ambient .env discovery is executed once per process.

        Returns:
            Configured Settings instance.
        """
        file_vars = load_environment(env_file)

        return cls(
            gemini_api_key=_extract_env_var(
                ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"],
                default="",
                source_dict=file_vars,
            ),
            openrouter_api_key=_extract_env_var(
                [
                    "OPENROUTER_API_KEY",
                    "OPEN_ROUTER_API_KEY",
                    "openrouter_api_key",
                    "open_router_api_key",
                ],
                default="",
                source_dict=file_vars,
            ),
            gemini_model=_extract_env_var(
                ["GEMINI_MODEL", "gemini_model"],
                default="gemini-1.5-flash",
                source_dict=file_vars,
            ),
            openrouter_model=_extract_env_var(
                [
                    "OPENROUTER_MODEL",
                    "OPEN_ROUTER_MODEL",
                    "openrouter_model",
                    "open_router_model",
                ],
                default="meta-llama/llama-3.3-70b-instruct",
                source_dict=file_vars,
            ),
            primary_provider=_extract_env_var(
                [
                    "PRIMARY_PROVIDER",
                    "LLM_PRIMARY_PROVIDER",
                    "primary_provider",
                    "llm_primary_provider",
                ],
                default="gemini",
                source_dict=file_vars,
            ),
            chroma_persist_dir=_extract_env_var(
                ["CHROMA_PERSIST_DIR", "chroma_persist_dir"],
                default="./data/chroma_db",
                source_dict=file_vars,
            ),
            chroma_collection_name=_extract_env_var(
                ["CHROMA_COLLECTION_NAME", "chroma_collection_name"],
                default="medical_evidence",
                source_dict=file_vars,
            ),
            top_k=_extract_int_env("TOP_K", 4, source_dict=file_vars, min_val=1),
            evaluator_confidence_threshold=_extract_float_env(
                "EVALUATOR_CONFIDENCE_THRESHOLD",
                0.65,
                source_dict=file_vars,
                min_val=0.0,
                max_val=1.0,
            ),
            request_timeout=_extract_float_env(
                "REQUEST_TIMEOUT",
                10.0,
                source_dict=file_vars,
                must_be_positive=True,
            ),
        )
