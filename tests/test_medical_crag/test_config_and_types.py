from medical_crag.config import Settings, mask_key
from medical_crag.types import (
    Document,
    EvaluationResult,
    MedicalCRAGResult,
    SourceCitation,
)


def test_document_and_citation_dataclasses():
    doc = Document(
        id="pmid-123",
        title="Sample Medical Study",
        text="Aspirin reduces cardiovascular events.",
        source="PubMed",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
    )
    assert doc.id == "pmid-123"
    assert doc.source == "PubMed"

    citation = SourceCitation(
        id="pmid-123",
        title="Sample Medical Study",
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
        source="PubMed",
        snippet="Aspirin reduces cardiovascular events.",
    )
    assert citation.id == "pmid-123"


def test_evaluation_result_schema():
    eval_res = EvaluationResult(
        is_sufficient=False,
        confidence=0.4,
        rationale="Local evidence is incomplete.",
        search_source="pubmed",
        search_keywords="aspirin cardiovascular risk",
    )
    assert eval_res.is_sufficient is False
    assert eval_res.search_source == "pubmed"


def test_medical_crag_result_schema():
    res = MedicalCRAGResult(
        claim="Aspirin prevents heart attacks",
        verdict="SUPPORTED",
        confidence=0.92,
        consumer_summary="Aspirin has been shown to reduce risk for certain patients.",
        clinical_explanation="Multiple randomized controlled trials support antiplatelet benefits [1].",
        sources=[
            SourceCitation(
                id="pmid-123",
                title="Study Title",
                url="https://pubmed.ncbi.nlm.nih.gov/123/",
                source="PubMed",
                snippet="Significant reduction in events.",
            )
        ],
        corrective_action_taken=True,
        execution_time_ms=450,
        llm_provider_used="gemini",
    )
    assert res.verdict == "SUPPORTED"
    assert len(res.sources) == 1
    assert res.corrective_action_taken is True


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./custom_chroma")
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "0.75")

    settings = Settings.from_env()
    assert settings.gemini_api_key == "test_gemini_key"
    assert settings.openrouter_api_key == "test_openrouter_key"
    assert settings.chroma_persist_dir == "./custom_chroma"
    assert settings.evaluator_confidence_threshold == 0.75


def test_settings_openrouter_alias_extraction(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "  sk-or-v1-alias-key  ")
    settings = Settings.from_env()
    assert settings.openrouter_api_key == "sk-or-v1-alias-key"


def test_settings_gemini_google_alias_extraction(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "  google-api-key-test  ")
    settings = Settings.from_env()
    assert settings.gemini_api_key == "google-api-key-test"


def test_settings_key_priority(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "alias-openrouter")
    monkeypatch.setenv("GEMINI_API_KEY", "primary-gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "alias-google")

    settings = Settings.from_env()
    assert settings.openrouter_api_key == "primary-openrouter"
    assert settings.gemini_api_key == "primary-gemini"


def test_settings_whitespace_only_falls_back_to_alias(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   \t  ")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-valid-fallback")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-valid-fallback")

    settings = Settings.from_env()
    assert settings.openrouter_api_key == "sk-valid-fallback"
    assert settings.gemini_api_key == "google-valid-fallback"


def test_settings_whitespace_and_quote_stripping(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", '  "  sk-or-v1-quoted-token  "  ')
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", " ' AIzaSySingleQuotedToken ' ")

    settings = Settings.from_env()
    assert settings.openrouter_api_key == "sk-or-v1-quoted-token"
    assert settings.gemini_api_key == "AIzaSySingleQuotedToken"


def test_settings_direct_instantiation_post_init_sanitization():
    s = Settings(
        openrouter_api_key='  "sk-or-v1-direct"  ',
        gemini_api_key="  'AIzaSyDirect'  ",
        top_k="6",
        evaluator_confidence_threshold="0.85",
        request_timeout="25.0",
    )
    assert s.openrouter_api_key == "sk-or-v1-direct"
    assert s.gemini_api_key == "AIzaSyDirect"
    assert s.top_k == 6
    assert isinstance(s.top_k, int)
    assert s.evaluator_confidence_threshold == 0.85
    assert isinstance(s.evaluator_confidence_threshold, float)
    assert s.request_timeout == 25.0
    assert isinstance(s.request_timeout, float)


def test_settings_from_env_with_custom_file_path(tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "OPEN_ROUTER_API_KEY=sk-custom-file-key\n"
        "GEMINI_API_KEY=gem-custom-file-key\n"
        "TOP_K=8\n"
        "EVALUATOR_CONFIDENCE_THRESHOLD=0.90\n"
        "REQUEST_TIMEOUT=15.0\n"
    )
    # Test Path object
    settings_path = Settings.from_env(env_file=env_file)
    assert settings_path.openrouter_api_key == "sk-custom-file-key"
    assert settings_path.gemini_api_key == "gem-custom-file-key"
    assert settings_path.top_k == 8
    assert settings_path.evaluator_confidence_threshold == 0.90
    assert settings_path.request_timeout == 15.0

    # Test str path
    settings_str = Settings.from_env(env_file=str(env_file))
    assert settings_str.openrouter_api_key == "sk-custom-file-key"
    assert settings_str.gemini_api_key == "gem-custom-file-key"


def test_settings_from_env_nonexistent_custom_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    non_existent = tmp_path / "missing.env"
    settings = Settings.from_env(env_file=non_existent)
    # Should not throw exception and fall back to ambient environment
    assert settings.openrouter_api_key == "env-key"


def test_settings_numeric_env_resilience(monkeypatch):
    monkeypatch.setenv("TOP_K", "not_an_int")
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "")
    monkeypatch.setenv("REQUEST_TIMEOUT", "invalid_float")

    settings = Settings.from_env()
    assert settings.top_k == 4
    assert settings.evaluator_confidence_threshold == 0.65
    assert settings.request_timeout == 10.0


def test_settings_helper_properties():
    # Empty keys
    empty_settings = Settings()
    assert empty_settings.has_openrouter_key is False
    assert empty_settings.has_gemini_key is False
    assert empty_settings.has_any_llm_key is False
    assert empty_settings.is_openrouter_key_valid is False
    assert empty_settings.is_gemini_key_valid is False
    assert empty_settings.masked_openrouter_key == "<unset>"
    assert empty_settings.masked_gemini_key == "<unset>"

    # Mock short keys
    mock_settings = Settings(
        openrouter_api_key="mock_key",
        gemini_api_key="mock_key",
    )
    assert mock_settings.has_openrouter_key is True
    assert mock_settings.has_gemini_key is True
    assert mock_settings.has_any_llm_key is True
    assert mock_settings.is_openrouter_key_valid is False
    assert mock_settings.is_gemini_key_valid is False
    assert mock_settings.masked_openrouter_key == "***"
    assert mock_settings.masked_gemini_key == "***"

    # Valid schema keys
    valid_or_key = "sk-or-v1-" + "a" * 64
    valid_gem_key = "AIzaSy" + "b" * 33
    valid_settings = Settings(
        openrouter_api_key=valid_or_key,
        gemini_api_key=valid_gem_key,
    )
    assert valid_settings.has_openrouter_key is True
    assert valid_settings.has_gemini_key is True
    assert valid_settings.has_any_llm_key is True
    assert valid_settings.is_openrouter_key_valid is True
    assert valid_settings.is_gemini_key_valid is True
    assert (
        valid_settings.masked_openrouter_key
        == f"{valid_or_key[:8]}...{valid_or_key[-4:]}"
    )
    assert (
        valid_settings.masked_gemini_key
        == f"{valid_gem_key[:8]}...{valid_gem_key[-4:]}"
    )


def test_settings_from_env_missing_keys_preserves_monkeypatch_delenv(monkeypatch):
    """Verify that unsetting all API keys causes Settings.from_env() to return empty keys without re-reading disk .env."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_api_key", raising=False)
    monkeypatch.delenv("open_router_api_key", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("gemini_api_key", raising=False)
    monkeypatch.delenv("google_api_key", raising=False)

    settings = Settings.from_env()
    assert settings.openrouter_api_key == ""
    assert settings.gemini_api_key == ""
    assert settings.has_openrouter_key is False
    assert settings.has_gemini_key is False
    assert settings.has_any_llm_key is False
    assert settings.masked_openrouter_key == "<unset>"
    assert settings.masked_gemini_key == "<unset>"


def test_settings_from_env_custom_file_does_not_mutate_process_environ(tmp_path):
    """Verify that loading settings from a custom env file does not leak variables into os.environ."""
    import os

    custom_env = tmp_path / "isolated.env"
    custom_env.write_text(
        "ISOLATION_TEST_SECRET_MARKER=super_secret_payload_12345\n"
        "OPENROUTER_API_KEY=sk-isolated-custom-env-file-key\n",
        encoding="utf-8",
    )
    os.environ.pop("ISOLATION_TEST_SECRET_MARKER", None)
    initial_openrouter_env = os.environ.get("OPENROUTER_API_KEY")

    settings = Settings.from_env(env_file=custom_env)

    # Settings parsed values correctly from custom file
    assert settings.openrouter_api_key == "sk-isolated-custom-env-file-key"
    # But process os.environ was NOT polluted or mutated
    assert "ISOLATION_TEST_SECRET_MARKER" not in os.environ
    assert os.environ.get("OPENROUTER_API_KEY") == initial_openrouter_env


def test_settings_numeric_bounds_validation(monkeypatch):
    """Verify bounds checking and fallback on negative, zero, NaN, Inf, and out-of-range numeric values."""
    # 1. Test negative values via environment variables fall back to defaults
    monkeypatch.setenv("TOP_K", "-5")
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "-0.25")
    monkeypatch.setenv("REQUEST_TIMEOUT", "-10.0")

    settings = Settings.from_env()
    assert settings.top_k == 4
    assert settings.evaluator_confidence_threshold == 0.65
    assert settings.request_timeout == 10.0

    # 2. Test zero top_k / timeout and out-of-range threshold (> 1.0) fall back to defaults
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "1.5")
    monkeypatch.setenv("REQUEST_TIMEOUT", "0.0")

    settings_zero = Settings.from_env()
    assert settings_zero.top_k == 4
    assert settings_zero.evaluator_confidence_threshold == 0.65
    assert settings_zero.request_timeout == 10.0

    # 3. Test NaN and Infinity string inputs fall back to defaults
    monkeypatch.setenv("EVALUATOR_CONFIDENCE_THRESHOLD", "nan")
    monkeypatch.setenv("REQUEST_TIMEOUT", "inf")

    settings_nan = Settings.from_env()
    assert settings_nan.evaluator_confidence_threshold == 0.65
    assert settings_nan.request_timeout == 10.0

    # 4. Test direct constructor instantiation bounds checking
    direct = Settings(
        top_k=-1,
        evaluator_confidence_threshold=float("nan"),
        request_timeout=0.0,
    )
    assert direct.top_k == 4
    assert direct.evaluator_confidence_threshold == 0.65
    assert direct.request_timeout == 10.0


def test_mask_key_security_thresholds():
    """Verify security masking thresholds for empty, short, medium, and production API keys."""
    # Empty / unset
    assert mask_key("") == "<unset>"
    assert mask_key(None) == "<unset>"
    assert mask_key("   ") == "<unset>"

    # Short / mock keys (<= 16 chars): mask completely to prevent leaking majority of token
    assert mask_key("short") == "***"
    assert mask_key("mock_key") == "***"
    assert mask_key("1234567890abc") == "***"  # 13 chars - previously leaked 12 chars!
    assert mask_key("1234567890123456") == "***"  # 16 chars

    # Medium keys (17-24 chars): mask inner characters, revealing prefix(2)...suffix(2)
    med_key = "12345678901234567"  # 17 chars
    assert mask_key(med_key) == "12...67"
    assert mask_key("sk-custom-path-key") == "sk...ey"  # 18 chars
    assert mask_key("123456789012345678901234") == "12...34"  # 24 chars

    # Intermediate keys (25-38 chars): reveal prefix(4)...suffix(4)
    inter_key = "1234567890123456789012345"  # 25 chars
    assert mask_key(inter_key) == "1234...2345"
    assert mask_key("12345678901234567890123456789012345678") == "1234...5678"  # 38 chars

    # Full production keys (>= 39 chars): reveal prefix(8)...suffix(4)
    long_gem_key = "AIzaSy" + "b" * 33  # 39 chars
    assert mask_key(long_gem_key) == f"{long_gem_key[:8]}...{long_gem_key[-4:]}"

    long_or_key = "sk-or-v1-" + "a" * 64  # 73 chars
    assert mask_key(long_or_key) == f"{long_or_key[:8]}...{long_or_key[-4:]}"


# ==============================================================================
# primary_provider Configuration & Validation Tests
# ==============================================================================


def test_settings_primary_provider_default():
    """Verify primary_provider defaults to 'gemini' in both Settings() and Settings.from_env()."""
    # Direct instantiation
    s = Settings()
    assert s.primary_provider == "gemini"

    # Environment extraction with no provider vars set
    s_env = Settings.from_env()
    assert s_env.primary_provider == "gemini"


def test_settings_primary_provider_direct_instantiation():
    """Verify Settings(primary_provider=...) accepts valid providers and normalizes casing/quotes."""
    # Standard openrouter
    s1 = Settings(primary_provider="openrouter")
    assert s1.primary_provider == "openrouter"

    # Standard gemini
    s2 = Settings(primary_provider="gemini")
    assert s2.primary_provider == "gemini"

    # Case normalization (uppercase)
    s3 = Settings(primary_provider="OPENROUTER")
    assert s3.primary_provider == "openrouter"

    # Mixed case and quoted string
    s4 = Settings(primary_provider='  "OpenRouter"  ')
    assert s4.primary_provider == "openrouter"

    # Single-quoted string with whitespace
    s5 = Settings(primary_provider=" 'Gemini' ")
    assert s5.primary_provider == "gemini"


def test_settings_primary_provider_validation_fallback(caplog):
    """Verify invalid or empty provider inputs safely fall back to 'gemini' with warning log."""
    import logging

    # Unsupported provider string
    with caplog.at_level(logging.WARNING):
        s_invalid = Settings(primary_provider="claude")
        assert s_invalid.primary_provider == "gemini"
        assert "Invalid primary_provider 'claude'" in caplog.text

    # Empty string
    s_empty = Settings(primary_provider="")
    assert s_empty.primary_provider == "gemini"

    # Whitespace only
    s_spaces = Settings(primary_provider="   \t  ")
    assert s_spaces.primary_provider == "gemini"

    # None value
    s_none = Settings(primary_provider=None)
    assert s_none.primary_provider == "gemini"


def test_settings_primary_provider_from_env_extraction(monkeypatch):
    """Verify PRIMARY_PROVIDER environment variable extraction, trimming, and lowercasing."""
    monkeypatch.setenv("PRIMARY_PROVIDER", "openrouter")
    settings = Settings.from_env()
    assert settings.primary_provider == "openrouter"

    # With whitespace and quotes
    monkeypatch.setenv("PRIMARY_PROVIDER", '  "OPENROUTER"  ')
    settings_quoted = Settings.from_env()
    assert settings_quoted.primary_provider == "openrouter"

    # Explicit gemini
    monkeypatch.setenv("PRIMARY_PROVIDER", "gemini")
    settings_gem = Settings.from_env()
    assert settings_gem.primary_provider == "gemini"


def test_settings_primary_provider_alias_and_precedence(monkeypatch):
    """Verify LLM_PRIMARY_PROVIDER alias and precedence of PRIMARY_PROVIDER over LLM_PRIMARY_PROVIDER."""
    # LLM_PRIMARY_PROVIDER alias used when PRIMARY_PROVIDER is absent
    monkeypatch.delenv("PRIMARY_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openrouter")
    settings_alias = Settings.from_env()
    assert settings_alias.primary_provider == "openrouter"

    # PRIMARY_PROVIDER takes precedence when both are provided
    monkeypatch.setenv("PRIMARY_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openrouter")
    settings_prec = Settings.from_env()
    assert settings_prec.primary_provider == "gemini"

    # Whitespace-only PRIMARY_PROVIDER falls back to LLM_PRIMARY_PROVIDER
    monkeypatch.setenv("PRIMARY_PROVIDER", "   \t  ")
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openrouter")
    settings_ws_fallback = Settings.from_env()
    assert settings_ws_fallback.primary_provider == "openrouter"


def test_settings_primary_provider_custom_file_isolation(tmp_path):
    """Verify PRIMARY_PROVIDER in custom .env file parses without polluting os.environ."""
    import os

    custom_env = tmp_path / "provider.env"
    custom_env.write_text("PRIMARY_PROVIDER=openrouter\n", encoding="utf-8")

    initial_env = os.environ.get("PRIMARY_PROVIDER")
    settings = Settings.from_env(env_file=custom_env)

    assert settings.primary_provider == "openrouter"
    assert os.environ.get("PRIMARY_PROVIDER") == initial_env

