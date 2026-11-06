"""Configuration settings for Medical CRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "medical_evidence"
    top_k: int = 4
    evaluator_confidence_threshold: float = 0.65
    request_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "medical_evidence"),
            top_k=int(os.getenv("TOP_K", "4")),
            evaluator_confidence_threshold=float(os.getenv("EVALUATOR_CONFIDENCE_THRESHOLD", "0.65")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "10.0")),
        )
