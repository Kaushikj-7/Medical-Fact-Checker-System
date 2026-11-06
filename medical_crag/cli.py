"""Command-line interface for testing and running Medical CRAG."""

from __future__ import annotations

import argparse
import sys
from medical_crag.pipeline import MedicalCRAGPipeline


def format_cli_output(result) -> str:
    sep = "=" * 65
    sub_sep = "-" * 65

    verdict_emoji = {
        "SUPPORTED": "✅",
        "REFUTED": "❌",
        "MISLEADING": "⚠️",
        "INSUFFICIENT_EVIDENCE": "❓",
    }.get(result.verdict, "ℹ️")

    lines = [
        "",
        sep,
        f"🏥 MEDICAL CORRECTIVE RAG VERIFICATION",
        sep,
        f"Claim: \"{result.claim}\"",
        f"Verdict: {verdict_emoji} {result.verdict} (Confidence: {result.confidence:.0%})",
        f"Execution Time: {result.execution_time_ms} ms | Provider: {result.llm_provider_used}",
        f"Corrective Action: {'Triggered (External Search & Ingested)' if result.corrective_action_taken else 'Not Needed (Local Store Sufficient)'}",
        sub_sep,
        "📋 PATIENT SUMMARY (Plain Language):",
        f"  {result.consumer_summary}",
        "",
        "🔬 CLINICAL PROOF CHAIN (Biomedical Evidence):",
        f"  {result.clinical_explanation}",
        "",
        "📚 AUTHORITATIVE SOURCES CITED:",
    ]

    if not result.sources:
        lines.append("  (None found)")
    else:
        for idx, src in enumerate(result.sources, start=1):
            lines.append(f"  [{idx}] {src.title}")
            lines.append(f"      Source: {src.source} | URL: {src.url}")
            lines.append(f"      Quote: \"{src.snippet[:140]}...\"")

    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Corrective RAG (CRAG) Claim Verifier")
    parser.add_argument("claim", type=str, nargs="?", help="Health claim to verify")
    args = parser.parse_args()

    claim = args.claim
    if not claim:
        print("Enter a health claim to verify (or Ctrl+C to exit):")
        try:
            claim = input("> ").strip()
        except KeyboardInterrupt:
            sys.exit(0)

    if not claim:
        print("Error: claim cannot be empty.")
        sys.exit(1)

    print(f"\n🔍 Processing claim through Corrective RAG: '{claim}'...")
    pipeline = MedicalCRAGPipeline()
    result = pipeline.verify(claim)
    print(format_cli_output(result))


if __name__ == "__main__":
    main()
