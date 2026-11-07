"""Unit tests for FastAPI REST and SSE streaming endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from medical_crag.api import app
from medical_crag.types import MedicalCRAGResult, SourceCitation


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.get("/health")
        assert r1.status_code == 200
        assert r1.json()["status"] == "ok"
        assert r1.json()["version"] == "2.0.0"

        r2 = await client.get("/api/v1/health")
        assert r2.status_code == 200
        assert r2.json()["status"] == "ok"


@pytest.mark.anyio
async def test_verify_claim_endpoint_empty_claim():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/verify", json={"claim": "  "})
        assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_verify_claim_endpoint_success():
    mock_pipeline = MagicMock()
    mock_result = MedicalCRAGResult(
        claim="Vitamin D supports bone health",
        verdict="SUPPORTED",
        confidence=0.92,
        consumer_summary="Vitamin D facilitates calcium absorption.",
        clinical_explanation="Meta-analyses confirm reduced fracture risk [1].",
        sources=[
            SourceCitation(
                id="doc-1",
                title="Endocrine Society Clinical Practice Guideline",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                source="PubMed",
                snippet="Vitamin D guideline excerpt",
            )
        ],
        citation_precision=1.0,
        verified_citations=["doc-1"],
    )
    mock_pipeline.verify_async = AsyncMock(return_value=mock_result)

    with patch("medical_crag.api.get_pipeline", return_value=mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/verify", json={"claim": "Vitamin D supports bone health"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "SUPPORTED"
            assert data["confidence"] == 0.92
            assert "patient_summary" in data or "consumer_summary" in data
            assert len(data["sources"]) == 1
            assert data["citation_precision"] == 1.0


@pytest.mark.anyio
async def test_verify_claim_stream_endpoint():
    mock_pipeline = MagicMock()

    async def mock_stream(claim):
        yield {"stage": "retrieve", "message": "Searching..."}
        yield {"stage": "evaluate", "message": "Evaluating..."}
        yield {
            "stage": "complete",
            "result": {
                "claim": claim,
                "verdict": "SUPPORTED",
                "confidence": 0.9,
            },
        }

    mock_pipeline.stream_verify = mock_stream

    with patch("medical_crag.api.get_pipeline", return_value=mock_pipeline):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/verify/stream?claim=Turmeric%20reduces%20inflammation")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            assert "event: retrieve" in body
            assert "event: evaluate" in body
            assert "event: complete" in body


@pytest.mark.anyio
async def test_verify_image_endpoint_invalid_content_type():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/verify/image",
            files={"image": ("test.txt", b"plain text content", "text/plain")},
        )
        assert resp.status_code == 415


@pytest.mark.anyio
async def test_verify_image_endpoint_success():
    mock_pipeline = MagicMock()
    mock_pipeline.llm.extract_claim_from_image.return_value = (
        "Turmeric reduces arthritis inflammation",
        "English",
    )
    mock_result = MedicalCRAGResult(
        claim="Turmeric reduces arthritis inflammation",
        verdict="SUPPORTED",
        confidence=0.88,
        consumer_summary="Turmeric reduces pain.",
        clinical_explanation="Curcumin inhibits NF-kB pathways [1].",
    )
    mock_pipeline.verify_async = AsyncMock(return_value=mock_result)

    with patch("medical_crag.api.get_pipeline", return_value=mock_pipeline):
        dummy_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
            b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/verify/image",
                files={"image": ("claim.png", io.BytesIO(dummy_png), "image/png")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["extracted_claim"] == "Turmeric reduces arthritis inflammation"
            assert data["detected_language"] == "English"
            assert data["verification"]["verdict"] == "SUPPORTED"
