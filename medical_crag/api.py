"""Production-grade FastAPI service with Server-Sent Events (SSE) streaming and multimodal image verification."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from medical_crag.config import Settings
from medical_crag.pipeline import MedicalCRAGPipeline

logger = logging.getLogger(__name__)

# Initialize FastAPI application with OpenAPI metadata
app = FastAPI(
    title="Autonomous Medical Corrective-RAG (CRAG) API",
    description="High-Precision Biomedical Claim Verification Platform with Multi-MCP Grounding",
    version="2.0.0",
)

# Enable permissive CORS for frontend extensions and client dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance initialized on startup or lazily
_pipeline: Optional[MedicalCRAGPipeline] = None


def get_pipeline() -> MedicalCRAGPipeline:
    global _pipeline
    if _pipeline is None:
        settings = Settings.from_env()
        _pipeline = MedicalCRAGPipeline(settings=settings)
    return _pipeline


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="Health or biomedical claim to verify")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Service health and readiness check."""
    return HealthResponse(
        status="ok",
        service="medical-crag",
        version="2.0.0",
    )


@app.post("/api/v1/verify", tags=["Verification"])
async def verify_claim(payload: VerifyRequest) -> Dict[str, Any]:
    """Verify a health claim and return the complete dual-layer MedicalCRAGResult."""
    claim = payload.claim.strip()
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Health claim must not be empty.",
        )

    pipeline = get_pipeline()
    try:
        result = await pipeline.verify_async(claim)
        return result.to_dict()
    except Exception as exc:
        logger.error("Error during claim verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {exc}",
        ) from exc


@app.get("/api/v1/verify/stream", tags=["Verification"])
async def verify_claim_stream(
    claim: str = Query(..., min_length=3, description="Health claim to verify with live stage updates"),
) -> StreamingResponse:
    """Stream real-time CRAG pipeline stage transitions via Server-Sent Events (SSE)."""
    claim_clean = claim.strip()
    if not claim_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Health claim must not be empty.",
        )

    pipeline = get_pipeline()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for step in pipeline.stream_verify(claim_clean):
                stage_name = step.get("stage", "update")
                payload_json = json.dumps(step)
                yield f"event: {stage_name}\ndata: {payload_json}\n\n"
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            err_payload = json.dumps({"stage": "error", "message": str(exc)})
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/verify/image", tags=["Verification"])
async def verify_image_claim(
    image: UploadFile = File(..., description="Screenshot or image containing medical claim"),
) -> Dict[str, Any]:
    """Extract health claim from uploaded image via multimodal vision LLM and verify through CRAG engine."""
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided.",
        )

    content_type = image.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Uploaded file must be an image (received: {content_type}).",
        )

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty.",
            )

        pipeline = get_pipeline()

        # Extract text / claim from image using LLMGateway vision capability
        extracted_claim, detected_lang = pipeline.llm.extract_claim_from_image(
            image_bytes=image_bytes,
            mime_type=content_type,
        )

        if not extracted_claim or extracted_claim.startswith("Unspecified"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect a legible medical claim in the provided image.",
            )

        # Run extracted claim through CRAG pipeline
        result = await pipeline.verify_async(extracted_claim)

        return {
            "extracted_claim": extracted_claim,
            "detected_language": detected_lang,
            "verification": result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Image verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {exc}",
        ) from exc
