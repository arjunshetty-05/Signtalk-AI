"""Pydantic request/response models for the SignTalk v3 API.

These mirror the Section 6 interface contracts. The ``/api/recognize`` endpoint
accepts EITHER multipart (a clip file) OR JSON (a base64 frame batch); this
module defines the JSON request/response shapes and the ``/api/vocab`` item.
Response models match Section 6.1 exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClientQuality(BaseModel):
    """Optional client-side quality hints (Section 6.1 request)."""

    hands_visible_pct: float | None = None
    brightness: float | None = None


class RecognizeRequest(BaseModel):
    """JSON body for POST /api/recognize (base64 frame batch, Section 6.1)."""

    frames: list[str] = Field(..., description="base64 JPEG frames, one per entry")
    fps: float = Field(..., gt=0, description="capture frame rate")
    signer_id: str = Field(..., description="signer id (guest id in Phase 1)")
    scenario_id: str | None = None
    client_quality: ClientQuality | None = None


class Candidate(BaseModel):
    """A single top-k candidate (Section 6.1 ``candidates[]``)."""

    label: str
    p: float


class Agreement(BaseModel):
    """Model / TTA agreement summary (Section 6.1)."""

    models: str
    tta: str


class LatencyMs(BaseModel):
    """Per-stage latency in milliseconds (Section 6.1 ``latency_ms``)."""

    extract: int
    models: int
    fusion: int
    total: int


class RecognizeResponse(BaseModel):
    """POST /api/recognize response — the exact Section 6.1 shape."""

    clip_id: str
    decision: str  # "accept" | "confirm" | "reject"
    label: str | None
    candidates: list[Candidate]
    confidence: float
    margin: float
    agreement: Agreement
    reject_reason: str | None
    latency_ms: LatencyMs


class VocabItem(BaseModel):
    """GET /api/vocab item (Section 6.6)."""

    label: str
    category: str
    demo: bool


class EnrollStartRequest(BaseModel):
    """POST /api/enroll/start body (Section 6.5)."""

    signer_id: str


class EnrollStartResponse(BaseModel):
    """POST /api/enroll/start response: the new enrollment session id."""

    signer_id: str
    session_id: str


class EnrollClipRequest(BaseModel):
    """POST /api/enroll/clip JSON body (base64 clip, Section 6.5)."""

    signer_id: str
    session_id: str
    sign: str
    clip: str = Field(..., description="base64-encoded clip bytes (WebM/MP4)")


class EnrollClipResponse(BaseModel):
    """POST /api/enroll/clip response: where the clip was saved + new count."""

    signer_id: str
    session_id: str
    sign: str
    clip_path: str
    count: int


class EnrollProgressResponse(BaseModel):
    """GET /api/enroll/progress/{signer_id} response: per-sign clip counts."""

    signer_id: str
    counts: dict[str, int]
    total: int
