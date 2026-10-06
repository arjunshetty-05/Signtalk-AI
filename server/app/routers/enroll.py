"""Enrollment endpoints (Section 6.5, Phase-1 subset).

Guest-mode personal enrollment backed by SQLite + the local filesystem:

  * POST /api/enroll/start           -> allocate a new session_id for a signer.
  * POST /api/enroll/clip            -> save a base64 clip under
    ``data/enroll/<signer>/<session>/`` and index it in SQLite.
  * GET  /api/enroll/progress/{id}   -> per-sign clip counts for a signer.

Clips live under ``data/`` which is gitignored (Section 7), so nothing from
enrollment is ever committed. The per-signer/-session directory is created on
demand. (POST /api/enroll/train and /report are later-phase; not in FEAT-003.)
"""

from __future__ import annotations

import base64
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from server.app.schemas import (
    EnrollClipRequest,
    EnrollClipResponse,
    EnrollProgressResponse,
    EnrollStartRequest,
    EnrollStartResponse,
)

router = APIRouter(prefix="/api/enroll", tags=["enroll"])

# Allow only safe path components for signer/session/sign (no traversal).
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _enroll_root() -> Path:
    """Root for enrollment clips: ``<repo>/data/enroll`` (gitignored).

    This file is at ``<root>/server/app/routers/enroll.py`` -> root is 3 up.
    """
    return Path(__file__).resolve().parents[3] / "data" / "enroll"


def _safe_component(value: str, field: str) -> str:
    """Sanitise a user-supplied path component; reject if it becomes empty."""
    cleaned = _SAFE.sub("_", value.strip())
    if not cleaned:
        raise HTTPException(status_code=422, detail=f"invalid {field}")
    return cleaned


@router.post("/start", response_model=EnrollStartResponse)
def enroll_start(req: EnrollStartRequest) -> EnrollStartResponse:
    """Start a new enrollment session for a signer and return its id."""
    signer = _safe_component(req.signer_id, "signer_id")
    session_id = uuid.uuid4().hex
    return EnrollStartResponse(signer_id=signer, session_id=session_id)


@router.post("/clip", response_model=EnrollClipResponse)
def enroll_clip(request: Request, req: EnrollClipRequest) -> EnrollClipResponse:
    """Save one enrollment clip and index it in SQLite; return the new count."""
    signer = _safe_component(req.signer_id, "signer_id")
    session = _safe_component(req.session_id, "session_id")
    sign = _safe_component(req.sign, "sign")

    payload = req.clip
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid base64 clip: {exc}") from exc
    if not raw:
        raise HTTPException(status_code=422, detail="empty clip")

    out_dir = _enroll_root() / signer / session
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / f"{sign}_{uuid.uuid4().hex}.webm"
    clip_path.write_bytes(raw)

    storage = request.app.state.storage
    storage.index_enroll_clip(
        signer_id=signer,
        session_id=session,
        sign=sign,
        clip_path=str(clip_path),
    )
    counts = storage.enroll_progress(signer)
    return EnrollClipResponse(
        signer_id=signer,
        session_id=session,
        sign=sign,
        clip_path=str(clip_path),
        count=int(counts.get(sign, 0)),
    )


@router.get("/progress/{signer_id}", response_model=EnrollProgressResponse)
def enroll_progress(request: Request, signer_id: str) -> EnrollProgressResponse:
    """Return per-sign clip counts (and the total) for a signer."""
    signer = _safe_component(signer_id, "signer_id")
    counts = request.app.state.storage.enroll_progress(signer)
    return EnrollProgressResponse(
        signer_id=signer,
        counts=counts,
        total=int(sum(counts.values())),
    )
