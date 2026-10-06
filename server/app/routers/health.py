"""Health endpoint (Section 6.7): GET /health -> {"status": "ok"}."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 with a status-ok body."""
    return {"status": "ok"}
