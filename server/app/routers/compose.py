"""Compose endpoint (Section 6.3): POST /api/compose.

Turns the accumulated signed words into a grammatical sentence in en/hi/kn via
:func:`signtalk_core.compose.compose_sentence` (scripted table -> optional LLM
with safety check -> template fallback). The LLM provider is built once at
startup and stored on app state; with ``LLM_PROVIDER=none`` (the default) this
endpoint is fully offline and deterministic.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from signtalk_core.compose import compose_sentence

from server.app.schemas import ComposeRequest, ComposeResponse, ComposeSentences

router = APIRouter(prefix="/api", tags=["compose"])


@router.post("/compose", response_model=ComposeResponse)
async def compose(request: Request, body: ComposeRequest) -> ComposeResponse:
    """Compose a sentence from signed words (Section 6.3)."""
    provider = getattr(request.app.state, "llm_provider", None)
    synonyms = getattr(request.app.state, "synonyms", None)

    t0 = time.perf_counter()
    result = compose_sentence(
        body.words,
        history=body.history,
        scenario_id=body.scenario_id,
        provider=provider,
        synonyms=synonyms,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return ComposeResponse(
        sentences=ComposeSentences(**result.sentences),
        source=result.source,
        verified=result.verified,
        latency_ms=latency_ms,
    )
