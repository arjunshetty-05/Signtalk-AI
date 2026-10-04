"""
core/offline_mode.py — SignTalk AI / Person C, Prompt C3

The ONE place offline mode is determined, for the "airplane-mode demo"
capability. Read from either an X-Offline-Mode request header or a
?offline=true query param. Every route that needs to know whether it's
running offline depends on get_offline_mode() — no duplicated
header-parsing logic anywhere else.
"""

from __future__ import annotations

from fastapi import Request


def get_offline_mode(request: Request) -> bool:
    """Type-hinted as Request for FastAPI's Depends() usage on REST routes
    (a Request | WebSocket union here breaks FastAPI's dependency-injection
    introspection with a FastAPIError at import time). The websocket router
    calls this directly as a plain function, not via Depends — Starlette's
    WebSocket exposes the same .headers/.query_params interface as Request,
    so it works fine there at runtime despite the narrower type hint."""
    header_value = request.headers.get("X-Offline-Mode", "").strip().lower()
    if header_value in ("true", "1", "yes"):
        return True

    query_value = request.query_params.get("offline", "").strip().lower()
    return query_value in ("true", "1", "yes")
