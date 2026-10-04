"""
core/offline_mode.py — SignTalk AI / Person C, Prompt C3

The ONE place offline mode is determined, for the "airplane-mode demo"
capability. Read from either an X-Offline-Mode request header or a
?offline=true query param. Every route that needs to know whether it's
running offline depends on get_offline_mode() — no duplicated
header-parsing logic anywhere else.
"""

from __future__ import annotations

from fastapi import Request, WebSocket


def get_offline_mode(request: Request | WebSocket) -> bool:
    """Works for both regular HTTP requests and WebSocket connections —
    Starlette's Request and WebSocket expose the same .headers/.query_params
    interface, so one implementation covers both."""
    header_value = request.headers.get("X-Offline-Mode", "").strip().lower()
    if header_value in ("true", "1", "yes"):
        return True

    query_value = request.query_params.get("offline", "").strip().lower()
    return query_value in ("true", "1", "yes")
