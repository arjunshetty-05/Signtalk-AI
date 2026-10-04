"""
api/socket_manager.py — SignTalk AI / Person C, Prompt C2

Socket.IO realtime sync, mounted alongside the existing FastAPI app so
REST/WebSocket and Socket.IO traffic share one ASGI process.

When a new conversation entry is saved (see api/ai/router.py), a
"conversation:new" event is broadcast to any connected clients in that
user's room — so a second device (e.g. the hearing person's phone) sees the
corrected sentence appear live without polling.

Deployment entrypoint changes from `api.main:app` to `api.socket_manager:socket_app`.
"""

from __future__ import annotations

import logging

import socketio

from api.core.config import settings
from api.main import app as fastapi_app

logger = logging.getLogger("signtalk.socketio")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=settings.CORS_ORIGINS)
socket_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


@sio.event
async def connect(sid, environ, auth):
    """Clients must connect with `auth: {token}` — a Firebase Auth ID token.
    The uid used to join a room comes from the verified token, never from a
    client-supplied value, so a socket can only ever join its own user's
    room."""
    from api.auth.dependencies import _ensure_firebase_initialized

    token = (auth or {}).get("token")
    if not token:
        logger.warning("Socket %s connected without a token — refusing", sid)
        raise socketio.exceptions.ConnectionRefusedError("Missing auth token")

    try:
        _ensure_firebase_initialized()
        from firebase_admin import auth as firebase_auth

        decoded = firebase_auth.verify_id_token(token)
    except Exception as exc:
        logger.warning("Socket %s auth failed: %s", sid, exc)
        raise socketio.exceptions.ConnectionRefusedError("Invalid or expired auth token") from exc

    uid = decoded.get("uid") or decoded.get("user_id")
    await sio.enter_room(sid, uid)
    logger.info("Socket %s joined room %s", sid, uid)


@sio.event
async def disconnect(sid):
    logger.info("Socket %s disconnected", sid)


async def broadcast_new_conversation(user_id: str, entry: dict) -> None:
    """Broadcasts a newly saved conversation entry to every device the
    given user has connected (their Socket.IO room)."""
    await sio.emit("conversation:new", entry, room=user_id)
