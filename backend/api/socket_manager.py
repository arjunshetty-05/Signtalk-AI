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

from api.main import app as fastapi_app

logger = logging.getLogger("signtalk.socketio")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)


@sio.event
async def connect(sid, environ, auth):
    """Clients should connect with `auth: {token, uid}` and immediately be
    joined to a room keyed by their Firebase uid."""
    uid = (auth or {}).get("uid")
    if uid:
        await sio.enter_room(sid, uid)
        logger.info("Socket %s joined room %s", sid, uid)
    else:
        logger.warning("Socket %s connected without a uid — not joined to any room", sid)


@sio.event
async def disconnect(sid):
    logger.info("Socket %s disconnected", sid)


async def broadcast_new_conversation(user_id: str, entry: dict) -> None:
    """Broadcasts a newly saved conversation entry to every device the
    given user has connected (their Socket.IO room)."""
    await sio.emit("conversation:new", entry, room=user_id)
