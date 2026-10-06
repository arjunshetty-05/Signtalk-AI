"""FastAPI application factory for the SignTalk v3 server.

Wires CORS (origins from the ``CORS_ORIGINS`` env var), a request-body size
limit, a per-request id + structured logging, shared app state (config +
SQLite storage), and the routers (health, vocab, recognize, enroll).

ML logic is NOT here — handlers delegate to :mod:`signtalk_core`.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:  # python-dotenv is optional; fall back to os.environ if absent.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001 - dotenv is a convenience, not a requirement
    pass

from signtalk_core.config import load_config

from server.app.storage import Storage
from server.app.recognizer import load_bundle, load_detector
from server.app.providers import build_provider
from server.app.routers import health, vocab, recognize, confirm, compose, enroll, speech

logger = logging.getLogger("signtalk.server")

# Max JSON/body size in bytes (base64 frame batches can be large). Overridable
# via MAX_REQUEST_BYTES; default 32 MiB.
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


def _cors_origins() -> list[str]:
    """Parse the comma-separated ``CORS_ORIGINS`` env var (Vite default)."""
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _max_request_bytes() -> int:
    """Request body size limit from env, defaulting to 32 MiB."""
    try:
        return int(os.environ.get("MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, enforce the body-size limit, and log each request."""

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "request body too large"},
                        headers={"x-request-id": request_id},
                    )
            except ValueError:
                pass

        logger.info(
            "request",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Shared, process-wide state: config + SQLite storage (guest mode).
        app.state.config = load_config()
        app.state.storage = Storage()
        # Optional serving artifacts: a MediaPipe detector and a trained
        # ensemble bundle. Both are None in skeleton mode (no env vars / no
        # artifacts), in which case recognize_clip runs its Phase-1 fallback.
        app.state.detector = load_detector()
        app.state.bundle = load_bundle()
        # Sentence-layer LLM provider (none/gemini/ollama); synonyms for the
        # LLM safety check. NullProvider by default -> scripted/template only.
        app.state.llm_provider = build_provider()
        app.state.synonyms = None
        yield
        app.state.storage.close()

    app = FastAPI(title="SignTalk AI v3", version="3.0.0", lifespan=lifespan)

    app.add_middleware(RequestContextMiddleware, max_bytes=_max_request_bytes())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(vocab.router)
    app.include_router(recognize.router)
    app.include_router(confirm.router)
    app.include_router(compose.router)
    app.include_router(enroll.router)
    app.include_router(speech.router)

    return app


app = create_app()
