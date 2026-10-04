"""
api/main.py — SignTalk AI / Person C, Prompt C1

The production-shaped, restructured FastAPI entrypoint. Wires together every
module's router behind Firebase-JWT auth, CORS restricted to two named
origins, slowapi rate limiting, and the shared error-handling pattern.

Run from the backend/ directory so sibling top-level modules (keypoint_utils,
classify, emotion, nlp_correction, speech, tts, translation) and this `api`
package both resolve as imports:

    cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000

Note: Person C's Prompt C2 (Socket.IO) wraps this `app` in an ASGIApp — see
socket_manager.py, which mounts this same app rather than replacing it.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.ai.router import router as ai_router
from api.analytics.router import router as analytics_router
from api.core.config import settings
from api.core.exceptions import register_exception_handlers
from api.core.limiter import limiter
from api.emotion.router import router as emotion_router
from api.pose.router import router as pose_router
from api.speech.router import router as speech_router
from api.translation.router import router as translation_router
from api.websocket.router import router as websocket_router
from keypoint_utils import load_movenet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signtalk.api.main")

app = FastAPI(
    title="SignTalk AI Backend",
    description="Real-time sign language recognition, emotion fusion, NLP correction, "
                 "speech, and translation — Person C's restructured, secured backend.",
    version="1.0.0",
)

# -- CORS: exactly two named origins, no wildcard ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Rate limiting -------------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -- Shared error handling ------------------------------------------------------
register_exception_handlers(app)

# -- Routers ---------------------------------------------------------------------
app.include_router(ai_router)
app.include_router(analytics_router)
app.include_router(emotion_router)
app.include_router(pose_router)
app.include_router(speech_router)
app.include_router(translation_router)
app.include_router(websocket_router)


@app.on_event("startup")
async def startup_event():
    """Load MoveNet once at process startup — never per-request/per-connection."""
    load_movenet()
    logger.info("SignTalk AI backend ready.")


@app.get("/health", tags=["health"])
async def health():
    """Unauthenticated liveness check — the only route without the Firebase
    JWT dependency."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
