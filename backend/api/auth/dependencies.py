"""
auth/dependencies.py — SignTalk AI / Person C, Prompt C1

get_current_user: validates a Firebase Auth JWT (Bearer token) using
firebase-admin's verify_id_token(). Applied to every route except GET
/health. Raises 401 (via AuthError -> {"error", "code"} JSON) on any
invalid/expired/missing token.
"""

from __future__ import annotations

import logging

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.core.exceptions import AuthError

logger = logging.getLogger("signtalk.auth")

_bearer_scheme = HTTPBearer(auto_error=False)
_firebase_initialized = False


def _ensure_firebase_initialized() -> None:
    global _firebase_initialized
    if _firebase_initialized:
        return
    import os

    import firebase_admin
    from firebase_admin import credentials

    if not firebase_admin._apps:
        cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
        if cred_path and os.path.exists(cred_path):
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        else:
            # Falls back to Application Default Credentials (e.g. in a GCP
            # environment). Verification will fail loudly at call time if
            # neither is configured — that's surfaced as a 401, not a crash.
            firebase_admin.initialize_app()
    _firebase_initialized = True


class CurrentUser:
    """Minimal identity object exposed to route handlers post-auth."""

    def __init__(self, uid: str, email: str | None, claims: dict):
        self.uid = uid
        self.email = email
        self.claims = claims


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CurrentUser:
    if credentials is None or not credentials.credentials:
        raise AuthError("Missing Authorization Bearer token")

    _ensure_firebase_initialized()

    from firebase_admin import auth as firebase_auth

    try:
        decoded = firebase_auth.verify_id_token(credentials.credentials)
    except Exception as exc:
        logger.warning("Firebase token verification failed: %s", exc)
        raise AuthError("Invalid or expired authentication token") from exc

    return CurrentUser(
        uid=decoded.get("uid") or decoded.get("user_id"),
        email=decoded.get("email"),
        claims=decoded,
    )
