"""
core/config.py — SignTalk AI / Person C, Prompt C1

Centralized settings, loaded from environment variables (see .env.example).
Every module reads config from here rather than calling os.environ directly,
so all tunables live in one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


# Load the backend configuration when the package is imported, regardless of
# whether the server was launched from the repository root or backend/.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))


class Settings:
    # CORS — two named origins, no wildcard (placeholders, documented as such)
    CORS_ORIGINS: list[str] = [
        os.environ.get("CORS_ORIGIN_WEB", "http://localhost:3000"),
        os.environ.get("CORS_ORIGIN_PROD", "https://signtalk.vercel.app"),
    ]
    if ENV := os.environ.get("ENV", "development"):
        if ENV == "development" and "http://127.0.0.1:3000" not in CORS_ORIGINS:
            CORS_ORIGINS.append("http://127.0.0.1:3000")

    # Rate limiting (slowapi)
    RATE_LIMIT_PREDICTION = os.environ.get("RATE_LIMIT_PREDICTION", "30/minute")
    RATE_LIMIT_READONLY = os.environ.get("RATE_LIMIT_READONLY", "100/minute")

    # Firebase
    FIREBASE_SERVICE_ACCOUNT_PATH = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "")
    FIREBASE_STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "")

    # Third-party API keys
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")

    # Misc
    ENV = os.environ.get("ENV", "development")


settings = Settings()
