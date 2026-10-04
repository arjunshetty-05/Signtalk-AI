"""
core/config.py — SignTalk AI / Person C, Prompt C1

Centralized settings, loaded from environment variables (see .env.example).
Every module reads config from here rather than calling os.environ directly,
so all tunables live in one place.
"""

from __future__ import annotations

import os


class Settings:
    # CORS — two named origins, no wildcard (placeholders, documented as such)
    CORS_ORIGINS: list[str] = [
        os.environ.get("CORS_ORIGIN_WEB", "http://localhost:3000"),
        os.environ.get("CORS_ORIGIN_PROD", "https://signtalk.vercel.app"),
    ]

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
