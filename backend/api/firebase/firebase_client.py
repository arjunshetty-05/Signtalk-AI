"""
api/firebase/firebase_client.py — SignTalk AI / Person C, Prompt C2

Firestore + Storage data layer. Five collections:
    users          — profile data, created via Firebase Auth, keyed by uid.
    conversations  — one doc per corrected sentence: user_id, session_id,
                      raw gesture tokens, corrected_sentence, emotion,
                      language, created_at.
    emotion_logs   — per-session emotion readings: session_id, emotion,
                      confidence, timestamp.
    gesture_logs   — per-prediction raw gesture events: session_id, label,
                      confidence, timestamp, inference_time_ms.
    translations   — cache layer: keyed by hash(text+target_lang), storing
                      translated result, hit_count, last_used.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore, storage

from api.core.config import settings

logger = logging.getLogger("signtalk.firebase")

COLLECTION_USERS = "users"
COLLECTION_CONVERSATIONS = "conversations"
COLLECTION_EMOTION_LOGS = "emotion_logs"
COLLECTION_GESTURE_LOGS = "gesture_logs"
COLLECTION_TRANSLATIONS = "translations"


@lru_cache(maxsize=1)
def _get_app():
    if not firebase_admin._apps:
        if settings.FIREBASE_SERVICE_ACCOUNT_PATH:
            cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
        else:
            cred = credentials.ApplicationDefault()
        options = {}
        if settings.FIREBASE_STORAGE_BUCKET:
            options["storageBucket"] = settings.FIREBASE_STORAGE_BUCKET
        firebase_admin.initialize_app(cred, options)
    return firebase_admin.get_app()


def _db():
    _get_app()
    return firestore.client()


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------
def save_conversation(entry: dict) -> str:
    """entry: {user_id, session_id, gesture_tokens, corrected_sentence,
    emotion, language, created_at (optional)}. Returns the new doc ID."""
    entry = dict(entry)
    entry.setdefault("created_at", time.time())
    doc_ref = _db().collection(COLLECTION_CONVERSATIONS).document()
    doc_ref.set(entry)
    return doc_ref.id


def get_conversation_history(user_id: str, limit: int = 50) -> list[dict]:
    """Returns this user's conversations, reverse-chronological."""
    query = (
        _db()
        .collection(COLLECTION_CONVERSATIONS)
        .where("user_id", "==", user_id)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    docs = query.stream()
    results = []
    for doc in docs:
        data = doc.to_dict()
        results.append({
            "sentence": data.get("corrected_sentence", ""),
            "language": data.get("language", "en"),
            "emotion": data.get("emotion", "neutral"),
            "created_at": data.get("created_at", 0.0),
        })
    return results


# ---------------------------------------------------------------------------
# emotion_logs / gesture_logs
# ---------------------------------------------------------------------------
def log_emotion(session_id: str, emotion: str, confidence: float) -> None:
    _db().collection(COLLECTION_EMOTION_LOGS).add({
        "session_id": session_id,
        "emotion": emotion,
        "confidence": confidence,
        "timestamp": time.time(),
    })


def log_gesture(session_id: str, label: str, confidence: float, inference_time_ms: float) -> None:
    _db().collection(COLLECTION_GESTURE_LOGS).add({
        "session_id": session_id,
        "label": label,
        "confidence": confidence,
        "timestamp": time.time(),
        "inference_time_ms": inference_time_ms,
    })


# ---------------------------------------------------------------------------
# translations cache
# ---------------------------------------------------------------------------
def get_translation_cache(cache_key: str) -> str | None:
    doc = _db().collection(COLLECTION_TRANSLATIONS).document(cache_key).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    _db().collection(COLLECTION_TRANSLATIONS).document(cache_key).update({
        "hit_count": firestore.Increment(1),
        "last_used": time.time(),
    })
    return data.get("translated_text")


def set_translation_cache(cache_key: str, translated_text: str) -> None:
    _db().collection(COLLECTION_TRANSLATIONS).document(cache_key).set({
        "translated_text": translated_text,
        "hit_count": 0,
        "last_used": time.time(),
    }, merge=True)


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
def upsert_user_profile(uid: str, profile: dict) -> None:
    _db().collection(COLLECTION_USERS).document(uid).set(profile, merge=True)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def upload_file(local_path: str, remote_path: str) -> str:
    """Uploads a local file (recorded audio/video artifact) to Firebase
    Storage and returns a public URL."""
    _get_app()
    bucket = storage.bucket()
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url
