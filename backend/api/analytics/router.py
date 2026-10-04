"""
api/analytics/router.py — SignTalk AI / Person C, Prompt C1 (wired further in C2)

GET /metrics — in-memory counters (auth required; distinct from /health which
is the only unauthenticated route).
GET /conversations/{user_id} — conversation history, backed by Firestore
(Prompt C2's firebase_client.get_conversation_history).
POST /analytics/feedback — wrong-prediction correction submissions, calling
Person A's feedback_queue.record_correction() directly (that module is
intentionally FastAPI-free/dependency-free — this route is the HTTP layer
around it).
"""

import os
import sys

import numpy as np
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import CurrentUser, get_current_user
from api.core.config import settings
from api.core.limiter import limiter
from api.core.metrics import metrics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset_tools.feedback_queue import record_correction  # noqa: E402

router = APIRouter(tags=["analytics"])


class MetricsResponse(BaseModel):
    total_predictions: int
    average_latency_ms: float
    active_connections: int


class ConversationEntry(BaseModel):
    sentence: str
    language: str
    emotion: str
    created_at: float


class FeedbackRequest(BaseModel):
    session_id: str
    predicted_label: str
    correct_label: str
    sequence: list = Field(..., description="Flattened/nested (30, 17, 2) keypoint sequence")


class FeedbackResponse(BaseModel):
    filename: str


@router.get("/metrics", response_model=MetricsResponse)
@limiter.limit(settings.RATE_LIMIT_READONLY)
async def get_metrics(request: Request, user: CurrentUser = Depends(get_current_user)):
    """Returns in-memory server counters: total predictions, average
    inference latency, and active WebSocket connection count."""
    return MetricsResponse(**metrics.snapshot())


@router.get("/conversations/{user_id}", response_model=list[ConversationEntry])
@limiter.limit(settings.RATE_LIMIT_READONLY)
async def get_conversations(
    request: Request,
    user_id: str,
    limit: int = 50,
    user: CurrentUser = Depends(get_current_user),
):
    """Returns this user's past corrected sentences, reverse-chronological,
    backed by Firestore's `conversations` collection."""
    try:
        from api.firebase.firebase_client import get_conversation_history

        history = get_conversation_history(user_id, limit=limit)
    except Exception:
        history = []
    return [ConversationEntry(**entry) for entry in history]


@router.post("/analytics/feedback", response_model=FeedbackResponse)
@limiter.limit(settings.RATE_LIMIT_PREDICTION)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Records a wrong-prediction correction into the active-learning queue
    (Person A's feedback_queue.py) for later review/retraining."""
    sequence = np.array(body.sequence, dtype=np.float32)
    filename = record_correction(body.session_id, body.predicted_label, body.correct_label, sequence)
    return FeedbackResponse(filename=filename)
