"""Vocabulary endpoint (Section 6.6): GET /api/vocab -> [{label,category,demo}].

Reads ``config/vocabulary.json`` (object with a ``signs`` array per FEAT-001)
and returns the list of items. The file is small and static, so it is read on
each call (no caching needed for the skeleton).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from server.app.schemas import VocabItem

router = APIRouter(prefix="/api", tags=["vocab"])


def _vocab_path() -> Path:
    """Path to ``config/vocabulary.json`` under the repo root.

    This file is at ``<root>/server/app/routers/vocab.py`` -> root is 3 up.
    """
    return Path(__file__).resolve().parents[3] / "config" / "vocabulary.json"


@router.get("/vocab", response_model=list[VocabItem])
def get_vocab() -> list[VocabItem]:
    """Return the demo vocabulary as a list of {label, category, demo} items."""
    path = _vocab_path()
    if not path.is_file():
        raise HTTPException(status_code=500, detail="vocabulary.json not found")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    signs = data["signs"] if isinstance(data, dict) else data
    return [
        VocabItem(label=s["label"], category=s["category"], demo=bool(s["demo"]))
        for s in signs
    ]
