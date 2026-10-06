"""Confirm endpoint (Section 6.2): POST /api/confirm.

When the decision engine returns ``decision="confirm"``, the web client shows
the top-3 candidate chips; the user taps the right one and the client POSTs the
chosen label here. The pick is stored (as a labelled sample for later human
review — Section 8.6, never auto-trained) and acknowledged.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.app.schemas import ConfirmRequest, ConfirmResponse

router = APIRouter(prefix="/api", tags=["confirm"])


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm(request: Request, body: ConfirmRequest) -> ConfirmResponse:
    """Record the user's chosen candidate for a clip (Section 6.2)."""
    request.app.state.storage.log_confirmation(
        clip_id=body.clip_id, chosen_label=body.chosen_label
    )
    return ConfirmResponse(
        clip_id=body.clip_id, chosen_label=body.chosen_label, stored=True
    )
