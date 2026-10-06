"""Recognition endpoint (Section 6.1): POST /api/recognize.

Accepts EITHER:
  * multipart/form-data with a ``clip`` file (WebM/MP4) plus ``fps``,
    ``signer_id`` and optional ``scenario_id`` form fields; or
  * application/json with a :class:`RecognizeRequest` (base64 frame batch).

In both cases the clip is reduced to a list of base64 JPEG frames and passed to
:func:`signtalk_core.recognize.recognize_clip`, which returns the exact Section
6.1 dict. The decision is then logged to SQLite before the response is returned.

No MediaPipe ``.task`` bundle is wired up in the Phase-1 skeleton, so
``recognize_clip`` runs in no-detector mode: a synthetic/solid-colour clip
legitimately comes back as ``reject`` / ``hands_not_visible`` (a valid
contract-shape result, per context.json).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from signtalk_core.landmarks import read_video_frames
from signtalk_core.recognize import recognize_clip

from server.app.schemas import RecognizeRequest, RecognizeResponse

router = APIRouter(prefix="/api", tags=["recognize"])


def _frames_to_base64(frames_bgr: list) -> list[str]:
    """Encode a list of BGR frames to base64 JPEG strings."""
    out: list[str] = []
    for frame in frames_bgr:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=400, detail="could not encode clip frame")
        out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return out


def _decode_multipart_clip(data: bytes) -> list[str]:
    """Write the uploaded clip to a temp file, decode frames, return base64."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        frames = read_video_frames(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not frames:
        raise HTTPException(status_code=400, detail="clip contained no decodable frames")
    return _frames_to_base64(frames)


def _run_and_log(
    request: Request,
    *,
    frames: list[str],
    fps: float,
    signer_id: str,
    scenario_id: str | None,
) -> dict:
    """Call recognize_clip, persist the decision, return the 6.1 dict."""
    cfg = request.app.state.config
    detector = getattr(request.app.state, "detector", None)
    bundle = getattr(request.app.state, "bundle", None)
    result = recognize_clip(
        frames, fps, signer_id, scenario_id, cfg=cfg, detector=detector, bundle=bundle
    )

    thresholds = {
        "t_accept": cfg.decision.t_accept,
        "m_accept": cfg.decision.m_accept,
        "t_confirm": cfg.decision.t_confirm,
    }
    request.app.state.storage.log_decision(
        clip_id=result["clip_id"],
        signer_id=signer_id,
        decision=result["decision"],
        label=result["label"],
        confidence=result["confidence"],
        margin=result["margin"],
        probs=result["candidates"],
        thresholds=thresholds,
    )
    return result


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    request: Request,
    clip: UploadFile | None = File(default=None),
    fps: float | None = Form(default=None),
    signer_id: str | None = Form(default=None),
    scenario_id: str | None = Form(default=None),
) -> dict:
    """Recognise a clip from multipart file OR JSON frame batch (Section 6.1)."""
    content_type = request.headers.get("content-type", "")

    # --- multipart branch ------------------------------------------------- #
    if clip is not None:
        if fps is None or signer_id is None:
            raise HTTPException(
                status_code=422, detail="multipart requires fps and signer_id"
            )
        data = await clip.read()
        frames = _decode_multipart_clip(data)
        return _run_and_log(
            request, frames=frames, fps=fps, signer_id=signer_id, scenario_id=scenario_id
        )

    # --- JSON branch ------------------------------------------------------ #
    if "application/json" not in content_type:
        raise HTTPException(
            status_code=422,
            detail="send multipart with a 'clip' file or JSON with 'frames'",
        )
    body = await request.json()
    try:
        payload = RecognizeRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - surface validation as 422
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _run_and_log(
        request,
        frames=payload.frames,
        fps=payload.fps,
        signer_id=payload.signer_id,
        scenario_id=payload.scenario_id,
    )
