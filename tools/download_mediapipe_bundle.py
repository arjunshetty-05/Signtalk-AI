"""Download MediaPipe Tasks model bundles (PROJECT_CONTEXT 4.4 / 5.3).

Landmark extraction uses the SEPARATE Pose + Hand landmarker tasks (not the
combined Holistic task, which crashes mid-video on mediapipe 0.10.14 / Windows:
"Check failed: holder_ != nullptr The packet is empty"). This script fetches
both ``.task`` bundles into ``models/`` (gitignored).

Run from the repo root:

    python tools/download_mediapipe_bundle.py

[VERIFY] the asset URLs at run time — Google occasionally moves the published
Tasks bundles. If a URL 404s, find the current one on the MediaPipe model card
and pass it with --pose-url / --hand-url.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# [VERIFY] current published URLs for the Pose + Hand Tasks bundles.
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_DEST = Path("models") / "pose_landmarker.task"
HAND_DEST = Path("models") / "hand_landmarker.task"


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (creating parent dirs), skipping if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[skip] already present: {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"[get ] {url}")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted Google host
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] download failed: {exc}", file=sys.stderr)
        print(
            "       The asset URL may have moved — find the current bundle URL "
            "on the MediaPipe model card and re-run with --pose-url/--hand-url.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(f"[ok  ] saved: {dest} ({dest.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download MediaPipe Pose + Hand .task bundles")
    ap.add_argument("--pose-url", default=POSE_URL, help="pose bundle URL [VERIFY]")
    ap.add_argument("--hand-url", default=HAND_URL, help="hand bundle URL [VERIFY]")
    ap.add_argument("--pose-dest", default=str(POSE_DEST))
    ap.add_argument("--hand-dest", default=str(HAND_DEST))
    args = ap.parse_args(argv)
    download(args.pose_url, Path(args.pose_dest))
    download(args.hand_url, Path(args.hand_dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
