"""Download the MediaPipe Tasks Holistic model bundle (PROJECT_CONTEXT 4.4 / 5.3).

The landmark extractor (:func:`signtalk_core.landmarks.create_holistic_detector`)
needs a ``.task`` model bundle at serving/preprocessing time. This script fetches
it once into ``models/`` (gitignored) so both the server and the preprocessing
pipeline can point at the same file.

Run from the repo root:

    python tools/download_mediapipe_bundle.py

[VERIFY] the asset URL at run time — Google occasionally moves the published
Tasks bundles. If the default URL 404s, pass a current one with --url (find it
on the MediaPipe "HolisticLandmarker" / "Pose landmarker" model card).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# [VERIFY] current published URL for the Holistic Tasks bundle.
DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
DEFAULT_DEST = Path("models") / "holistic_landmarker.task"


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
            "       The asset URL may have moved — find the current "
            "'holistic_landmarker.task' URL on the MediaPipe model card and "
            "re-run with --url <url>.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(f"[ok  ] saved: {dest} ({dest.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download the MediaPipe Holistic .task bundle")
    ap.add_argument("--url", default=DEFAULT_URL, help="model bundle URL [VERIFY]")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="destination path")
    args = ap.parse_args(argv)
    download(args.url, Path(args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
