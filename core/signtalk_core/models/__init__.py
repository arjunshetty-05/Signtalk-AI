"""signtalk_core.models — model definitions (Phase 2+).

Placeholder package shell for Phase 0. Later phases add the ensemble members
(PROJECT_CONTEXT Section 5.5):

    * M1 — pose-sequence model (TCN or BiGRU)      -> pose_tcn.py
    * M2 — pose-sequence model (small Transformer) -> pose_transformer.py
    * M3 — RGB-crop video classifier (optional,    -> rgb_video.py
           GPU; skipped for now but the package is
           kept shaped so an RGB branch can be
           added later)

No ML logic is defined yet.
"""

__all__: list[str] = []
