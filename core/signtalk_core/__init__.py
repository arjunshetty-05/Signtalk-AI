"""signtalk_core — accuracy-first Indian Sign Language recognition core.

This installable package holds ALL machine-learning logic for SignTalk AI v3
(PROJECT_CONTEXT Section 5.14): the same code is imported by the training
scripts and by the FastAPI server, so training and serving can never diverge.

Phase 0 ships only the package shell and the config accessor; feature,
landmark, model, fusion and decision modules are added in later phases.
"""

__version__ = "0.1.0"

from signtalk_core.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
