"""SignTalk AI v3 FastAPI server package.

Thin HTTP layer over :mod:`signtalk_core`. All ML logic lives in the core
package; routers here only parse requests, call ``recognize_clip`` / storage,
and shape responses. Storage is SQLite (guest mode, NO Firebase — locked owner
decision, PROJECT_CONTEXT Section 2 / context.json).
"""
