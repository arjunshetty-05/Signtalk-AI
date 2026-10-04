"""
core/limiter.py — SignTalk AI / Person C, Prompt C1

Single shared slowapi Limiter instance, imported by every router. Do not
instantiate a second Limiter elsewhere — slowapi expects one instance
attached to app.state.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
