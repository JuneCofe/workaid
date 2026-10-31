"""Detect article bodies damaged by the corpus builder.

The source JSON was split on every line that starts with ``第X条``. A cross-reference that wraps
onto a new line (``…依照本法`` / ``第四十七条规定的…``) therefore produced a phantom article whose
key collides with, and can overwrite, the real one, and left the preceding article truncated.
"""
from __future__ import annotations

import re

_FRAGMENT_LEAD = re.compile(r"^(的|之|规定|[，、；。）)]|第[一二三四五六七八九十百千零〇0-9]+[条款项章节])")
_TERMINATED = re.compile(r"([。；！？：”）)\]]|制定本(法|条例|办法|规定|解释|决定))\s*$")
MIN_BODY_CHARS = 8


def classify(content: str) -> str:
    """Return ``"fragment"`` (not a usable article), ``"truncated"`` (cut off mid-sentence) or ``""``."""
    text = (content or "").strip()
    if not text:
        return ""
    if len(text) < MIN_BODY_CHARS or _FRAGMENT_LEAD.match(text):
        return "fragment"
    if not _TERMINATED.search(text):
        return "truncated"
    return ""
