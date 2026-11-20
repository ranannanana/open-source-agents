"""Pre-write screen for model text that later reaches another model's prompt.

Idea from adk-memshield's pre_write hook: memory that one model writes and another reads
is a prompt-injection path, so instruction-like text is stopped before it is stored. The
patterns are deliberately narrow; they catch steering attempts, not normal prose.
"""

from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|messages)|"
    r"\byou are now\b|\bsystem prompt\b|\bnew instructions\s*:|"
    r"disregard .{0,30}(rules|instructions)|\bact as (an? )?(admin|root|system)\b|"
    r"</?\s*(system|untrusted)\s*>", re.I)

WITHHELD = "[withheld: instruction-like text]"


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text or ""))
