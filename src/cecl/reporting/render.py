"""Template rendering: {{dotted.path:format}} placeholders resolved from a context dict.

Used for README.md and the model development document so that no result is ever typed by hand.
Unresolved placeholders raise, so a missing metric fails the build instead of printing a blank.
"""
from __future__ import annotations

import re
from typing import Callable

PLACEHOLDER = re.compile(r"\{\{\s*([\w.<=+\-]+)\s*(?::([^}]+))?\}\}")


def lookup(ctx: dict, path: str):
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def render(text: str, ctx: dict, tables: dict[str, Callable[[dict], str]] | None = None) -> str:
    tables = tables or {}

    def sub(mt: re.Match) -> str:
        key, fmt = mt.group(1), mt.group(2)
        if key in tables:
            return tables[key](ctx)
        val = lookup(ctx, key)
        return format(val, fmt) if fmt else str(val)

    out = PLACEHOLDER.sub(sub, text)
    left = re.findall(r"\{\{.*?\}\}", out)
    if left:
        raise ValueError(f"Unresolved placeholders: {left}")
    return out
