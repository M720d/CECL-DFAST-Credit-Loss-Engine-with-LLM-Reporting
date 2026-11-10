from __future__ import annotations

import pytest

from cecl.reporting.render import render


def test_render_formats_and_tables():
    ctx = {"a": {"b": 0.12345, "lst": [{"x": 3}]}}
    out = render("v={{a.b:.1%}} n={{a.lst.0.x}} t={{tbl}}", ctx, {"tbl": lambda c: "T"})
    assert out == "v=12.3% n=3 t=T"


def test_unresolved_placeholder_fails():
    with pytest.raises(KeyError):
        render("{{missing.key}}", {})
