"""Layout contract against the raw files, and the metrics writer."""
from __future__ import annotations

import json
import math

import pytest

from cecl.config import load_config
from cecl.data.schema import ORIG_FIELDS, PERF_FIELDS
from cecl.metrics import update_metrics

CFG = load_config()
RAW = CFG.path("raw_freddie")


def test_layout_field_counts():
    assert len(ORIG_FIELDS) == 31 and len(PERF_FIELDS) == 35
    assert len({n for n, _ in ORIG_FIELDS}) == 31 and len({n for n, _ in PERF_FIELDS}) == 35


@pytest.mark.skipif(not (RAW / "sample_2010").exists(), reason="raw Freddie data not downloaded")
@pytest.mark.parametrize("kind,fields", [("orig", ORIG_FIELDS), ("perf", PERF_FIELDS)])
def test_raw_files_match_layout(kind, fields):
    with open(RAW / "sample_2010" / f"sample_{kind}_2010.txt") as f:
        for _ in range(1000):
            line = f.readline().rstrip("\n")
            # Performance lines end with an empty final field (trailing '|').
            assert len(line.split("|")) == len(fields), line


def test_metrics_section_replace_and_nan(tmp_path):
    p = tmp_path / "m.json"
    update_metrics(p, "a", {"x": 1.0, "bad": math.nan})
    update_metrics(p, "b", {"y": 2})
    update_metrics(p, "a", {"x": 3.0})
    d = json.loads(p.read_text())
    assert d["a"] == {"x": 3.0} and d["b"] == {"y": 2}
    update_metrics(p, "c", {"bad": math.nan})
    assert json.loads(p.read_text())["c"]["bad"] is None
