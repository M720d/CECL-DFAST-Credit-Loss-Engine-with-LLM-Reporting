from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_metrics import compare  # noqa: E402


def test_identical_and_differing():
    a = {"phase1": {"x": 1.0, "y": [1, 2, {"z": 3.0}]}, "phase5": {"acl": 100.0}}
    b = {"phase1": {"x": 1.0, "y": [1, 2, {"z": 3.0}]}, "phase5": {"acl": 100.5}}
    r = compare(a, b)
    assert r["sections"]["phase1"]["differing"] == 0
    assert r["values_differing"] == 1 and r["worst"] == "phase5.acl"
    assert abs(r["max_rel_diff"] - 0.5 / 100.5) < 1e-12
