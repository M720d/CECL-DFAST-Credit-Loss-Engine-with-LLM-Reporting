"""Reproducibility check: compare two metrics.json files value by value (phases 1-6).

  uv run python scripts/compare_metrics.py <fresh_run/metrics.json> [--record]

Walks every numeric leaf in the shared sections and reports the largest relative difference.
Run-time measurements (phase1_convert) and timestamps (_meta) are excluded by design.
With --record, writes the summary to this repo's metrics.json as 'phase8_repro'.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402

SECTIONS = ["phase1", "phase2", "phase2_recalibration", "phase2_diagnostics", "phase3", "phase4", "phase5", "phase6"]


def leaves(obj, path=""):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}.{i}")


def compare(a: dict, b: dict, tol: float = 1e-9) -> dict:
    out = {"sections": {}, "values_compared": 0, "values_differing": 0, "max_rel_diff": 0.0, "worst": None, "missing": []}
    for s in SECTIONS:
        if s not in a or s not in b:
            out["missing"].append(s)
            continue
        la, lb = dict(leaves(a[s])), dict(leaves(b[s]))
        keys = la.keys() & lb.keys()
        diffs = []
        for k in keys:
            x, y = la[k], lb[k]
            if math.isnan(x) and math.isnan(y):
                continue
            rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
            diffs.append((rel, k))
        n_diff = sum(r > tol for r, _ in diffs)
        worst = max(diffs) if diffs else (0.0, None)
        out["sections"][s] = {"values": len(keys), "differing": n_diff, "max_rel_diff": worst[0],
                              "only_in_one": len(la.keys() ^ lb.keys())}
        out["values_compared"] += len(keys)
        out["values_differing"] += n_diff
        if worst[0] > out["max_rel_diff"]:
            out["max_rel_diff"], out["worst"] = worst[0], f"{s}.{worst[1]}"
    out["tolerance"] = tol
    return out


def main() -> None:
    cfg = load_config()
    fresh = json.loads(Path(sys.argv[1]).read_text())
    current = json.loads(cfg.path("metrics").read_text())
    res = compare(current, fresh)
    print(json.dumps({k: v for k, v in res.items() if k != "sections"}, indent=2))
    for s, r in res["sections"].items():
        print(f"  {s:22s} values={r['values']:>6} differing={r['differing']:>4} max_rel_diff={r['max_rel_diff']:.2e}")
    if "--record" in sys.argv:
        update_metrics(cfg.path("metrics"), "phase8_repro", res)


if __name__ == "__main__":
    main()
