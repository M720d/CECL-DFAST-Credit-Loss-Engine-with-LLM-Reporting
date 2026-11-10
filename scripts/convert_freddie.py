"""Convert all configured Freddie Mac sample vintages to parquet (idempotent).

Records wall time, peak memory and sizes in metrics.json ('phase1_convert') when a
full conversion is run with --overwrite, so laptop-feasibility claims come from code.
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.data.convert import convert_vintage  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402


def _size(paths) -> int:
    return sum(p.stat().st_size for p in paths)


def main() -> None:
    cfg = load_config()
    lo, hi = cfg["data"]["vintages"]
    raw, out = cfg.path("raw_freddie"), cfg.path("interim") / "freddie"
    overwrite = "--overwrite" in sys.argv
    t0, rows = time.time(), {"orig": 0, "perf": 0}
    for y in range(lo, hi + 1):
        t = time.time()
        s = convert_vintage(raw, out, y, overwrite=overwrite)
        rows["orig"] += s["orig_rows"]
        rows["perf"] += s["perf_rows"]
        print(f"{y}: orig={s['orig_rows']:>7,} perf={s['perf_rows']:>11,}  {time.time()-t:5.1f}s", flush=True)
    if overwrite:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6
        update_metrics(cfg.path("metrics"), "phase1_convert", {
            "seconds": round(time.time() - t0, 1),
            "peak_rss_gb": round(peak_gb, 2),
            "raw_gb": round(_size(raw.glob("sample_*/*.txt")) / 1e9, 2),
            "parquet_gb": round(_size(out.glob("*.parquet")) / 1e9, 2),
            "orig_rows": rows["orig"], "perf_rows": rows["perf"],
        })


if __name__ == "__main__":
    main()
