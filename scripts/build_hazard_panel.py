"""Phase 3 data: monthly macro table, state HPI betas, sampled hazard panels per vintage."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.hazard.macro import (load_fed_quarterly, load_fmhpi_states, quarterly_to_monthly,  # noqa: E402
                               state_hpi_betas)
from cecl.hazard.panel import build_vintage_panel  # noqa: E402
from cecl.pd.features import load_pmms  # noqa: E402


def main() -> None:
    cfg = load_config()
    h, d, seed = cfg["hazard"], cfg["data"], cfg["seed"]
    out = cfg.path("processed") / "hazard"
    (out / "fit").mkdir(parents=True, exist_ok=True)
    (out / "valid").mkdir(parents=True, exist_ok=True)

    fed_q = load_fed_quarterly(cfg.path("raw_dfast") / "2026_Final_Historic_Domestic.csv")
    macro_m = quarterly_to_monthly(fed_q, h["macro_lag_quarters"])
    pmms_m = load_pmms(cfg.path("raw_macro") / "MORTGAGE30US.csv").with_columns(pl.col("t").cast(pl.Int32))
    hpi = load_fmhpi_states(cfg.path("raw_macro") / "fmhpi_master_file.csv")
    y0, y1 = h["beta_estimation_quarters"]
    betas = state_hpi_betas(hpi, fed_q, y0 * 4, y1 * 4 + 3)
    macro_m.write_parquet(out / "macro_monthly.parquet")
    betas.write_parquet(out / "state_hpi_betas.parquet")
    betas.write_csv(cfg.path("tables") / "hazard_state_hpi_betas.csv")

    lo, hi = h["vintages"]
    for y in range(lo, hi + 1):
        t = time.time()
        fit, valid = build_vintage_panel(
            pl.scan_parquet(cfg.path("interim") / "freddie" / f"perf_{y}.parquet"),
            pl.scan_parquet(cfg.path("interim") / "freddie" / f"orig_{y}.parquet"),
            pl.scan_parquet(cfg.path("processed") / "outcomes" / f"outcomes_{y}.parquet"),
            macro_m, pmms_m, hpi, d, h, seed)
        fit.write_parquet(out / "fit" / f"fit_{y}.parquet")
        valid.write_parquet(out / "valid" / f"valid_{y}.parquet")
        print(f"{y}: fit={fit.height:>8,} (def {int((fit['y'] == 1).sum()):>5,}, pre {int((fit['y'] == 2).sum()):>6,}) "
              f"valid={valid.height:>9,}  {time.time() - t:4.1f}s", flush=True)


if __name__ == "__main__":
    main()
