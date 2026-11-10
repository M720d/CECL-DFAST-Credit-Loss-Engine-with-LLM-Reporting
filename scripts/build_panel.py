"""Phase 1: loan outcomes, PD snapshots, data-quality report, default rates by vintage.

Processes one vintage at a time (peak memory ~ one vintage's performance file).
Writes data/processed/{outcomes,snapshots}/*.parquet, results/tables/*.csv,
results/figures/default_rates_by_vintage.png, and the 'phase1' section of metrics.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.data.panel import EVENT_DEFAULT, EVENT_PREPAY, loan_outcomes, pd_snapshots  # noqa: E402
from cecl.data.qc import qc_vintage  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    lo, hi = d["vintages"]
    inter = cfg.path("interim") / "freddie"
    proc = cfg.path("processed")
    (proc / "outcomes").mkdir(parents=True, exist_ok=True)
    (proc / "snapshots").mkdir(parents=True, exist_ok=True)
    cfg.path("tables").mkdir(parents=True, exist_ok=True)
    cfg.path("figures").mkdir(parents=True, exist_ok=True)

    qc_all, rows = {}, []
    for y in range(lo, hi + 1):
        orig = pl.scan_parquet(inter / f"orig_{y}.parquet")
        perf = pl.scan_parquet(inter / f"perf_{y}.parquet")
        out = loan_outcomes(perf, d).collect()
        out = out.join(orig.select("loan_id", "orig_upb", "orig_term").collect(), on="loan_id", how="left")
        out.write_parquet(proc / "outcomes" / f"outcomes_{y}.parquet")
        snaps = pd_snapshots(perf, out.lazy(), d).collect()
        snaps.write_parquet(proc / "snapshots" / f"snapshots_{y}.parquet")
        qc_all[str(y)] = qc_vintage(orig, perf, out, d)

        # Default within 12 months of first payment (age <= 12) vs lifetime to date.
        n = out.height
        dflt = out.filter(pl.col("event") == EVENT_DEFAULT)
        rows.append({
            "vintage": y,
            "loans": n,
            "default_12m_from_orig": dflt.filter(pl.col("default_age") <= d["default_horizon_months"]).height / n,
            "default_lifetime_to_date": dflt.height / n,
            "prepay_lifetime_to_date": out.filter(pl.col("event") == EVENT_PREPAY).height / n,
            "active_at_cutoff": out.filter(pl.col("zb_code").is_null()).height / n,
            "defaults_in_forbearance": dflt.filter(pl.col("default_in_forbearance")).height,
            "defaults_cured": dflt.filter(pl.col("default_cured")).height,
            "default_lifetime_ex_cured": dflt.filter(~pl.col("default_cured")).height / n,
            "default_lifetime_spec_def": out.filter(pl.col("default_period_spec").is_not_null()).height / n,
            "snapshot_12m_default_rate_spec_def": (snaps.filter(pl.col("eligible_spec"))["target_spec"].mean()
                                                   if snaps.height else None),
            "snapshots": snaps.height,
            "snapshot_12m_default_rate": snaps["target"].mean() if snaps.height else None,
            "snapshot_defaults": int(snaps["target"].sum()) if snaps.height else 0,
            "default_dispositions_with_loss": qc_all[str(y)]["dispositions_with_loss"],
            "mean_loss_severity": (dflt.filter(pl.col("actual_loss").is_not_null())
                                   .select((pl.col("actual_loss").sum() / pl.col("zb_removal_upb").sum()))
                                   .item() if qc_all[str(y)]["dispositions_with_loss"] else None),
        })
        r = rows[-1]
        print(f"{y}: loans={n:,} 12m={r['default_12m_from_orig']:.4%} life={r['default_lifetime_to_date']:.3%} "
              f"cured={r['defaults_cured']}/{dflt.height} forb={r['defaults_in_forbearance']} snaps={r['snapshots']:,} snapPD={r['snapshot_12m_default_rate'] or 0:.3%}", flush=True)

    tbl = pl.DataFrame(rows)
    tbl.write_csv(cfg.path("tables") / "default_rates_by_vintage.csv")

    # Pooled snapshot default rate by calendar year of snapshot (macro sensitivity view).
    snaps_all = pl.scan_parquet(proc / "snapshots" / "*.parquet")
    by_year = (snaps_all.group_by((pl.col("snapshot_period") // 100).alias("snapshot_year"))
               .agg(snapshots=pl.len(), defaults=pl.col("target").sum(), pd=pl.col("target").mean(),
                    pd_spec_def=pl.col("target_spec").filter(pl.col("eligible_spec")).mean())
               .sort("snapshot_year").collect())
    by_year.write_csv(cfg.path("tables") / "snapshot_pd_by_calendar_year.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(tbl["vintage"], tbl["default_lifetime_to_date"] * 100, color="#4C72B0")
    ax[0].set_title("Cumulative default rate to Jun-2026, by vintage"); ax[0].set_ylabel("%")
    ax[1].plot(by_year["snapshot_year"], by_year["pd"] * 100, marker="o", color="#C44E52")
    ax[1].set_title("12-month default rate by snapshot year"); ax[1].set_ylabel("%")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.path("figures") / "default_rates_by_vintage.png", dpi=150)

    total = {
        "loans": int(tbl["loans"].sum()),
        "loan_months": int(sum(q["rows"] for q in qc_all.values())),
        "defaults_lifetime": int(sum(r["default_lifetime_to_date"] * r["loans"] for r in rows) + 0.5),
        "snapshots": int(tbl["snapshots"].sum()),
        "snapshot_defaults": int(tbl["snapshot_defaults"].sum()),
        "max_period": max(q["max_period"] for q in qc_all.values()),
        "dup_loan_months": sum(q["dup_loan_month"] for q in qc_all.values()),
        "loans_with_gaps": sum(q["loans_with_gaps"] for q in qc_all.values()),
        "missing_months": sum(q["missing_months"] for q in qc_all.values()),
        "unknown_zb_loans": sum(q["unknown_zb_loans"] for q in qc_all.values()),
        "default_dispositions": sum(q["default_dispositions"] for q in qc_all.values()),
        "dispositions_with_loss": sum(q["dispositions_with_loss"] for q in qc_all.values()),
        "loss_identity_pass": sum(q["loss_identity_pass"] for q in qc_all.values()),
    }
    update_metrics(cfg.path("metrics"), "phase1", {
        "totals": total, "by_vintage": rows, "by_snapshot_year": by_year.to_dicts(), "qc": qc_all,
    })
    print(total)


if __name__ == "__main__":
    main()
