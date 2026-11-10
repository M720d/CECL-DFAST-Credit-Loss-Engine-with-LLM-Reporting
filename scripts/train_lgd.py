"""Phase 4: LGD (cure x fractional-logit severity) and EAD (amortization schedule).

Development defaults: through lgd.max_default_period. Train on defaults <= train_default_max,
test out-of-time on later ones, then refit on all development defaults (production).
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cecl.config import load_config  # noqa: E402
from cecl.hazard.macro import load_fmhpi_states  # noqa: E402
from cecl.lgd.data import build_lgd_vintage  # noqa: E402
from cecl.lgd.ead import scheduled_balance  # noqa: E402
from cecl.lgd.model import fit_lgd  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.validation import gini  # noqa: E402


def evaluate(model, df: pl.DataFrame, label: str) -> dict:
    res = df.filter(pl.col("resolution").is_in(["liquidated", "cured"]))
    liq = res.filter(pl.col("resolution") == "liquidated")
    yc = (res["resolution"] == "cured").to_numpy().astype(int)
    pc = model.p_cure(res)
    sv, sh = liq["severity"].to_numpy(), model.severity(liq)
    lgd_act = np.where(yc == 1, 0.0, res["severity"].fill_null(0.0).to_numpy())
    lgd_hat = model.lgd(res)
    w = res["upb_default"].to_numpy()
    return {
        "sample": label, "defaults_resolved": res.height, "liquidations": liq.height,
        "cure_rate_actual": float(yc.mean()), "cure_rate_pred": float(pc.mean()), "cure_gini": gini(yc, pc),
        "severity_actual": float(sv.mean()), "severity_pred": float(sh.mean()),
        "severity_mae": float(np.abs(sv - sh).mean()), "severity_spearman": float(spearmanr(sv, sh)[0]),
        "lgd_actual": float(lgd_act.mean()), "lgd_pred": float(lgd_hat.mean()),
        "lgd_actual_upb_weighted": float((lgd_act * w).sum() / w.sum()),
        "lgd_pred_upb_weighted": float((lgd_hat * w).sum() / w.sum()),
    }


def main() -> None:
    cfg = load_config()
    lcfg, dcfg = cfg["lgd"], cfg["data"]
    proc, tables, figs = cfg.path("processed"), cfg.path("tables"), cfg.path("figures")

    hpi = load_fmhpi_states(cfg.path("raw_macro") / "fmhpi_master_file.csv")
    macro_m = pl.read_parquet(proc / "hazard" / "macro_monthly.parquet")
    lo, hi = cfg["data"]["vintages"]
    df = pl.concat([build_lgd_vintage(
        pl.scan_parquet(cfg.path("interim") / "freddie" / f"perf_{y}.parquet"),
        pl.scan_parquet(cfg.path("interim") / "freddie" / f"orig_{y}.parquet"),
        pl.scan_parquet(proc / "outcomes" / f"outcomes_{y}.parquet"), hpi, macro_m, dcfg, lcfg)
        for y in range(lo, hi + 1)], how="diagonal_relaxed")
    df.write_parquet(proc / "lgd_dataset.parquet")

    resolution_counts = dict(df.group_by("resolution").len().iter_rows())
    dev = df.filter(pl.col("default_period") <= lcfg["max_default_period"])
    dev_res = dict(dev.group_by("resolution").len().iter_rows())
    liq_dev = dev.filter(pl.col("resolution") == "liquidated")
    clip_info = {"share_below_0": float((liq_dev["severity_raw"] < 0).mean()),
                 "share_above_1": float((liq_dev["severity_raw"] > 1).mean()),
                 "mean_raw": float(liq_dev["severity_raw"].mean()),
                 "mean_clipped": float(liq_dev["severity"].mean()),
                 "median_raw": float(liq_dev["severity_raw"].median())}

    train = dev.filter(pl.col("default_period") <= lcfg["train_default_max"])
    test = dev.filter(pl.col("default_period") > lcfg["train_default_max"])
    m_dev = fit_lgd(train, lcfg)
    m_prod = fit_lgd(dev, lcfg)
    for name, m in (("dev", m_dev), ("prod", m_prod)):
        c, s = m.tables()
        c.write_csv(tables / f"lgd_cure_coefficients_{name}.csv")
        s.write_csv(tables / f"lgd_severity_coefficients_{name}.csv")
        with open(proc / "models" / f"lgd_{name}.pkl", "wb") as fh:
            pickle.dump(m, fh)

    ev_train, ev_test = evaluate(m_dev, train, "train <= 2016"), evaluate(m_dev, test, "OOT 2017-2021")
    print("train", ev_train, "\ntest", ev_test, flush=True)

    # By default year (development model): actual vs predicted cure, severity, LGD
    rows = []
    for (yr,), g in dev.filter(pl.col("resolution").is_in(["liquidated", "cured"])).group_by("default_year"):
        e = evaluate(m_dev, g, str(yr)) if g.filter(pl.col("resolution") == "liquidated").height >= 20 else None
        if e:
            rows.append({"default_year": int(yr), **{k: v for k, v in e.items() if k != "sample"}})
    by_year = pl.DataFrame(rows).sort("default_year")
    by_year.write_csv(tables / "lgd_backtest_by_default_year.csv")

    # Sensitivities on the production model (test-period defaults)
    base = m_prod.lgd(test.filter(pl.col("resolution").is_in(["liquidated", "cured"])))
    shocked = test.filter(pl.col("resolution").is_in(["liquidated", "cured"]))
    sens = {
        "hpi_minus_20pct": float(m_prod.lgd(shocked.with_columns(pl.col("mtm_ltv_default") / 0.8)).mean() / base.mean()),
        "hpi_fwd_minus_10pct": float(m_prod.lgd(shocked.with_columns(pl.col("hpi_fwd") - 0.10)).mean() / base.mean()),
        "hpi_minus_20pct_and_fwd_minus_10pct": float(m_prod.lgd(shocked.with_columns(
            pl.col("mtm_ltv_default") / 0.8, pl.col("hpi_fwd") - 0.10)).mean() / base.mean()),
        "ur_plus_2pt": float(m_prod.lgd(shocked.with_columns(pl.col("ur") + 2, pl.col("ur_yoy") + 2)).mean() / base.mean()),
    }

    # Discounting sensitivity: PV of net recoveries at the note rate over months to disposition
    l = liq_dev.filter(pl.col("months_to_disposition").is_not_null())
    df_ = (1 + l["rate_default"].to_numpy() / 1200.0) ** (-l["months_to_disposition"].to_numpy())
    loss = l["severity"].to_numpy()
    disc = np.clip(1 - (1 - loss) * df_, 0, 1)
    discount = {"severity_nominal": float(loss.mean()), "severity_discounted_note_rate": float(disc.mean()),
                "median_months_to_disposition": float(np.median(l["months_to_disposition"].to_numpy()))}

    # EAD: actual UPB at default vs scheduled balance projected L months earlier
    e = dev.filter(pl.col("upb_pre").is_not_null() & (pl.col("upb_pre") > 0) & pl.col("rem_pre").is_not_null())
    sched = scheduled_balance(e["upb_pre"].to_numpy(), e["rate_pre"].to_numpy(), e["rem_pre"].to_numpy(),
                              np.full(e.height, lcfg["ead_lookback_months"]))
    act = e["upb_default"].to_numpy()
    ok = sched > 0.01 * e["upb_pre"].to_numpy()   # avoid near-zero denominators at maturity
    ratio = act[ok] / sched[ok]
    ead = {"n": int(ok.sum()), "ratio_mean": float(ratio.mean()), "ratio_median": float(np.median(ratio)),
           "ead_factor_upb_weighted": float(act[ok].sum() / sched[ok].sum()),
           "share_ratio_above_1_05": float((ratio > 1.05).mean()), "share_ratio_below_0_95": float((ratio < 0.95).mean())}
    print("ead", ead, "\nsens", sens, "\ndiscount", discount, flush=True)

    # Figure
    x = by_year["default_year"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (act_c, pred_c, title) in zip(ax, [("cure_rate_actual", "cure_rate_pred", "Cure rate after default"),
                                              ("severity_actual", "severity_pred", "Severity given liquidation"),
                                              ("lgd_actual_upb_weighted", "lgd_pred_upb_weighted", "LGD (UPB-weighted)")]):
        a.plot(x, by_year[act_c] * 100, "o-", color="#333", label="Actual")
        a.plot(x, by_year[pred_c] * 100, "s--", color="#c0504d", label="Predicted (dev model)")
        a.axvspan(lcfg["train_default_max"] // 100 + 0.5, x.max() + 0.5, color="#f2e6c9", alpha=0.6, label="Out-of-time")
        a.set_title(title); a.set_ylabel("%"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "lgd_backtest.png", dpi=150); plt.close(fig)

    update_metrics(cfg.path("metrics"), "phase4", {
        "resolution_counts_all": resolution_counts, "resolution_counts_dev": dev_res,
        "severity_clipping": clip_info, "train": ev_train, "test_oot": ev_test,
        "by_default_year": by_year.to_dicts(), "sensitivities_prod": sens, "discounting": discount, "ead": ead,
        "state_timeline_months_top": sorted(m_prod.state_timeline.items(), key=lambda kv: -kv[1])[:5],
        "zb16_treatment": lcfg["zb16_treatment"],
    })


if __name__ == "__main__":
    main()
