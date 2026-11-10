"""Phase 3: competing-risk hazard. Development fit (<= Dec 2017), out-of-time backtest
(2018-2025, different loans), production refit (<= Dec 2025), shock sensitivities.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cecl.config import load_config  # noqa: E402
from cecl.hazard.model import design, fit_hazard  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.validation import gini  # noqa: E402

T0 = time.time()
EXPECTED_SIGNS = {  # economic priors checked on the production model (default, prepay)
    "fico": (-1, None), "dti": (1, None), "sato": (1, None), "ur": (1, None), "ur_yoy": (1, None),
    "single_borrower": (1, None), "occ_invest": (1, None), "incentive": (None, 1),
}


def log(m: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def fit_on(df: pl.DataFrame, hcfg: dict):
    X, names = design(df, hcfg)
    groups = df["loan_id"].cast(pl.Categorical).to_physical().to_numpy()
    return fit_hazard(X, df["y"].to_numpy(), groups, names, hcfg["survive_sample_rate"])


def predict_valid(model, files: list[Path], hcfg: dict, period_max: int, shock=None) -> pl.DataFrame:
    """Monthly hazards for every valid loan-month; optional covariate shock (for sensitivities)."""
    out = []
    for f in files:
        df = pl.read_parquet(f).filter(pl.col("period") <= period_max)
        if shock is not None:
            df = df.with_columns(**shock)
        X, _ = design(df, hcfg)
        h = model.hazards(X)
        out.append(df.select("loan_id", "period", "y").with_columns(
            pd_hat=pl.Series(h[:, 0]), pp_hat=pl.Series(h[:, 1]), vintage=pl.lit(int(f.stem[-4:]))))
    return pl.concat(out)


def main() -> None:
    cfg = load_config()
    hcfg = cfg["hazard"]
    base = cfg.path("processed") / "hazard"
    tables, figs = cfg.path("tables"), cfg.path("figures")
    models = cfg.path("processed") / "models"

    fit_all = pl.concat([pl.read_parquet(f) for f in sorted((base / "fit").glob("*.parquet"))])
    for a, b in hcfg["exclude_estimation_periods"]:
        fit_all = fit_all.filter(~pl.col("period").is_between(a, b))
    dev = fit_all.filter(pl.col("period") <= hcfg["dev_period_max"])
    prod = fit_all.filter(pl.col("period") <= hcfg["prod_period_max"])
    log(f"dev rows={dev.height:,} prod rows={prod.height:,}")

    m_dev = fit_on(dev, hcfg)
    log(f"dev fit: events default={m_dev.n_events[0]:,} prepay={m_dev.n_events[1]:,}")
    m_prod = fit_on(prod, hcfg)
    log(f"prod fit: events default={m_prod.n_events[0]:,} prepay={m_prod.n_events[1]:,}")
    for name, m in (("dev", m_dev), ("prod", m_prod)):
        m.table().write_csv(tables / f"hazard_coefficients_{name}.csv")
        with open(models / f"hazard_{name}.pkl", "wb") as fh:
            pickle.dump(m, fh)

    # ---------------- Backtest on valid loans with the development model ----------------
    vfiles = sorted((base / "valid").glob("*.parquet"))
    v = predict_valid(m_dev, vfiles, hcfg, hcfg["prod_period_max"])
    lo, hi = hcfg["valid_period"]
    v = v.with_columns(oot=pl.col("period") >= lo, q=(pl.col("period") // 100) * 10 + ((pl.col("period") % 100 - 1) // 3 + 1))
    by_q = (v.group_by("q").agg(n=pl.len(), oot=pl.col("oot").first(),
                                d_act=(pl.col("y") == 1).sum(), d_hat=pl.col("pd_hat").sum(),
                                p_act=(pl.col("y") == 2).sum(), p_hat=pl.col("pp_hat").sum()).sort("q"))
    by_q.write_csv(tables / "hazard_backtest_by_quarter.csv")

    def summ(frame: pl.DataFrame, label: str) -> dict:
        yd, yp = (frame["y"] == 1).to_numpy().astype(int), (frame["y"] == 2).to_numpy().astype(int)
        return {"sample": label, "loan_months": frame.height,
                "defaults_actual": int(yd.sum()), "defaults_pred": float(frame["pd_hat"].sum()),
                "default_actual_to_pred": float(yd.sum() / frame["pd_hat"].sum()),
                "prepays_actual": int(yp.sum()), "prepays_pred": float(frame["pp_hat"].sum()),
                "prepay_actual_to_pred": float(yp.sum() / frame["pp_hat"].sum()),
                "default_gini_monthly": gini(yd, frame["pd_hat"].to_numpy()),
                "prepay_gini_monthly": gini(yp, frame["pp_hat"].to_numpy())}

    # Forbearance history (known at any forecast date): drives post-COVID under-prediction. D-026
    ob = pl.concat([pl.read_parquet(f).select("loan_id", "ever_forbearance")
                    for f in sorted((cfg.path("processed") / "outcomes").glob("*.parquet"))])
    v = v.join(ob, on="loan_id", how="left").with_columns(pl.col("ever_forbearance").fill_null(False))
    in_time = summ(v.filter(~pl.col("oot")), "valid loans, months <= 2017 (in-time)")
    oot = summ(v.filter(pl.col("oot")), "valid loans, 2018-2025 (out-of-time)")
    ca, cb = hcfg["exclude_estimation_periods"][0]
    oot_ex_covid = summ(v.filter(pl.col("oot") & ~pl.col("period").is_between(ca, cb)), "OOT excl. COVID policy months")
    covid = summ(v.filter(pl.col("period").is_between(ca, cb)), "COVID policy months only")
    oot_xc = v.filter(pl.col("oot") & ~pl.col("period").is_between(ca, cb))
    oot_never_forb = summ(oot_xc.filter(~pl.col("ever_forbearance")), "OOT ex-COVID, never in forbearance")
    oot_ever_forb = summ(oot_xc.filter(pl.col("ever_forbearance")), "OOT ex-COVID, forbearance history")
    post = v.filter((pl.col("period") > cb) & ~pl.col("ever_forbearance")).with_columns(
        grp=pl.when(pl.col("vintage") >= 2022).then(pl.lit("2022+")).when(pl.col("vintage") >= 2020)
        .then(pl.lit("2020-2021")).otherwise(pl.lit("<=2019")))
    post_covid_never_forb = {g: summ(post.filter(pl.col("grp") == g), f"post-COVID, never forbearance, vintages {g}")
                             for g in ("<=2019", "2020-2021", "2022+")}
    log(f"OOT ex-COVID never-forbearance default A/P={oot_never_forb['default_actual_to_pred']:.3f} "
        f"(share of defaults from forbearance-history loans "
        f"{oot_ever_forb['defaults_actual'] / (oot_ever_forb['defaults_actual'] + oot_never_forb['defaults_actual']):.2f})")
    by_year = (v.filter(pl.col("oot")).group_by(pl.col("period") // 100)
               .agg(d_act=(pl.col("y") == 1).sum(), d_hat=pl.col("pd_hat").sum(),
                    p_act=(pl.col("y") == 2).sum(), p_hat=pl.col("pp_hat").sum()).sort("period")
               .with_columns(d_ratio=pl.col("d_act") / pl.col("d_hat"), p_ratio=pl.col("p_act") / pl.col("p_hat")))
    log(f"OOT default A/P={oot['default_actual_to_pred']:.3f} prepay A/P={oot['prepay_actual_to_pred']:.3f} "
        f"default gini={oot['default_gini_monthly']:.3f}")

    # ---------------- Shock sensitivities (production model, OOT valid months) ----------------
    def agg(shock):
        s = predict_valid(m_prod, vfiles, hcfg, hcfg["prod_period_max"], shock).filter(pl.col("period") >= lo)
        return float(s["pd_hat"].sum()), float(s["pp_hat"].sum())

    b_d, b_p = agg(None)
    sens = {}
    for label, shock in {
        "ur_plus_2pt": {"ur": pl.col("ur") + 2.0, "ur_yoy": pl.col("ur_yoy") + 2.0},
        "hpi_minus_20pct": {"mtm_ltv": pl.col("mtm_ltv") / 0.8},
        "mortgage_rate_minus_1pt": {"incentive": pl.col("incentive") + 1.0},
    }.items():
        d_, p_ = agg(shock)
        sens[label] = {"default_multiplier": d_ / b_d, "prepay_multiplier": p_ / b_p}
    log(f"sensitivities {sens}")

    # ---------------- Sign checks ----------------
    tbl = m_prod.table()
    signs = {}
    for f, (sd, sp) in EXPECTED_SIGNS.items():
        r = tbl.filter(pl.col("feature") == f).row(0, named=True)
        signs[f] = {"default_ok": None if sd is None else bool(np.sign(r["beta_default"]) == sd),
                    "prepay_ok": None if sp is None else bool(np.sign(r["beta_prepay"]) == sp)}

    # ---------------- Figures ----------------
    qd = by_q.with_columns(x=(pl.col("q") // 10) + ((pl.col("q") % 10) - 1) / 4.0,
                           dr=pl.col("d_act") / pl.col("n") * 1200, dh=pl.col("d_hat") / pl.col("n") * 1200,
                           cpr=(1 - (1 - pl.col("p_act") / pl.col("n")) ** 12) * 100,
                           cprh=(1 - (1 - pl.col("p_hat") / pl.col("n")) ** 12) * 100)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(qd["x"], qd["dr"], color="#333", lw=1.4, label="Actual")
    ax[0].plot(qd["x"], qd["dh"], color="#c0504d", lw=1.4, label="Predicted (dev model)")
    ax[0].set_title("Default rate, annualized (% of loans at risk)")
    ax[1].plot(qd["x"], qd["cpr"], color="#333", lw=1.4, label="Actual")
    ax[1].plot(qd["x"], qd["cprh"], color="#1f4e79", lw=1.4, label="Predicted (dev model)")
    ax[1].set_title("Prepayment speed, CPR (%)")
    for a in ax:
        a.axvspan(lo // 100, 2026, color="#f2e6c9", alpha=0.5, label="Out-of-time")
        a.axvspan(ca // 100 + (ca % 100 - 1) / 12, cb // 100 + cb % 100 / 12, color="#bbb", alpha=0.4,
                  label="COVID policy (excluded from estimation)")
        a.legend(fontsize=8); a.grid(alpha=0.3); a.set_xlim(1999, 2026)
    fig.suptitle("Hazard backtest on held-out loans (fit through Dec 2017)", fontsize=11)
    fig.tight_layout(); fig.savefig(figs / "hazard_backtest.png", dpi=150); plt.close(fig)

    update_metrics(cfg.path("metrics"), "phase3", {
        "dev": {"rows": dev.height, "defaults": m_dev.n_events[0], "prepays": m_dev.n_events[1]},
        "prod": {"rows": prod.height, "defaults": m_prod.n_events[0], "prepays": m_prod.n_events[1]},
        "n_parameters_per_outcome": len(m_prod.names),
        "backtest_in_time": in_time, "backtest_oot": oot, "backtest_oot_ex_covid": oot_ex_covid, "backtest_covid_only": covid,
        "backtest_oot_never_forbearance": oot_never_forb, "backtest_oot_forbearance_history": oot_ever_forb,
        "backtest_post_covid_never_forbearance_by_vintage": post_covid_never_forb,
        "backtest_oot_by_year": by_year.to_dicts(),
        "sensitivities_prod": sens, "sign_checks_prod": signs,
        "state_hpi_betas_top": pl.read_parquet(base / "state_hpi_betas.parquet").sort("beta", descending=True)
        .head(5).select("geo", "beta", "r2").to_dicts(),
    })
    log("done")


if __name__ == "__main__":
    main()
