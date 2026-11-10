"""Phase 5: CECL allowance at the as-of date.

Steps: historical loss rate -> overlay calibration on past year-ends -> scenario projections
(baseline / downside / upside) -> R&S + reversion blending -> probability-weighted ACL ->
sensitivities -> Jan-Jun 2026 backtest. Everything reported goes to metrics.json['phase5'].
Use --no-sens to skip the sensitivity reruns.
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

from cecl.allowance.engine import project  # noqa: E402
from cecl.allowance.paths import long_run, monthly_path, scenario_quarterly  # noqa: E402
from cecl.allowance.portfolio import SEGMENTS, build_portfolio  # noqa: E402
from cecl.config import load_config  # noqa: E402
from cecl.hazard.macro import load_fed_quarterly, load_fmhpi_states  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.features import load_pmms  # noqa: E402

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def to_t(ym: int) -> int:
    return (ym // 100) * 12 + ym % 100 - 1


def hist_loss_rate(cfg) -> tuple[float, pl.DataFrame]:
    """Annual credit loss / average outstanding UPB, averaged over hist_loss_years (D-035)."""
    lo, hi = cfg["cecl"]["hist_loss_years"]
    lg = pl.read_parquet(cfg.path("processed") / "lgd_dataset.parquet").filter(
        pl.col("default_year").is_between(lo, hi))
    lg = lg.with_columns(lgd_act=pl.when(pl.col("resolution") == "cured").then(0.0)
                         .when(pl.col("resolution") == "liquidated").then(pl.col("severity")).otherwise(None))
    by_y = lg.group_by("default_year").agg(
        def_upb=pl.col("upb_default").sum(),
        lgd=(pl.col("lgd_act") * pl.col("upb_default")).sum() / pl.col("upb_default").filter(pl.col("lgd_act").is_not_null()).sum())
    out = (pl.scan_parquet(cfg.path("interim") / "freddie" / "perf_*.parquet")
           .filter((pl.col("period") // 100).is_between(lo, hi))
           .group_by("period").agg(upb=pl.col("cur_upb").sum())
           .group_by(year=pl.col("period") // 100).agg(avg_upb=pl.col("upb").mean()).collect())
    t = by_y.join(out, left_on="default_year", right_on="year").with_columns(
        loss=pl.col("def_upb") * pl.col("lgd")).with_columns(loss_rate=pl.col("loss") / pl.col("avg_upb")).sort("default_year")
    return float(t["loss_rate"].mean()), t


def blend(proj, rs_m: int, rev_m: int, hist_monthly: float) -> np.ndarray:
    """Month-by-segment expected loss after output reversion to the historical loss rate."""
    H = proj.month_el.shape[0]
    k = np.arange(1, H + 1)
    w = np.where(k <= rs_m, 1.0, np.clip(1.0 - (k - rs_m) / max(rev_m, 1e-9), 0.0, 1.0) if rev_m > 0 else 0.0)
    w = np.where(k <= rs_m, 1.0, w)
    return w[:, None] * proj.month_el + (1 - w[:, None]) * hist_monthly * proj.month_exposure


def main() -> None:
    cfg = load_config()
    c, hcfg, lcfg = cfg["cecl"], cfg["hazard"], cfg["lgd"]
    raw, proc = cfg.path("raw_dfast"), cfg.path("processed")
    tables, figs = cfg.path("tables"), cfg.path("figures")
    pm = load_pmms(cfg.path("raw_macro") / "MORTGAGE30US.csv").with_columns(pl.col("t").cast(pl.Int32))
    hpi = load_fmhpi_states(cfg.path("raw_macro") / "fmhpi_master_file.csv")
    fh = load_fed_quarterly(raw / "2026_Final_Historic_Domestic.csv")
    base = load_fed_quarterly(raw / "2026_Final_Supervisory_Baseline_Domestic.csv")
    sa = load_fed_quarterly(raw / "2026_Final_Supervisory_Severely_Adverse_Domestic.csv")
    betas = pl.read_parquet(proc / "hazard" / "state_hpi_betas.parquet")
    hz = pickle.load(open(proc / "models" / "hazard_prod.pkl", "rb"))
    lg = pickle.load(open(proc / "models" / "lgd_prod.pkl", "rb"))
    lr = long_run(fh, c["long_run"])
    H = c["max_horizon_months"]

    # ---------------- 1. Historical loss rate ----------------
    hist_annual, hist_tbl = hist_loss_rate(cfg)
    hist_tbl.write_csv(tables / "cecl_historical_loss_rates.csv")
    log(f"historical annual loss rate {hist_annual:.4%}")

    # ---------------- 2. Overlay calibration on past year-ends ----------------
    oc = c["overlay"]
    rows = []
    for d in oc["calib_as_of"]:
        td = to_t(d)
        pf_d = build_portfolio(cfg, d, pm, hpi, outcome_months=oc["horizon_months"])
        pf_d = pf_d.filter((pl.col("segment") != "S4_default") & pl.col("orig_value").is_not_null())
        # PD-only, so no forward HPI buffer is needed (realized history ends 2025Q4)
        path = monthly_path(fh, pm, hpi, betas, td - 24, td + oc["horizon_months"],
                            hcfg["macro_lag_quarters"], td + oc["horizon_months"], "actual")
        pr = project(pf_d, path, hz, lg, hcfg, lcfg, td, oc["horizon_months"], pd_only=True)
        g = pf_d.select("segment", "default_next").with_columns(pred=pl.Series(pr.loan_pd)).group_by("segment").agg(
            n=pl.len(), actual=pl.col("default_next").sum(), pred=pl.col("pred").sum())
        rows.append(g.with_columns(as_of=pl.lit(d)))
        log(f"overlay calib {d}: {pf_d.height:,} loans")
    cal = pl.concat(rows)
    cal.write_csv(tables / "cecl_overlay_calibration_by_date.csv")
    seg_cal = cal.group_by("segment").agg(pl.col("n").sum(), pl.col("actual").sum(), pl.col("pred").sum()).with_columns(
        ap=pl.col("actual") / pl.col("pred")).sort("segment")
    ap0 = seg_cal.filter(pl.col("segment") == "S0_clean")["ap"].item()
    seg_cal = seg_cal.with_columns(multiplier=pl.col("ap") / ap0)
    seg_cal.write_csv(tables / "cecl_overlay_multipliers.csv")
    overlay = {r["segment"]: r["multiplier"] for r in seg_cal.iter_rows(named=True) if r["segment"] != "S0_clean"}
    log(f"overlay multipliers {overlay} (S0 A/P {ap0:.2f})")

    # ---------------- 3. Portfolio and scenario runs ----------------
    tas = to_t(c["as_of"])
    pf = build_portfolio(cfg, c["as_of"], pm, hpi, outcome_months=c["backtest_months"])
    pf = pf.with_columns(pl.col("orig_value").fill_null(pl.col("orig_upb") / 0.8))   # 1 loan without LTV
    upb0 = float(pf["cur_upb"].sum())
    nq = (H + 48) // 3 + 4

    def run(scale: float, rs_q: int, rev_q: int, ov=overlay):
        fq = scenario_quarterly(fh, base, sa, scale, rs_q, rev_q, lr, c["ur_floor"], nq)
        path = monthly_path(fq, pm, hpi, betas, tas - 24, tas + H + 24, hcfg["macro_lag_quarters"], tas, "betas")
        return project(pf, path, hz, lg, hcfg, lcfg, tas, H, ov, oc), fq

    def acl_of(proj, rs_q, rev_q):
        model = proj.month_el.sum()
        blended = blend(proj, rs_q * 3, rev_q * 3, hist_annual / 12.0)
        return {"acl_model_input_reversion": float(model), "acl_output_reversion": float(blended.sum()),
                "by_segment_output": dict(zip(SEGMENTS, blended.sum(0).tolist())),
                "by_segment_model": dict(zip(SEGMENTS, proj.month_el.sum(0).tolist()))}

    main_key = "acl_output_reversion" if c["reversion_method"] == "output" else "acl_model_input_reversion"
    results, projs, paths_q = {}, {}, {}
    for name, sc in c["scenarios"].items():
        pr, fq = run(sc["scale"], c["rs_quarters"], c["reversion_quarters"])
        projs[name], paths_q[name] = pr, fq
        a = acl_of(pr, c["rs_quarters"], c["reversion_quarters"])
        w_upb = pf["cur_upb"].to_numpy()
        results[name] = {**a, "weight": sc["weight"], "acl_pct_upb": a[main_key] / upb0,
                         "lifetime_pd_upb_wtd": float((pr.loan_pd * w_upb).sum() / w_upb.sum()),
                         "lifetime_pd_loan_avg": float(pr.loan_pd.mean()),
                         "avg_lgd_on_defaults": float(pr.month_el.sum() / pr.month_def_upb.sum()),
                         "expected_wal_years": float(pr.month_exposure.sum() / 12.0 / upb0),
                         "lifetime_prepay_prob": float(pr.loan_prepay.mean())}
        log(f"{name}: ACL {results[name][main_key] / upb0:.3%} of UPB (model-only {a['acl_model_input_reversion'] / upb0:.3%})")
    weighted = sum(results[s]["weight"] * results[s][main_key] for s in results)
    weighted_model = sum(results[s]["weight"] * results[s]["acl_model_input_reversion"] for s in results)

    # ACL by segment and vintage group (weighted)
    seg_tbl = pl.DataFrame({"segment": SEGMENTS,
                            "loans": [pf.filter(pl.col("segment") == s).height for s in SEGMENTS],
                            "upb": [float(pf.filter(pl.col("segment") == s)["cur_upb"].sum()) for s in SEGMENTS],
                            "acl": [sum(results[n]["weight"] * results[n]["by_segment_output" if main_key == "acl_output_reversion" else "by_segment_model"][s]
                                        for n in results) for s in SEGMENTS]}).with_columns(acl_pct=pl.col("acl") / pl.col("upb"))
    seg_tbl.write_csv(tables / "cecl_acl_by_segment.csv")
    vin = pf.select("vintage", "cur_upb").with_columns(
        el=pl.Series(sum(results[n]["weight"] * projs[n].loan_el for n in results)),
        grp=pl.when(pl.col("vintage") <= 2008).then(pl.lit("<=2008")).when(pl.col("vintage") <= 2019)
        .then(pl.lit("2009-2019")).when(pl.col("vintage") <= 2021).then(pl.lit("2020-2021")).otherwise(pl.lit("2022-2025")))
    vin_tbl = vin.group_by("grp").agg(loans=pl.len(), upb=pl.col("cur_upb").sum(), el_model=pl.col("el").sum()).with_columns(
        el_pct=pl.col("el_model") / pl.col("upb")).sort("grp")
    vin_tbl.write_csv(tables / "cecl_acl_by_vintage_model_basis.csv")

    # ---------------- 4. Backtest Jan-Jun 2026 (baseline, never used in development) ----------------
    B = c["backtest_months"]
    pr6, _ = run(c["scenarios"]["baseline"]["scale"], c["rs_quarters"], c["reversion_quarters"])
    non4 = (pf["segment"] != "S4_default").to_numpy()
    bt = pf.select("segment", "default_next", "prepay_next").filter(pl.Series(non4))
    # 6-month cumulative probabilities from a short projection
    pf6 = pf.filter(pl.Series(non4))
    fq_b = scenario_quarterly(fh, base, sa, 0.0, c["rs_quarters"], c["reversion_quarters"], lr, c["ur_floor"], nq)
    path_b = monthly_path(fq_b, pm, hpi, betas, tas - 24, tas + B + 24, hcfg["macro_lag_quarters"], tas, "betas")
    p6 = project(pf6, path_b, hz, lg, hcfg, lcfg, tas, B, overlay, oc, pd_only=True)
    bt = bt.with_columns(pd6=pl.Series(p6.loan_pd), pp6=pl.Series(p6.loan_prepay))
    bt_seg = bt.group_by("segment").agg(n=pl.len(), def_actual=pl.col("default_next").sum(), def_pred=pl.col("pd6").sum(),
                                        pre_actual=pl.col("prepay_next").sum(), pre_pred=pl.col("pp6").sum()).sort("segment")
    bt_seg.write_csv(tables / "cecl_backtest_2026h1_by_segment.csv")
    backtest = {"months": B, "defaults_actual": int(bt["default_next"].sum()), "defaults_pred": float(bt["pd6"].sum()),
                "prepays_actual": int(bt["prepay_next"].sum()), "prepays_pred": float(bt["pp6"].sum())}
    backtest["default_actual_to_pred"] = backtest["defaults_actual"] / backtest["defaults_pred"]
    backtest["prepay_actual_to_pred"] = backtest["prepays_actual"] / backtest["prepays_pred"]
    log(f"backtest 2026H1: defaults {backtest['defaults_actual']} vs {backtest['defaults_pred']:.0f}; "
        f"prepays {backtest['prepays_actual']} vs {backtest['prepays_pred']:.0f}")

    # ---------------- 5. Sensitivities ----------------
    sens = []

    def add(label, accl):
        sens.append({"case": label, "acl": accl, "acl_pct_upb": accl / upb0, "vs_main": accl / weighted - 1})

    add("main", weighted)
    add("model-only (input reversion)", weighted_model)
    for a_, b_ in c["sensitivity"]["hist_loss_windows"]:
        r_ = float(hist_tbl.filter(pl.col("default_year").is_between(a_, b_))["loss_rate"].mean())
        add(f"historical loss window {a_}-{b_} ({r_:.3%}/yr)",
            sum(results[s]["weight"] * blend(projs[s], c["rs_quarters"] * 3, c["reversion_quarters"] * 3, r_ / 12.0).sum()
                for s in results))
    for wname, ws in c["sensitivity"]["weight_sets"].items():
        add(f"weights {wname}", sum(ws[s] * results[s][main_key] for s in results))
    add("no overlays (model only for delinquent/history loans)",
        sum(results[s]["weight"] * acl_of(run(c["scenarios"][s]["scale"], c["rs_quarters"], c["reversion_quarters"], None)[0],
                                          c["rs_quarters"], c["reversion_quarters"])[main_key] for s in results)
        if "--no-sens" not in sys.argv else float("nan"))
    if "--no-sens" not in sys.argv:
        for rs in c["sensitivity"]["rs_quarters"]:
            add(f"R&S {rs} quarters", sum(results[s]["weight"] * acl_of(run(c["scenarios"][s]["scale"], rs, c["reversion_quarters"])[0],
                                                                        rs, c["reversion_quarters"])[main_key] for s in results))
            log(f"sens R&S {rs}")
        for rv in c["sensitivity"]["reversion_quarters"]:
            add(f"reversion {rv} quarters", sum(results[s]["weight"] * acl_of(run(c["scenarios"][s]["scale"], c["rs_quarters"], rv)[0],
                                                                              c["rs_quarters"], rv)[main_key] for s in results))
            log(f"sens reversion {rv}")
        for ds in c["sensitivity"]["downside_scale"]:
            dn = acl_of(run(ds, c["rs_quarters"], c["reversion_quarters"])[0], c["rs_quarters"], c["reversion_quarters"])[main_key]
            add(f"downside = {ds:.0%} of severely adverse",
                sum(results[s]["weight"] * (dn if s == "downside" else results[s][main_key]) for s in results))
    pl.DataFrame(sens).write_csv(tables / "cecl_sensitivity.csv")

    # ---------------- 6. Figures ----------------
    qs = np.arange(tas // 3 - 8, tas // 3 + 20)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = {"baseline": "#333", "downside": "#c0504d", "upside": "#2e8b57"}
    for name, fq in paths_q.items():
        f = fq.filter(pl.col("q").is_in(qs)).sort("q")
        x = f["q"].to_numpy() / 4.0
        ax[0].plot(x, f["ur"], color=cols[name], label=name)
        ax[1].plot(x, f["hpi_nat"] / f.filter(pl.col("q") == tas // 3)["hpi_nat"].item() * 100, color=cols[name])
        ax[2].plot(x, f["mort_rate"], color=cols[name])
    for a, ttl in zip(ax, ["Unemployment rate (%)", "National HPI (as-of = 100)", "30Y mortgage rate (%)"]):
        a.axvspan(tas // 3 / 4 + 0.25, tas // 3 / 4 + 0.25 + c["rs_quarters"] / 4, color="#f2e6c9", alpha=0.6)
        a.axvspan(tas // 3 / 4 + 0.25 + c["rs_quarters"] / 4, tas // 3 / 4 + 0.25 + (c["rs_quarters"] + c["reversion_quarters"]) / 4,
                  color="#ddd", alpha=0.6)
        a.set_title(ttl); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("CECL scenarios: R&S period (tan), reversion (grey), then long-run", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "cecl_scenarios.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    k = np.arange(1, 121)
    for name in results:
        bl = blend(projs[name], c["rs_quarters"] * 3, c["reversion_quarters"] * 3, hist_annual / 12.0).sum(1)[:120]
        ax.plot(k, bl / 1e6, color=cols[name], label=f"{name} ({results[name]['acl_pct_upb']:.2%} of UPB)")
        ax.plot(k, projs[name].month_el.sum(1)[:120] / 1e6, ":", color=cols[name], lw=1)
    ax.axvspan(0, c["rs_quarters"] * 3, color="#f2e6c9", alpha=0.6)
    ax.axvspan(c["rs_quarters"] * 3, (c["rs_quarters"] + c["reversion_quarters"]) * 3, color="#ddd", alpha=0.6)
    ax.set_xlabel("Months after as-of date"); ax.set_ylabel("Expected loss ($M per month)")
    ax.set_title("Timing of lifetime expected loss (solid: with reversion; dotted: model only)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / "cecl_loss_timing.png", dpi=150); plt.close(fig)

    update_metrics(cfg.path("metrics"), "phase5", {
        "as_of": c["as_of"], "portfolio": {"loans": pf.height, "upb": upb0,
                                           "by_segment": seg_tbl.select("segment", "loans", "upb").to_dicts()},
        "hist_annual_loss_rate": hist_annual, "long_run": lr,
        "overlay_multipliers": overlay, "overlay_s0_actual_to_pred": ap0,
        "overlay_calibration": seg_cal.to_dicts(),
        "scenarios": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("by_segment")} for k, v in results.items()},
        "acl_weighted": weighted, "acl_weighted_pct_upb": weighted / upb0,
        "acl_weighted_model_only": weighted_model, "acl_weighted_model_only_pct_upb": weighted_model / upb0,
        "acl_by_segment": seg_tbl.to_dicts(), "acl_by_vintage_model_basis": vin_tbl.to_dicts(),
        "backtest_2026h1": backtest, "backtest_2026h1_by_segment": bt_seg.to_dicts(),
        "sensitivity": sens, "method": {"rs_quarters": c["rs_quarters"], "reversion_quarters": c["reversion_quarters"],
                                        "reversion_method": c["reversion_method"]},
    })
    log("done")


if __name__ == "__main__":
    main()
