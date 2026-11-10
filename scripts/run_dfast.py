"""Phase 6: DFAST 9-quarter credit-loss projection (baseline vs severely adverse),
crisis replay (Dec-2007 portfolio on the realized 2008-2010 path), downturn-LGD check.

Loss rate (Fed definition): cumulative 9-quarter losses / average balance over the horizon.
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
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from cecl.allowance.engine import project  # noqa: E402
from cecl.allowance.paths import long_run, monthly_path, scenario_quarterly  # noqa: E402
from cecl.allowance.portfolio import SEGMENTS, build_portfolio  # noqa: E402
from cecl.config import load_config  # noqa: E402
from cecl.hazard.macro import load_fed_quarterly, load_fmhpi_states  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.features import load_pmms  # noqa: E402
from run_cecl import to_t  # noqa: E402

T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def quarterly(proj, nq: int) -> dict:
    el = proj.month_el.sum(1)[: nq * 3].reshape(nq, 3).sum(1)
    du = proj.month_def_upb.sum(1)[: nq * 3].reshape(nq, 3).sum(1)
    ex = proj.month_exposure.sum(1)[: nq * 3]
    return {"loss": el, "defaulted_upb": du, "avg_balance": float(ex.mean()),
            "loss_rate_fed_def": float(el.sum() / ex.mean()), "lgd": float(el.sum() / du.sum())}


def realized_losses(cfg, pf: pl.DataFrame, t_start: int, months: int) -> dict:
    """Actual losses on the replay portfolio: defaults in the window x realized LGD (cure = 0)."""
    lg = pl.read_parquet(cfg.path("processed") / "lgd_dataset.parquet").select(
        "loan_id", "default_period", "resolution", "severity", "upb_default")
    d = (pf.select("loan_id", "segment").join(lg, on="loan_id", how="inner")
         .with_columns(t=(pl.col("default_period") // 100) * 12 + pl.col("default_period") % 100 - 1)
         .filter(pl.col("t").is_between(t_start + 1, t_start + months) | (pl.col("segment") == "S4_default")))
    d = d.with_columns(lgd=pl.when(pl.col("resolution") == "cured").then(0.0)
                       .when(pl.col("resolution") == "liquidated").then(pl.col("severity")).otherwise(None))
    known = d.filter(pl.col("lgd").is_not_null())
    scale = d["upb_default"].sum() / known["upb_default"].sum()        # gross up unresolved/excluded at known LGD
    d = d.with_columns(q=((pl.col("t") - t_start - 1).clip(0, None) // 3))
    by_q = (known.with_columns(q=((pl.col("t") - t_start - 1).clip(0, None) // 3))
            .group_by("q").agg(loss=(pl.col("upb_default") * pl.col("lgd")).sum() * scale,
                               def_upb=pl.col("upb_default").sum()).sort("q"))
    return {"loss": by_q["loss"].to_numpy(), "lgd": float((known["upb_default"] * known["lgd"]).sum() / known["upb_default"].sum()),
            "defaults": d.height, "resolved_share": known.height / max(d.height, 1)}


def main() -> None:
    cfg = load_config()
    df_cfg, hcfg, lcfg, c = cfg["dfast"], cfg["hazard"], cfg["lgd"], cfg["cecl"]
    raw, proc, tables, figs = cfg.path("raw_dfast"), cfg.path("processed"), cfg.path("tables"), cfg.path("figures")
    pm = load_pmms(cfg.path("raw_macro") / "MORTGAGE30US.csv").with_columns(pl.col("t").cast(pl.Int32))
    hpi = load_fmhpi_states(cfg.path("raw_macro") / "fmhpi_master_file.csv")
    fh = load_fed_quarterly(raw / "2026_Final_Historic_Domestic.csv")
    base = load_fed_quarterly(raw / "2026_Final_Supervisory_Baseline_Domestic.csv")
    sa = load_fed_quarterly(raw / "2026_Final_Supervisory_Severely_Adverse_Domestic.csv")
    betas = pl.read_parquet(proc / "hazard" / "state_hpi_betas.parquet")
    hz = pickle.load(open(proc / "models" / "hazard_prod.pkl", "rb"))
    lg = pickle.load(open(proc / "models" / "lgd_prod.pkl", "rb"))
    ov = {r["segment"]: r["multiplier"] for r in pl.read_csv(tables / "cecl_overlay_multipliers.csv").iter_rows(named=True)
          if r["segment"] != "S0_clean"}
    oc = c["overlay"]
    lr = long_run(fh, c["long_run"])
    nq = df_cfg["horizon_quarters"]
    H = nq * 3

    # ---------------- 1. Supervisory scenarios on the Dec-2025 portfolio ----------------
    tas = to_t(c["as_of"])
    pf = build_portfolio(cfg, c["as_of"], pm, hpi).with_columns(pl.col("orig_value").fill_null(pl.col("orig_upb") / 0.8))
    upb0 = float(pf["cur_upb"].sum())
    res, paths_q = {}, {}
    for name, scale in df_cfg["scenarios"].items():
        n_scn = base.height
        fq = scenario_quarterly(fh, base, sa, scale, n_scn, 4, lr, 0.0, (H + 60) // 3)
        path = monthly_path(fq, pm, hpi, betas, tas - 24, tas + H + 24, hcfg["macro_lag_quarters"], tas, "betas")
        pr = project(pf, path, hz, lg, hcfg, lcfg, tas, H, ov, oc)
        q = quarterly(pr, nq)
        seg_loss = dict(zip(SEGMENTS, pr.month_el[:H].sum(0).tolist()))
        res[name] = {**{k: v for k, v in q.items() if k not in ("loss", "defaulted_upb")},
                     "quarterly_loss": q["loss"].tolist(), "cum_loss": float(q["loss"].sum()),
                     "cum_loss_pct_initial_upb": float(q["loss"].sum() / upb0),
                     "cum_default_rate": float(pr.loan_pd.mean()), "loss_by_segment": seg_loss}
        paths_q[name] = fq
        log(f"{name}: 9q loss rate {q['loss_rate_fed_def']:.3%} (avg balance), LGD {q['lgd']:.1%}, PD {pr.loan_pd.mean():.2%}")
    sa_lr, base_lr = res["severely_adverse"]["loss_rate_fed_def"], res["baseline"]["loss_rate_fed_def"]

    # Downturn sensitivity: severely adverse with the post-2013 servicing regime switched off (D-040)
    import copy
    lg_off = copy.copy(lg)
    lg_off.lcfg = {**lg.lcfg, "servicing_regime_start_year": 9999}
    fq = scenario_quarterly(fh, base, sa, 1.0, base.height, 4, lr, 0.0, (H + 60) // 3)
    path = monthly_path(fq, pm, hpi, betas, tas - 24, tas + H + 24, hcfg["macro_lag_quarters"], tas, "betas")
    q_off = quarterly(project(pf, path, hz, lg_off, hcfg, lcfg, tas, H, ov, oc), nq)
    sa_no_servicing = {"loss_rate_fed_def": q_off["loss_rate_fed_def"], "lgd": q_off["lgd"]}
    log(f"SA without servicing regime: loss rate {q_off['loss_rate_fed_def']:.3%}, LGD {q_off['lgd']:.1%}")
    pk = sa.filter(pl.col("q") <= sa["q"].min() + nq - 1)
    hist_q = fh.filter(pl.col("q") == sa["q"].min() - 1).row(0, named=True)
    scen_summary = {"ur_peak_sa": float(pk["ur"].max()),
                    "hpi_trough_change_sa": float(pk["hpi_nat"].min() / hist_q["hpi_nat"] - 1),
                    "ur_jumpoff": float(hist_q["ur"])}

    # ---------------- 2. Crisis replay: Dec-2007 portfolio, realized 2008-2010 macro ----------------
    def equity(pf_, t_):
        """Mark-to-market LTV at the as-of date (state FMHPI, same 3-month lag as the models)."""
        h_ = hpi.filter(pl.col("t") == t_ - hcfg["hpi_lag_months"]).select(hpi_geo="geo", hpi_now="hpi")
        m_ = pf_.join(h_, on="hpi_geo", how="left").select(
            mtm=100 * pl.col("cur_upb") / (pl.col("orig_value") * pl.col("hpi_now") / pl.col("hpi_0")),
            w=pl.col("cur_upb")).drop_nulls()
        sh = lambda thr, f=1.0: float(m_.filter(pl.col("mtm") / f > thr)["w"].sum() / m_["w"].sum())  # noqa: E731
        return {"mtm_ltv_upb_wtd": float((m_["mtm"] * m_["w"]).sum() / m_["w"].sum()),
                "share_upb_mtm_gt80": sh(80), "share_upb_mtm_gt100_after_30pct_hpi_drop": sh(100, 0.7)}

    rp = df_cfg["replay"]["as_of"]
    trp = to_t(rp)
    pf07 = build_portfolio(cfg, rp, pm, hpi).filter(pl.col("orig_value").is_not_null())
    path07 = monthly_path(fh, pm, hpi, betas, trp - 24, trp + H + 24, hcfg["macro_lag_quarters"], trp + H + 24, "actual")
    pr07 = project(pf07, path07, hz, lg, hcfg, lcfg, trp, H, ov, oc)
    q07 = quarterly(pr07, nq)
    act07 = realized_losses(cfg, pf07, trp, H)
    upb07 = float(pf07["cur_upb"].sum())
    act_q = np.zeros(nq)
    act_q[: len(act07["loss"])] = act07["loss"][:nq]
    replay = {"as_of": rp, "loans": pf07.height, "upb": upb07,
              "pred_cum_loss": float(q07["loss"].sum()), "actual_cum_loss": float(act_q.sum()),
              "pred_loss_rate_fed_def": q07["loss_rate_fed_def"], "actual_loss_rate_fed_def": float(act_q.sum() / q07["avg_balance"]),
              "pred_lgd": q07["lgd"], "actual_lgd": act07["lgd"], "actual_resolved_share": act07["resolved_share"],
              "pred_quarterly": q07["loss"].tolist(), "actual_quarterly": act_q.tolist()}
    replay["actual_to_pred"] = replay["actual_cum_loss"] / replay["pred_cum_loss"]
    rh = fh.filter(pl.col("q").is_between(trp // 3, trp // 3 + nq))
    replay["ur_peak"] = float(rh["ur"].max())
    replay["hpi_change"] = float(rh["hpi_nat"].min() / rh["hpi_nat"][0] - 1)
    log(f"replay 2008-10: predicted loss rate {replay['pred_loss_rate_fed_def']:.3%} vs actual {replay['actual_loss_rate_fed_def']:.3%}; "
        f"LGD {replay['pred_lgd']:.1%} vs {replay['actual_lgd']:.1%}")

    eq = {"dec2025": equity(pf, tas), "dec2007": equity(pf07, trp)}
    log(f"equity: {eq}")

    # ---------------- 3. Downturn LGD check ----------------
    p4 = pl.read_csv(tables / "lgd_backtest_by_default_year.csv").filter(pl.col("default_year").is_between(2008, 2010))
    downturn = {"model_lgd_severely_adverse": res["severely_adverse"]["lgd"],
                "model_lgd_baseline": res["baseline"]["lgd"],
                "realized_lgd_2008_2010_upb_wtd": float((p4["lgd_actual_upb_weighted"] * p4["defaults_resolved"]).sum() / p4["defaults_resolved"].sum()),
                "replay_pred_lgd": replay["pred_lgd"], "replay_actual_lgd": replay["actual_lgd"],
                "sa_without_servicing_regime": sa_no_servicing, "equity": eq}

    # ---------------- Tables, figures, metrics ----------------
    qlab = [f"{(tas // 3 + i + 1) // 4}Q{(tas // 3 + i + 1) % 4 + 1}" for i in range(nq)]
    pl.DataFrame({"quarter": qlab, **{f"loss_{k}": v["quarterly_loss"] for k, v in res.items()},
                  "replay_pred": replay["pred_quarterly"], "replay_actual": replay["actual_quarterly"]}
                 ).write_csv(tables / "dfast_quarterly_losses.csv")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(1, nq + 1)
    for name, col in (("baseline", "#333"), ("severely_adverse", "#c0504d")):
        ax[0].bar(x + (0.2 if name != "baseline" else -0.2), np.array(res[name]["quarterly_loss"]) / upb0 * 100,
                  width=0.4, color=col, label=f"{name} (9q {res[name]['loss_rate_fed_def']:.2%})")
    ax[0].set_xticks(x, qlab, rotation=45, fontsize=7)
    ax[0].set_title("Quarterly losses, % of initial UPB"); ax[0].legend(fontsize=8)
    for name, col in (("baseline", "#333"), ("severely_adverse", "#c0504d")):
        cum = np.cumsum(res[name]["quarterly_loss"]) / res[name]["avg_balance"] * 100
        ax[1].plot(x, cum, "o-", color=col, label=name)
    ax[1].axhline(df_cfg["fed_benchmark"]["first_lien_loss_rate_sa"] * 100, ls="--", color="#1f4e79",
                  label="Fed 2026 SA, all banks (first lien)")
    ax[1].set_xticks(x, qlab, rotation=45, fontsize=7)
    ax[1].set_title("Cumulative loss rate (Fed definition, %)"); ax[1].legend(fontsize=8)
    ax[2].plot(x, np.cumsum(replay["actual_quarterly"]) / q07["avg_balance"] * 100, "o-", color="#333", label="Actual")
    ax[2].plot(x, np.cumsum(replay["pred_quarterly"]) / q07["avg_balance"] * 100, "s--", color="#c0504d", label="Engine on realized macro")
    ax[2].set_xticks(x, [f"Q{i}" for i in x])
    ax[2].set_title(f"Crisis replay: Dec-2007 portfolio, 2008Q1-2010Q1"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / "dfast_losses.png", dpi=150); plt.close(fig)

    update_metrics(cfg.path("metrics"), "phase6", {
        "portfolio": {"as_of": c["as_of"], "loans": pf.height, "upb": upb0},
        "scenario_summary": scen_summary, "results": res,
        "sa_minus_baseline_loss_rate": sa_lr - base_lr, "sa_to_baseline_ratio": sa_lr / base_lr,
        "fed_benchmark": df_cfg["fed_benchmark"], "sa_vs_fed_benchmark_ratio": sa_lr / df_cfg["fed_benchmark"]["first_lien_loss_rate_sa"],
        "crisis_replay": replay, "downturn_lgd": downturn, "quarters": qlab,
    })
    log("done")


if __name__ == "__main__":
    main()
