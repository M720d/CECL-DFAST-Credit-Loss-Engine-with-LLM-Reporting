"""Phase 2: 12-month PD scorecard (WoE logistic) + LightGBM challenger.

Samples (D-008):
  fit      vintages 1999-2014, 80% of loans          -> binning, selection, coefficients
  holdout  vintages 1999-2014, other 20% of loans    -> in-time check, LightGBM early stopping
  oot      vintages 2015-2019                        -> out-of-time validation (never touched in fitting)
Writes metrics.json['phase2'], results/tables/pd_*.csv, results/figures/pd_*.png,
and pickled models under data/processed/models/.
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
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cecl.config import load_config  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.features import CATEGORICAL, NUMERIC, build_features, load_hpi, load_pmms  # noqa: E402
from cecl.pd.scorecard import fit_scorecard, points_table, score_from_pd  # noqa: E402
from cecl.pd.validation import (calibration_table, gini, gini_bootstrap, ks,  # noqa: E402
                                psi_categorical, psi_quantile)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load_features(cfg) -> pd.DataFrame:
    pcfg = cfg["pd"]
    out = cfg.path("processed") / "pd_features.parquet"
    lo, hi = pcfg["train_vintages"][0], pcfg["oot_vintages"][1]
    if not out.exists() or "--rebuild" in sys.argv:
        pm = load_pmms(cfg.path("raw_macro") / "MORTGAGE30US.csv")
        hp = load_hpi(cfg.path("raw_macro") / "fmhpi_master_file.csv")
        frames = []
        for y in range(lo, hi + 1):
            snaps = pl.scan_parquet(cfg.path("processed") / "snapshots" / f"snapshots_{y}.parquet")
            orig = pl.scan_parquet(cfg.path("interim") / "freddie" / f"orig_{y}.parquet")
            frames.append(build_features(snaps, orig, pm, hp, pcfg["hpi_lag_months"]).collect())
        pl.concat(frames).write_parquet(out)
    df = pl.read_parquet(out).with_columns(
        [pl.col(c).fill_null("NA").cast(pl.Categorical) for c in CATEGORICAL]
        + [pl.col(c).cast(pl.Float64) for c in NUMERIC]
    ).to_pandas()
    return df


def split(df: pd.DataFrame, cfg, seed: int) -> dict[str, pd.DataFrame]:
    pcfg = cfg["pd"]
    tr_lo, tr_hi = pcfg["train_vintages"]
    oo_lo, oo_hi = pcfg["oot_vintages"]
    # Deterministic loan-level assignment: all snapshots of a loan land in the same set.
    u = pd.util.hash_pandas_object(df["loan_id"].astype(str) + str(seed), index=False).to_numpy() % 10_000
    in_train = df["vintage"].between(tr_lo, tr_hi).to_numpy()
    hold = u < pcfg["holdout_share"] * 10_000
    return {"fit": df[in_train & ~hold].reset_index(drop=True),
            "holdout": df[in_train & hold].reset_index(drop=True),
            "oot": df[df["vintage"].between(oo_lo, oo_hi)].reset_index(drop=True)}


def disc(y, p, name: str) -> dict:
    return {"sample": name, "n": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
            "mean_pd": float(p.mean()), "gini": gini(y, p), "ks": ks(y, p)}


def fit_lgbm(fit: pd.DataFrame, hold: pd.DataFrame, feats: list[str], lcfg: dict, seed: int):
    import lightgbm as lgb
    params = dict(lcfg["params"], seed=seed, deterministic=True, force_row_wise=True,
                  monotone_constraints=[lcfg["monotone"].get(f, 0) for f in feats])
    dtr = lgb.Dataset(fit[feats], fit["target"], categorical_feature=[f for f in feats if f in CATEGORICAL])
    dva = lgb.Dataset(hold[feats], hold["target"], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=lcfg["num_boost_round"], valid_sets=[dva],
                     callbacks=[lgb.early_stopping(lcfg["early_stopping_rounds"], verbose=False)])


def main() -> None:
    cfg = load_config()
    pcfg, seed = cfg["pd"], cfg["seed"]
    tables, figs = cfg.path("tables"), cfg.path("figures")
    models = cfg.path("processed") / "models"
    models.mkdir(parents=True, exist_ok=True)

    df = load_features(cfg)
    S = split(df, cfg, seed)
    del df
    log("samples: " + ", ".join(f"{k}={len(v):,} ({int(v['target'].sum()):,} def)" for k, v in S.items()))
    groups_fit = S["fit"]["loan_id"].astype("category").cat.codes.to_numpy()

    # Training-only stability screen (early vs late training vintages). Uses no OOT data.
    sc_cfg = pcfg["stability_screen"]
    early = S["fit"][S["fit"]["vintage"].between(*sc_cfg["early_vintages"])]
    late = S["fit"][S["fit"]["vintage"].between(*sc_cfg["late_vintages"])]
    screen = {}
    for c in NUMERIC + CATEGORICAL:
        if c in CATEGORICAL:
            screen[c] = psi_categorical(early[c].astype(str), late[c].astype(str))
        else:
            e, l_ = early[c].dropna().to_numpy(), late[c].dropna().to_numpy()
            screen[c] = psi_quantile(e, l_)
    pd.Series(screen, name="psi_early_vs_late_train").sort_values(ascending=False).to_csv(
        tables / "pd_stability_screen_train.csv")
    excl = pcfg["exclude_features"]
    num_feats = [c for c in NUMERIC if c not in excl]
    cat_feats = [c for c in CATEGORICAL if c not in excl]
    log(f"excluded {excl}; screen: " + ", ".join(f"{c}={screen[c]:.2f}" for c in excl))

    # ---------------- Champion: WoE logistic scorecard ----------------
    sc = fit_scorecard(S["fit"], S["fit"]["target"].to_numpy(), groups_fit, num_feats, cat_feats, pcfg, seed)
    log(f"scorecard features ({len(sc.features)}): {sc.features}")
    with open(models / "pd_scorecard.pkl", "wb") as f:
        pickle.dump(sc, f)

    P = {k: sc.predict(v) for k, v in S.items()}
    Y = {k: v["target"].to_numpy() for k, v in S.items()}
    res_sc = [disc(Y[k], P[k], k) for k in S]
    ci = gini_bootstrap(Y["oot"], P["oot"], S["oot"]["loan_id"].to_numpy(), pcfg["bootstrap_reps"], seed)
    log(f"scorecard OOT gini={res_sc[2]['gini']:.4f} CI={ci}")

    # Coefficients, IV, VIF
    W = sc.woe(S["fit"].sample(200_000, random_state=seed))
    vif = {f: float(variance_inflation_factor(np.column_stack([np.ones(len(W)), W.to_numpy()]), i + 1))
           for i, f in enumerate(sc.features)}
    iv = sc.iv.set_index("name")["iv"]
    coef = pd.DataFrame({"feature": sc.features,
                         "coef": sc.result.params[sc.features].values,
                         "se_clustered": sc.result.bse[sc.features].values,
                         "p_value": sc.result.pvalues[sc.features].values,
                         "iv": [float(iv[f]) for f in sc.features],
                         "vif": [vif[f] for f in sc.features]})
    coef.to_csv(tables / "pd_scorecard_coefficients.csv", index=False)
    sc.iv.to_csv(tables / "pd_iv_all_features.csv", index=False)
    pts = points_table(sc, **pcfg["points"])
    pts.to_csv(tables / "pd_scorecard_points.csv", index=False)

    cal = {k: calibration_table(Y[k], P[k], pcfg["calib_bins"]) for k in ("holdout", "oot")}
    for k, t in cal.items():
        t.to_csv(tables / f"pd_calibration_{k}.csv", index=False)

    # Stability: score PSI and per-feature CSI (on WoE bins), development vs OOT
    score = {k: score_from_pd(P[k], **pcfg["points"]) for k in ("fit", "oot")}
    psi_score = psi_quantile(score["fit"], score["oot"])
    bins_fit = sc.binning.transform(S["fit"][sc.binning.variable_names], metric="bins")
    bins_oot = sc.binning.transform(S["oot"][sc.binning.variable_names], metric="bins")
    csi = {f: psi_categorical(bins_fit[f].astype(str), bins_oot[f].astype(str)) for f in sc.features}

    by_vintage = [{"vintage": int(v), **disc(Y["oot"][m], P["oot"][m], str(v))}
                  for v in sorted(S["oot"]["vintage"].unique()) for m in [(S["oot"]["vintage"] == v).to_numpy()]]
    yr = (S["oot"]["snapshot_period"] // 100).to_numpy()
    by_year = [{"snapshot_year": int(y_), **disc(Y["oot"][yr == y_], P["oot"][yr == y_], str(y_))}
               for y_ in np.unique(yr) if Y["oot"][yr == y_].sum() >= 30]

    # Sensitivity: strict (spec) default label on the OOT sample
    m = S["oot"]["eligible_spec"].to_numpy()
    spec = disc(S["oot"]["target_spec"].to_numpy()[m], P["oot"][m], "oot_spec_label")

    # ---------------- Challenger: LightGBM + SHAP ----------------
    feats = num_feats + cat_feats      # same feature universe as the scorecard
    gbm = fit_lgbm(S["fit"], S["holdout"], feats, pcfg["lgbm"], seed)
    gbm.save_model(str(models / "pd_lgbm.txt"))
    G = {k: gbm.predict(v[feats], num_iteration=gbm.best_iteration) for k, v in S.items()}
    res_gbm = [disc(Y[k], G[k], k) for k in S]
    ci_g = gini_bootstrap(Y["oot"], G["oot"], S["oot"]["loan_id"].to_numpy(), pcfg["bootstrap_reps"], seed)
    cal_g = calibration_table(Y["oot"], G["oot"], pcfg["calib_bins"])
    cal_g.to_csv(tables / "pd_calibration_oot_lgbm.csv", index=False)
    log(f"lgbm best_iter={gbm.best_iteration} OOT gini={res_gbm[2]['gini']:.4f}")

    import shap
    xs = S["oot"][feats].sample(pcfg["shap_sample"], random_state=seed)
    sv = shap.TreeExplainer(gbm).shap_values(xs)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_rank = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    shap_rank.rename("mean_abs_shap").to_csv(tables / "pd_lgbm_shap_importance.csv")
    plt.figure()
    shap.summary_plot(sv, xs, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(figs / "pd_lgbm_shap_summary.png", dpi=150); plt.close("all")

    # ---------------- Robustness ----------------
    rob = {}
    ccfg = pcfg["calendar_split"]
    tr = pd.concat([S["fit"], S["holdout"]])
    cal_tr = tr[tr["snapshot_period"] <= ccfg["train_snapshot_max"]].reset_index(drop=True)
    cal_te = S["oot"][S["oot"]["snapshot_period"] >= ccfg["test_snapshot_min"]].reset_index(drop=True)
    sc_cal = fit_scorecard(cal_tr, cal_tr["target"].to_numpy(),
                           cal_tr["loan_id"].astype("category").cat.codes.to_numpy(),
                           num_feats, cat_feats, pcfg, seed)
    rob["calendar_split"] = {"train_n": len(cal_tr), "features": sc_cal.features,
                             **disc(cal_te["target"].to_numpy(), sc_cal.predict(cal_te), "calendar_test")}
    del tr, cal_tr
    log(f"calendar split gini={rob['calendar_split']['gini']:.4f}")

    orig_num = [c for c in num_feats if c not in pcfg["behavioral_features"]]
    orig_cat = [c for c in cat_feats if c not in pcfg["behavioral_features"]]
    sc_o = fit_scorecard(S["fit"], Y["fit"], groups_fit, orig_num, orig_cat, pcfg, seed)
    rob["origination_only"] = {"features": sc_o.features,
                               **disc(Y["oot"], sc_o.predict(S["oot"]), "oot_origination_only")}
    a0 = (S["oot"]["delq_num"] == 0).to_numpy()
    rob["oot_current_loans_only"] = disc(Y["oot"][a0], P["oot"][a0], "oot_current_only")
    log(f"origination-only gini={rob['origination_only']['gini']:.4f}")

    # ---------------- Figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, ls in (("holdout", "--"), ("oot", "-")):
        for name, pr, col in (("Scorecard", P, "#1f4e79"), ("LightGBM", G, "#c0504d")):
            fpr, tpr, _ = roc_curve(Y[k], pr[k])
            ax[0].plot(fpr, tpr, ls, color=col, lw=1.4, label=f"{name} {k} (Gini {gini(Y[k], pr[k]):.3f})")
    ax[0].plot([0, 1], [0, 1], ":", color="grey"); ax[0].set_title("ROC")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(fontsize=8)
    for t, name, col in ((cal["oot"], "Scorecard", "#1f4e79"), (cal_g, "LightGBM", "#c0504d")):
        ax[1].plot(t["pd_pred"] * 100, t["dr_actual"] * 100, "o-", color=col, label=name)
    mx = max(cal["oot"]["pd_pred"].max(), cal["oot"]["dr_actual"].max(), cal_g["pd_pred"].max()) * 100
    ax[1].plot([0, mx], [0, mx], ":", color="grey")
    ax[1].set_title("Out-of-time calibration by decile"); ax[1].set_xlabel("Predicted PD (%)")
    ax[1].set_ylabel("Actual default rate (%)"); ax[1].legend()
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / "pd_roc_calibration.png", dpi=150); plt.close(fig)

    # ---------------- Metrics ----------------
    update_metrics(cfg.path("metrics"), "phase2", {
        "excluded_features": excl,
        "stability_screen_train_psi": screen,
        "samples": {k: {"n": len(v), "defaults": int(v["target"].sum()), "loans": int(v["loan_id"].nunique())}
                    for k, v in S.items()},
        "scorecard": {
            "features": sc.features, "n_features": len(sc.features),
            "dropped": sc.dropped, "intercept": float(sc.result.params["const"]),
            "discrimination": res_sc, "oot_gini_ci95": list(ci),
            "calibration_oot": cal["oot"].to_dict("records"),
            "calibration_holdout": cal["holdout"].to_dict("records"),
            "oot_actual_to_pred": float(Y["oot"].mean() / P["oot"].mean()),
            "holdout_actual_to_pred": float(Y["holdout"].mean() / P["holdout"].mean()),
            "psi_score_fit_vs_oot": psi_score, "csi_fit_vs_oot": csi,
            "max_vif": max(vif.values()), "oot_by_vintage": by_vintage, "oot_by_snapshot_year": by_year,
            "oot_spec_label": spec,
        },
        "lgbm": {"best_iteration": int(gbm.best_iteration), "discrimination": res_gbm,
                 "oot_gini_ci95": list(ci_g), "calibration_oot": cal_g.to_dict("records"),
                 "oot_actual_to_pred": float(Y["oot"].mean() / G["oot"].mean()),
                 "shap_top10": shap_rank.head(10).to_dict()},
        "gini_uplift_lgbm_minus_scorecard_oot": res_gbm[2]["gini"] - res_sc[2]["gini"],
        "robustness": rob,
    })
    log("done")


if __name__ == "__main__":
    main()
