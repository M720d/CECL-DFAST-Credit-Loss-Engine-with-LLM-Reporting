"""Phase 2 add-on: logistic recalibration of the scorecard PD on a recent window (D-020).

logit(PD_recal) = a + b * logit(PD_scorecard), fitted on OOT snapshots in the calibration
window and evaluated only on later snapshots. Rank order (Gini) is unchanged; only levels move.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from cecl.config import load_config  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402
from cecl.pd.features import CATEGORICAL, NUMERIC  # noqa: E402
from cecl.pd.validation import calibration_table, gini  # noqa: E402


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main() -> None:
    cfg = load_config()
    pcfg = cfg["pd"]
    rc = pcfg["recalibration"]
    with open(cfg.path("processed") / "models" / "pd_scorecard.pkl", "rb") as f:
        sc = pickle.load(f)
    lo, hi = pcfg["oot_vintages"]
    df = (pl.read_parquet(cfg.path("processed") / "pd_features.parquet")
          .filter(pl.col("vintage").is_between(lo, hi))
          .with_columns([pl.col(c).fill_null("NA").cast(pl.Categorical) for c in CATEGORICAL]
                        + [pl.col(c).cast(pl.Float64) for c in NUMERIC])
          .to_pandas())
    # Forbearance status at the snapshot (not a model feature; used for diagnostics only).
    fb = pl.concat([pl.read_parquet(cfg.path("processed") / "snapshots" / f"snapshots_{v}.parquet")
                    .select("loan_id", "snapshot_period", "forbearance_row") for v in range(lo, hi + 1)])
    df = df.merge(fb.to_pandas(), on=["loan_id", "snapshot_period"], how="left")
    p = sc.predict(df)
    y = df["target"].to_numpy()
    snap = df["snapshot_period"].to_numpy()
    cal = (snap >= rc["calib_snapshots"][0]) & (snap <= rc["calib_snapshots"][1])
    test = snap >= rc["test_snapshot_min"]

    res = sm.Logit(y[cal], sm.add_constant(logit(p[cal]))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["loan_id"].astype("category").cat.codes.to_numpy()[cal]})
    a, b = float(res.params[0]), float(res.params[1])
    p_new = 1 / (1 + np.exp(-(a + b * logit(p))))

    before = calibration_table(y[test], p[test], pcfg["calib_bins"])
    after = calibration_table(y[test], p_new[test], pcfg["calib_bins"])
    before.to_csv(cfg.path("tables") / "pd_recal_test_before.csv", index=False)
    after.to_csv(cfg.path("tables") / "pd_recal_test_after.csv", index=False)

    def summary(t, yy, pp):
        return {"actual_to_pred": float(yy.mean() / pp.mean()), "gini": gini(yy, pp),
                "decile_ratio_min": float(t["ratio_actual_to_pred"].min()),
                "decile_ratio_max": float(t["ratio_actual_to_pred"].max()),
                "mean_abs_log_ratio": float(np.mean(np.abs(np.log(t["ratio_actual_to_pred"]))))}

    out = {"intercept_a": a, "slope_b": b, "se": [float(x) for x in res.bse],
           "calib_n": int(cal.sum()), "calib_defaults": int(y[cal].sum()),
           "test_n": int(test.sum()), "test_defaults": int(y[test].sum()),
           "test_before": summary(before, y[test], p[test]),
           "test_after": summary(after, y[test], p_new[test]),
           "calibration_test_after": after.to_dict("records")}
    update_metrics(cfg.path("metrics"), "phase2_recalibration", out)

    # Diagnostics: where the raw scorecard is miscalibrated out of time.
    d = pl.DataFrame({"y": y, "p": p, "year": snap // 100, "delq": df["delq_num"].to_numpy(),
                      "forb": df["forbearance_row"].fillna(False).to_numpy().astype(bool)})
    d = d.with_columns(delq_bucket=pl.when(pl.col("delq") >= 3).then(pl.lit("3+"))
                       .otherwise(pl.col("delq").cast(pl.Int64).cast(pl.Utf8)))
    seg = (d.group_by("delq_bucket", "forb").agg(n=pl.len(), pred=pl.col("p").mean(), actual=pl.col("y").mean())
           .sort("delq_bucket", "forb"))
    yr = d.group_by("year").agg(n=pl.len(), pred=pl.col("p").mean(), actual=pl.col("y").mean()).sort("year")
    seg.write_csv(cfg.path("tables") / "pd_oot_calibration_by_segment.csv")
    yr.write_csv(cfg.path("tables") / "pd_oot_calibration_by_year.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    tb = cfg.path("tables")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for f_, lab, col in (("pd_calibration_oot.csv", "Scorecard (raw)", "#1f4e79"),
                         ("pd_calibration_oot_lgbm.csv", "LightGBM (raw)", "#c0504d")):
        t = pd.read_csv(tb / f_)
        ax[0].loglog(t["pd_pred"] * 100, t["dr_actual"] * 100, "o-", color=col, label=lab)
    ax[0].set_title("OOT 2015-2019, by decile (log scale)")
    ax[1].loglog(before["pd_pred"] * 100, before["dr_actual"] * 100, "o-", color="#1f4e79", label="Scorecard raw")
    ax[1].loglog(after["pd_pred"] * 100, after["dr_actual"] * 100, "s-", color="#2e8b57", label="Recalibrated")
    ax[1].set_title("Test: snapshots 2019+ (recalibrated on 2015-2017)")
    for a_ in ax:
        lims = [0.01, 10]
        a_.plot(lims, lims, ":", color="grey"); a_.set_xlim(lims); a_.set_ylim(lims)
        a_.set_xlabel("Predicted PD (%)"); a_.set_ylabel("Actual default rate (%)"); a_.legend(); a_.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(cfg.path("figures") / "pd_calibration_log.png", dpi=150); plt.close(fig)

    update_metrics(cfg.path("metrics"), "phase2_diagnostics",
                   {"oot_by_segment": seg.to_dicts(), "oot_by_snapshot_year": yr.to_dicts()})
    print({k: v for k, v in out.items() if k != "calibration_test_after"})


if __name__ == "__main__":
    main()
