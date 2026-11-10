"""WoE/IV binning, feature selection and logistic scorecard.

Sign convention (optbinning): WoE = ln(%non-default / %default), so riskier bins have
lower WoE and every coefficient in a logit of P(default) on WoE must be negative.
A positive coefficient means the variable's effect flipped after controlling for the
others (usually collinearity); such features are removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from optbinning import BinningProcess


@dataclass
class Scorecard:
    binning: BinningProcess
    features: list[str]
    result: object                      # statsmodels LogitResults
    iv: pd.DataFrame
    dropped: dict[str, str] = field(default_factory=dict)

    def woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.binning.transform(X[self.binning.variable_names], metric="woe")[self.features]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.result.predict(sm.add_constant(self.woe(X), has_constant="add")))


def fit_binning(X: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str], bcfg: dict) -> BinningProcess:
    params = {}
    for c in numeric + categorical:
        trend = bcfg["trend_overrides"].get(c, bcfg["default_trend"]) if c in numeric else "auto"
        params[c] = {"monotonic_trend": trend}
        if c in bcfg.get("min_bin_size_overrides", {}):
            # Must apply at pre-binning too, or the rare value never gets a split to begin with.
            params[c]["min_bin_size"] = bcfg["min_bin_size_overrides"][c]
            params[c]["min_prebin_size"] = bcfg["min_bin_size_overrides"][c]
    bp = BinningProcess(
        variable_names=numeric + categorical, categorical_variables=categorical,
        max_n_prebins=bcfg["max_n_prebins"], min_prebin_size=bcfg["min_prebin_size"],
        max_n_bins=bcfg["max_n_bins"], min_bin_size=bcfg["min_bin_size"],
        binning_fit_params=params,
    )
    return bp.fit(X[numeric + categorical], y)


def select_by_iv_and_corr(bp: BinningProcess, X: pd.DataFrame, iv_min: float, corr_max: float,
                          sample: int, seed: int) -> tuple[list[str], pd.DataFrame, dict]:
    summ = bp.summary()[["name", "dtype", "n_bins", "iv", "gini", "quality_score"]].copy()
    summ["iv"] = summ["iv"].astype(float)
    summ = summ.sort_values("iv", ascending=False).reset_index(drop=True)
    dropped = {r.name: f"IV {r.iv:.4f} < {iv_min}" for r in summ.itertuples() if r.iv < iv_min}
    cand = [n for n in summ["name"] if n not in dropped]
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    W = bp.transform(Xs[bp.variable_names], metric="woe")[cand]
    corr = W.corr().abs()
    keep: list[str] = []
    for c in cand:  # highest IV first; drop anything too correlated with a kept feature
        hit = [k for k in keep if corr.loc[c, k] > corr_max]
        if hit:
            dropped[c] = f"|corr| {corr.loc[c, hit[0]]:.2f} with {hit[0]} (higher IV)"
        else:
            keep.append(c)
    return keep, summ, dropped


def fit_logit(W: pd.DataFrame, y: np.ndarray, groups: np.ndarray, pvalue_max: float,
              dropped: dict) -> tuple[object, list[str]]:
    """Backward elimination on wrong sign, then p-value; clustered SEs by loan."""
    feats = list(W.columns)
    while True:
        res = sm.Logit(y, sm.add_constant(W[feats])).fit(
            disp=0, method="newton", maxiter=100,
            cov_type="cluster", cov_kwds={"groups": groups})
        coefs, pv = res.params.drop("const"), res.pvalues.drop("const")
        wrong = coefs[coefs >= 0]
        if len(wrong):
            f = wrong.index[0]
            dropped[f] = f"wrong sign after controls (coef {coefs[f]:+.3f})"
        elif pv.max() > pvalue_max:
            f = pv.idxmax()
            dropped[f] = f"p-value {pv[f]:.3g} > {pvalue_max}"
        else:
            return res, feats
        feats.remove(f)


def fit_scorecard(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, numeric: list[str],
                  categorical: list[str], cfg: dict, seed: int) -> Scorecard:
    bp = fit_binning(X, y, numeric, categorical, cfg["binning"])
    keep, iv, dropped = select_by_iv_and_corr(bp, X, cfg["iv_min"], cfg["corr_max"], 500_000, seed)
    W = bp.transform(X[bp.variable_names], metric="woe")[keep]
    res, feats = fit_logit(W, y, groups, cfg["pvalue_max"], dropped)
    return Scorecard(bp, feats, res, iv, dropped)


def points_table(sc: Scorecard, base_score: float, base_odds: float, pdo: float) -> pd.DataFrame:
    """Scaled points per bin: score = offset + factor * ln(odds of non-default).

    ln(odds_default) = a + sum_i b_i * WoE_i, so each bin gets
    points = -(b_i * WoE + a / n) * factor + offset / n, and points add up to the score.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    a, n = sc.result.params["const"], len(sc.features)
    rows = []
    for f in sc.features:
        bt = sc.binning.get_binned_variable(f).binning_table.build()
        bt = bt[bt.index != "Totals"] if "Totals" in bt.index else bt.iloc[:-1]
        b = sc.result.params[f]
        for _, r in bt.iterrows():
            if r["Count"] == 0:
                continue
            woe = float(r["WoE"])
            rows.append({"feature": f, "bin": str(r["Bin"]), "count": int(r["Count"]),
                         "event_rate": float(r["Event rate"]), "woe": woe,
                         "points": -(b * woe + a / n) * factor + offset / n})
    return pd.DataFrame(rows)


def score_from_pd(pd_: np.ndarray, base_score: float, base_odds: float, pdo: float) -> np.ndarray:
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log((1 - pd_) / pd_)
