"""PD validation metrics and scorecard mechanics on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from cecl.pd.scorecard import fit_scorecard, points_table, score_from_pd
from cecl.pd.validation import calibration_table, gini, ks, psi_quantile

RNG = np.random.default_rng(7)


def synth(n=40_000):
    x1 = RNG.normal(size=n)                       # risk increases with x1
    x2 = RNG.normal(size=n)                       # risk decreases with x2
    cat = RNG.choice(["A", "B", "C"], size=n)
    noise = RNG.normal(size=n)                    # no signal
    lo = -4 + 1.0 * x1 - 0.7 * x2 + np.where(cat == "C", 0.8, 0.0)
    y = (RNG.random(n) < 1 / (1 + np.exp(-lo))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise, "cat": cat})
    return X, y


def test_gini_and_ks_match_references():
    y = RNG.integers(0, 2, 5000)
    p = np.clip(y * 0.3 + RNG.random(5000), 0, None)
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)
    assert ks(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic, abs=1e-9)


def test_psi_zero_for_same_and_positive_for_shift():
    a = RNG.normal(size=50_000)
    assert psi_quantile(a, RNG.normal(size=50_000)) < 0.01
    assert psi_quantile(a, RNG.normal(0.5, 1, size=50_000)) > 0.1


def test_calibration_table_totals():
    y = RNG.integers(0, 2, 1000)
    p = RNG.random(1000)
    t = calibration_table(y, p, 10)
    assert t["n"].sum() == 1000 and t["defaults"].sum() == y.sum()
    assert t["pd_pred"].is_monotonic_increasing


@pytest.fixture(scope="module")
def card():
    X, y = synth()
    cfg = {"binning": {"max_n_prebins": 20, "min_prebin_size": 0.05, "max_n_bins": 6, "min_bin_size": 0.05,
                       "default_trend": "auto_asc_desc", "trend_overrides": {}},
           "iv_min": 0.02, "corr_max": 0.7, "pvalue_max": 0.01}
    sc = fit_scorecard(X, y, np.arange(len(y)), ["x1", "x2", "noise"], ["cat"], cfg, seed=1)
    return sc, X, y


def test_noise_feature_dropped_and_signal_kept(card):
    sc, _, _ = card
    assert "noise" not in sc.features
    assert {"x1", "x2"} <= set(sc.features)


def test_coefficients_negative_on_woe(card):
    """optbinning WoE = ln(good%/bad%): every coefficient must be negative."""
    sc, _, _ = card
    assert (sc.result.params[sc.features] < 0).all()


def test_points_sum_to_score(card):
    sc, X, _ = card
    base = {"base_score": 600, "base_odds": 50, "pdo": 20}
    pts = points_table(sc, **base)
    xs = X.head(200)
    bins = sc.binning.transform(xs[sc.binning.variable_names], metric="bins")
    total = np.zeros(len(xs))
    for f in sc.features:
        lookup = pts[pts["feature"] == f].set_index("bin")["points"]
        total += bins[f].astype(str).map(lookup).to_numpy()
    expected = score_from_pd(sc.predict(xs), **base)
    np.testing.assert_allclose(total, expected, atol=1e-6)


def test_pdo_doubles_odds():
    pd_a = np.array([0.02])
    odds_a = (1 - pd_a) / pd_a
    pd_b = 1 / (1 + 2 * odds_a)                    # odds of non-default doubled
    s = score_from_pd(np.concatenate([pd_a, pd_b]), 600, 50, 20)
    assert s[1] - s[0] == pytest.approx(20)


def test_rare_high_risk_state_gets_own_bin_with_override():
    """Regression for the Phase 2 v1 bug: a 1%-share, high-risk value was merged away."""
    from cecl.pd.scorecard import fit_binning
    n = 50_000
    d = np.where(RNG.random(n) < 0.01, 1.0, 0.0)               # 1% of rows are 'delinquent'
    y = (RNG.random(n) < np.where(d == 1, 0.30, 0.01)).astype(int)
    X = pd.DataFrame({"d": d})
    # Same pre-binning floor as config.yaml: the first fix missed it and this test now covers it.
    base = {"max_n_prebins": 20, "min_prebin_size": 0.02, "max_n_bins": 8, "min_bin_size": 0.03,
            "default_trend": "auto_asc_desc", "trend_overrides": {}}
    iv_without = fit_binning(X, y, ["d"], [], base).summary()["iv"].astype(float).iloc[0]
    iv_with = fit_binning(X, y, ["d"], [], base | {"min_bin_size_overrides": {"d": 0.001}}
                          ).summary()["iv"].astype(float).iloc[0]
    assert iv_without < 0.01 < 0.2 < iv_with
