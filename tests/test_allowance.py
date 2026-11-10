"""CECL engine, scenario construction and reversion blending."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cecl.allowance.engine import _overlay_weight, project
from cecl.allowance.paths import scenario_quarterly
from cecl.allowance.portfolio import SEGMENTS
from cecl.lgd.ead import scheduled_balance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cecl import blend  # noqa: E402


class ConstHazard:
    def __init__(self, hd, hp):
        self.hd, self.hp = hd, hp

    def hazards(self, X):
        return np.column_stack([np.full(len(X), self.hd), np.full(len(X), self.hp)])


class ConstLGD:
    def __init__(self, v):
        self.v = v

    def lgd(self, df):
        return np.full(df.height, self.v)


HCFG = {"hpi_lag_months": 0, "incentive_lag_months": 0,
        "knots": {"age_years": [1], "fico": [700], "mtm_ltv": [1.0], "incentive": [0.0]}}
LCFG = {"hpi_lag_months": 0, "hpi_forward_months": 12}


def tiny_portfolio(seg="S0_clean", rem=24):
    return pl.DataFrame({
        "loan_id": ["A"], "cur_upb": [100_000.0], "cur_rate": [6.0], "rem_months": [rem], "loan_age": [12],
        "orig_value": [150_000.0], "hpi_0": [100.0], "hpi_geo": ["US"], "fico": [720], "cltv": [80], "dti": [30],
        "sato": [0.0], "orig_upb": [110_000.0], "purpose": ["P"], "occupancy": ["P"], "n_borrowers": [2],
        "state": ["CA"], "mi_pct": [0], "prop_type": ["SF"], "units": [1],
        "seg_idx": pl.Series([SEGMENTS.index(seg)], dtype=pl.Int8)})


def flat_path(t0, n):
    return {"t0": t0, "ur": np.full(n, 5.0), "ur_yoy": np.zeros(n), "pmms": np.full(n, 6.0),
            "hpi": {"US": np.full(n, 100.0)}}


def test_engine_matches_closed_form():
    hd, hp, lgd, H = 0.002, 0.01, 0.3, 24
    pf = tiny_portfolio(rem=H)
    t = 24300
    pr = project(pf, flat_path(t - 30, 100), ConstHazard(hd, hp), ConstLGD(lgd), HCFG, LCFG, t, H)
    S = (1 - hd - hp) ** np.arange(H)
    bal = scheduled_balance(100_000.0, 6.0, H, np.arange(H))
    assert pr.loan_pd[0] == pytest.approx((S * hd).sum())
    assert pr.loan_el[0] == pytest.approx((S * hd * lgd * bal).sum())
    assert pr.month_exposure[:, 0].sum() == pytest.approx((S * bal).sum())


def test_in_default_loans_default_in_month_one():
    pf = tiny_portfolio(seg="S4_default")
    pr = project(pf, flat_path(24270, 100), ConstHazard(0.001, 0.01), ConstLGD(0.25), HCFG, LCFG, 24300, 12)
    assert pr.loan_pd[0] == pytest.approx(1.0)
    assert pr.loan_el[0] == pytest.approx(0.25 * 100_000.0)


def test_overlay_multiplier_and_decay():
    assert _overlay_weight(12, 12, 12) == 1.0
    assert _overlay_weight(18, 12, 12) == pytest.approx(0.5)
    assert _overlay_weight(30, 12, 12) == 0.0
    pf = tiny_portfolio(seg="S2_dpd30", rem=1)
    pr = project(pf, flat_path(24270, 100), ConstHazard(0.01, 0.0), ConstLGD(1.0), HCFG, LCFG, 24300, 1,
                 overlay={"S2_dpd30": 5.0}, ocfg={"horizon_months": 12, "decay_months": 12})
    assert pr.loan_pd[0] == pytest.approx(0.05)


def test_scenario_scaling_reversion_and_floor():
    q = np.arange(8090, 8104, dtype=np.int32)
    hist = pl.DataFrame({"q": q, "ur": 4.0, "mort_rate": 6.0, "hpi_nat": 100.0 * 1.01 ** (q - q[0])})
    qs = np.arange(8104, 8117, dtype=np.int32)
    base = pl.DataFrame({"q": qs, "ur": 4.0, "mort_rate": 6.0, "hpi_nat": 115.0})
    sa = pl.DataFrame({"q": qs, "ur": 10.0, "mort_rate": 5.0, "hpi_nat": 80.0})
    lr = {"ur": 5.0, "mort_rate": 6.5, "hpi_growth_q": 0.01}
    down = scenario_quarterly(hist, base, sa, 0.5, 8, 4, lr, 3.0, 20).filter(pl.col("q") >= 8104)
    assert down["ur"][0] == pytest.approx(7.0)                       # halfway to severely adverse
    assert down["ur"][11] == pytest.approx(5.0)                      # fully reverted after R&S + 4q
    assert down["ur"][9] == pytest.approx(7.0 + 0.5 * (5.0 - 7.0))   # straight line
    up = scenario_quarterly(hist, base, sa, -0.5, 8, 4, lr, 3.0, 20).filter(pl.col("q") >= 8104)
    assert up["ur"].min() >= 3.0                                     # floor applies (4 - 3 = 1 -> 3)
    with pytest.raises(ValueError):
        scenario_quarterly(hist, base, sa, 0.0, 20, 4, lr, 3.0, 30)  # R&S longer than scenario


def test_output_reversion_blend():
    class P:
        month_el = np.ones((10, 1))
        month_exposure = np.full((10, 1), 100.0)
    out = blend(P, rs_m=2, rev_m=4, hist_monthly=0.05)
    np.testing.assert_allclose(out[:2, 0], 1.0)            # R&S: model
    np.testing.assert_allclose(out[2, 0], 0.75 * 1 + 0.25 * 5.0)
    np.testing.assert_allclose(out[6:, 0], 5.0)            # after reversion: historical rate x exposure
