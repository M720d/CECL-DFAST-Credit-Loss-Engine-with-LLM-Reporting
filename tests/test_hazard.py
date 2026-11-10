"""Competing-risk hazard: sampling correction, probabilities, risk set, state betas."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cecl.hazard.macro import quarterly_to_monthly, state_hpi_betas
from cecl.hazard.model import HazardModel, fit_hazard

RNG = np.random.default_rng(11)


def test_choice_based_sampling_correction_recovers_truth():
    n = 400_000
    x = RNG.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    B = np.array([[-5.0, -3.0], [0.8, -0.5]])
    e = np.exp(X @ B)
    P = e / (1 + e.sum(1, keepdims=True))
    u = RNG.random(n)
    y = np.where(u < P[:, 0], 1, np.where(u < P[:, 0] + P[:, 1], 2, 0))
    r = 0.05
    keep = (y > 0) | (RNG.random(n) < r)
    m = fit_hazard(X[keep], y[keep], np.arange(keep.sum()), ["const", "x"], r)
    np.testing.assert_allclose(m.beta, B, atol=0.12)
    # Without the correction the constants would be off by ln(1/r) ~ 3.0
    assert abs((m.beta[0, 0] - np.log(r)) - B[0, 0]) > 2.5


def test_hazards_are_valid_probabilities_and_stable():
    m = HazardModel(["const", "x"], np.array([[-5.0, -3.0], [800.0, -0.5]]), np.zeros((2, 2)),
                    np.zeros((2, 2)), 1.0, 0, (0, 0))
    X = np.column_stack([np.ones(5), np.array([-2.0, 0.0, 1.0, 2.0, 5.0])])
    h = m.hazards(X)
    assert np.isfinite(h).all() and (h >= 0).all() and (h.sum(1) <= 1 + 1e-12).all()
    assert h[-1, 0] == pytest.approx(1.0)   # huge eta -> default certain, no overflow


def test_quarterly_to_monthly_lag():
    q = pl.DataFrame({"q": np.arange(8000, 8008, dtype=np.int32), "ur": np.arange(8, dtype=float),
                      "mort_rate": 0.0, "hpi_nat": 100.0})
    m = quarterly_to_monthly(q, lag_q=1)
    # months of quarter 8001 carry quarter 8000's value
    assert m.filter(pl.col("t").is_between(8001 * 3, 8001 * 3 + 2))["ur"].to_list() == [0.0, 0.0, 0.0]
    assert m.filter(pl.col("t") == 8005 * 3)["ur_yoy"].item() == pytest.approx(4.0)


def test_state_beta_recovery():
    nq = 120
    dn = RNG.normal(0.01, 0.02, nq)
    hpi_nat = 100 * np.exp(np.cumsum(dn))
    ds = 0.002 + 1.5 * dn + RNG.normal(0, 0.002, nq)
    st = 100 * np.exp(np.cumsum(ds))
    q = np.arange(8000, 8000 + nq, dtype=np.int32)
    fed = pl.DataFrame({"q": q, "ur": 5.0, "mort_rate": 5.0, "hpi_nat": hpi_nat})
    # monthly state index: flat within quarter so the quarterly average equals st
    fm = pl.DataFrame({"geo": "XX", "t": np.repeat(q * 3, 3) + np.tile([0, 1, 2], nq),
                       "hpi": np.repeat(st, 3)}).with_columns(pl.col("t").cast(pl.Int32))
    b = state_hpi_betas(fm, fed, int(q[0]), int(q[-1])).row(0, named=True)
    assert b["beta"] == pytest.approx(1.5, abs=0.05)


def test_panel_risk_set_and_labels():
    """Default loan: months up to default; prepay loan: up to payoff; censored: drop ZB month."""
    from cecl.data.schema import PERF_FIELDS
    from cecl.hazard.panel import build_vintage_panel

    def rows(lid, delq, zb=None, term=360):
        out = []
        for i, d in enumerate(delq):
            r = {n: None for n, _ in PERF_FIELDS}
            r.update(loan_id=lid, period=202001 + i, delq=d, loan_age=i, rem_months=term - i,
                     cur_upb=1e5, cur_rate=5.0, zb_code=(zb if i == len(delq) - 1 else None))
            out.append(r)
        return out

    perf = pl.DataFrame(rows("D", ["00", "01", "02", "03"]) + rows("P", ["00", "00", "00"], zb="01")
                        + rows("C", ["00", "00", "00"], zb="96"),
                        schema=dict(PERF_FIELDS)).with_columns(vintage=pl.lit(2020, pl.Int16))
    dcfg = {"zb_prepay": ["01"], "zb_default": ["02", "03", "09", "15"], "zb_censor": ["16", "96"],
            "dpd_default_threshold": 3, "default_definition": "ex_forbearance", "covid_window": [0, 0]}
    from cecl.data.panel import loan_outcomes
    outc = loan_outcomes(perf.lazy(), dcfg)
    orig = pl.DataFrame({"loan_id": ["D", "P", "C"], "fico": [700] * 3, "cltv": [80] * 3, "ltv": [80] * 3,
                         "dti": [30] * 3, "orig_upb": [1e5] * 3, "orig_rate": [5.0] * 3, "purpose": ["P"] * 3,
                         "occupancy": ["P"] * 3, "n_borrowers": [2] * 3, "state": ["CA"] * 3,
                         "first_pay_date": [202001] * 3})
    t = np.arange(24000, 24400, dtype=np.int32)
    macro = pl.DataFrame({"t": t, "ur": 5.0, "ur_yoy": 0.0})
    pmms = pl.DataFrame({"t": t, "pmms": 4.0})
    hpi = pl.DataFrame({"geo": ["CA"] * len(t), "t": t, "hpi": 100.0})
    hcfg = {"fit_loan_share": 1.0, "valid_loan_share": 0.0, "survive_sample_rate": 1.0,
            "hpi_lag_months": 0, "incentive_lag_months": 0}
    fit, _ = build_vintage_panel(perf.lazy(), orig.lazy(), outc, macro, pmms, hpi, dcfg, hcfg, seed=1)
    g = {k: v.sort("period")["y"].to_list() for k, v in fit.partition_by("loan_id", as_dict=True).items()}
    assert g[("D",)] == [0, 0, 1]       # first month dropped (no lag), default in month 4
    assert g[("P",)] == [0, 2]
    assert g[("C",)] == [0]             # ZB 96 month removed, no event
