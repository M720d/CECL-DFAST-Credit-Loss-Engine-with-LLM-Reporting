"""EAD amortization, fractional logit, two-part LGD."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from cecl.lgd.ead import scheduled_balance

RNG = np.random.default_rng(3)


def test_scheduled_balance_matches_amortization_loop():
    b0, rate, n = 200_000.0, 6.0, 360
    r = rate / 1200
    pay = b0 * r / (1 - (1 + r) ** -n)
    b = b0
    for _ in range(12):
        b = b * (1 + r) - pay
    assert scheduled_balance(b0, rate, n, 12) == pytest.approx(b, rel=1e-10)
    assert scheduled_balance(b0, rate, n, n) == pytest.approx(0.0, abs=1e-6)     # fully amortized
    assert scheduled_balance(b0, rate, 12, 24) == pytest.approx(0.0, abs=1e-6)   # k capped at n
    assert scheduled_balance(120.0, 0.0, 12, 6) == pytest.approx(60.0)            # zero-rate loan


def test_fractional_logit_recovers_mean_function():
    n = 20_000
    x = RNG.normal(size=n)
    mu = 1 / (1 + np.exp(-(-0.5 + 0.9 * x)))
    y = np.clip(RNG.beta(mu * 5, (1 - mu) * 5), 0, 1)     # fractional outcome with E[y|x] = mu
    y[RNG.random(n) < 0.05] = 0.0                           # mass at the bounds is allowed
    y[RNG.random(n) < 0.02] = 1.0
    X = sm.add_constant(x)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1")
    assert res.params[1] == pytest.approx(0.9, abs=0.12)


def test_two_part_lgd_and_resolution_rules():
    """Resolution classes and E[LGD] = (1 - p_cure) * severity on a hand-built frame."""
    from cecl.lgd.data import build_lgd_vintage
    from cecl.data.schema import PERF_FIELDS

    def rows(lid, delq, zb=None, loss=None, dai=0.0):
        out = []
        for i, d in enumerate(delq):
            last = i == len(delq) - 1
            r = {n: None for n, _ in PERF_FIELDS}
            r.update(loan_id=lid, period=202001 + i, delq=d, loan_age=12 + i, rem_months=300 - i,
                     cur_upb=0.0 if (last and zb) else 100_000.0, cur_rate=6.0, zb_code=zb if last else None,
                     actual_loss=loss if last else None, delinq_accrued_int=dai if last else None)
            out.append(r)
        return out

    perf = pl.DataFrame(rows("L", ["00", "01", "02", "03", "04", "RA"], zb="09", loss=40_000.0, dai=5_000.0)
                        + rows("C", ["00", "01", "02", "03", "00", "00"])
                        + rows("R", ["00", "01", "02", "03", "00"], zb="16")
                        + rows("U", ["00", "01", "02", "03", "04", "05"]),
                        schema=dict(PERF_FIELDS)).with_columns(vintage=pl.lit(2019, pl.Int16))
    dcfg = {"zb_prepay": ["01"], "zb_default": ["02", "03", "09", "15"], "zb_censor": ["16", "96"],
            "dpd_default_threshold": 3, "default_definition": "ex_forbearance", "covid_window": [0, 0]}
    from cecl.data.panel import loan_outcomes
    outc = loan_outcomes(perf.lazy(), dcfg).collect()
    orig = pl.DataFrame({"loan_id": list("LCRU"), "fico": [700] * 4, "dti": [30] * 4, "ltv": [80] * 4,
                         "mi_pct": [0] * 4, "orig_upb": [1e5] * 4, "state": ["CA"] * 4, "prop_type": ["SF"] * 4,
                         "occupancy": ["P"] * 4, "units": [1] * 4, "purpose": ["P"] * 4, "n_borrowers": [2] * 4,
                         "first_pay_date": [201901] * 4})
    t = np.arange(24000, 24400, dtype=np.int32)
    hpi = pl.DataFrame({"geo": ["CA"] * len(t) + ["US"] * len(t), "t": np.concatenate([t, t]), "hpi": 100.0})
    macro = pl.DataFrame({"t": t, "ur": 5.0, "ur_yoy": 0.0})
    lcfg = {"zb16_treatment": "cure", "severity_basis": "principal", "clip": [0.0, 1.0], "hpi_lag_months": 0,
            "ead_lookback_months": 1, "hpi_forward_months": 12}
    df = build_lgd_vintage(perf.lazy(), orig.lazy(), outc.lazy(), hpi, macro, dcfg, lcfg)
    res = dict(zip(df["loan_id"], df["resolution"]))
    assert res == {"L": "liquidated", "C": "cured", "R": "cured", "U": "unresolved"}
    sev = df.filter(pl.col("loan_id") == "L")["severity"].item()
    assert sev == pytest.approx((40_000 - 5_000) / 100_000)   # principal basis, UPB at default denominator
