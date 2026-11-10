"""Monthly loan-level projection: survival x hazard x LGD x EAD.

For loan i in month k (t = as-of + k):
  b_{k-1}   scheduled balance at the start of month k (EAD, D-032)
  h_d, h_p  default / prepay hazards from the production hazard model, with overlay
            multiplier m_seg(k) on h_d for loans delinquent or with history at the as-of date
  md_k      = S_{k-1} * h_d           marginal (unconditional) default probability
  S_k       = S_{k-1} * (1 - h_d - h_p)
  EL_k      = md_k * LGD_k * b_{k-1} * ead_factor
LGD_k uses the mark-to-market LTV in month k and the 12-month forward HPI from the path.
Loans in default at the as-of date (S4) default in month 1 with certainty.
Undiscounted (non-DCF method, D-029).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from cecl.allowance.portfolio import SEGMENTS
from cecl.hazard.model import design
from cecl.lgd.ead import scheduled_balance

HAZ_STATIC = ["fico", "cltv", "dti", "sato", "orig_upb", "purpose", "occupancy", "n_borrowers"]
LGD_STATIC = ["fico", "dti", "state", "occupancy", "purpose", "mi_pct", "prop_type", "units", "n_borrowers"]


@dataclass
class Projection:
    loan_pd: np.ndarray            # lifetime cumulative default probability
    loan_prepay: np.ndarray
    loan_el: np.ndarray            # lifetime expected loss ($)
    month_el: np.ndarray           # (H, n_seg) model expected loss by month and segment
    month_def_upb: np.ndarray      # (H, n_seg) expected defaulted UPB
    month_exposure: np.ndarray     # (H, n_seg) surviving balance at the start of the month
    month_pd: np.ndarray           # (H,) expected defaults (count)
    month_prepay: np.ndarray       # (H,) expected prepayments (count)


def _overlay_weight(k: np.ndarray | int, horizon: int, decay: int) -> float:
    if k <= horizon:
        return 1.0
    if decay <= 0:
        return 0.0
    return max(0.0, 1.0 - (k - horizon) / decay)


def project(pf: pl.DataFrame, path: dict, hazard, lgd_model, hcfg: dict, lcfg: dict, as_of_t: int,
            horizon: int, overlay: dict[str, float] | None = None, ocfg: dict | None = None,
            pd_only: bool = False, ead_factor: float = 1.0) -> Projection:
    n = pf.height
    nseg = len(SEGMENTS)
    seg = pf["seg_idx"].to_numpy().astype(int)
    upb0 = pf["cur_upb"].to_numpy().astype(float)
    rate = pf["cur_rate"].to_numpy().astype(float)
    rem0 = pf["rem_months"].to_numpy().astype(float)
    age0 = pf["loan_age"].to_numpy().astype(float)
    ov = pf["orig_value"].to_numpy().astype(float)
    hpi0 = pf["hpi_0"].to_numpy().astype(float)
    in_default = seg == SEGMENTS.index("S4_default")
    mult = np.ones(nseg)
    if overlay:
        for s, v in overlay.items():
            mult[SEGMENTS.index(s)] = v

    geos = sorted(path["hpi"])
    gidx = pf["hpi_geo"].replace_strict({g: i for i, g in enumerate(geos)}, default=geos.index("US")).to_numpy()
    hpi_mat = np.vstack([path["hpi"][g] for g in geos])
    t0 = path["t0"]
    hl, il = hcfg["hpi_lag_months"], hcfg["incentive_lag_months"]
    lhl, fwd = lcfg["hpi_lag_months"], lcfg["hpi_forward_months"]

    hz_static = pf.select(HAZ_STATIC)
    lg_static = pf.select(LGD_STATIC)
    S = np.ones(n)
    loan_pd, loan_pp, loan_el = np.zeros(n), np.zeros(n), np.zeros(n)
    m_el, m_du, m_ex = np.zeros((horizon, nseg)), np.zeros((horizon, nseg)), np.zeros((horizon, nseg))
    m_pd, m_pp = np.zeros(horizon), np.zeros(horizon)

    for k in range(1, horizon + 1):
        t = as_of_t + k
        alive = rem0 >= k
        b = scheduled_balance(upb0, rate, rem0, k - 1) * alive
        hpi_t = hpi_mat[gidx, t - hl - t0]
        mtm = 100.0 * b / (ov * hpi_t / hpi0)
        dyn = pl.DataFrame({
            "loan_age": age0 + k, "mtm_ltv": mtm,
            "incentive": rate - path["pmms"][t - il - t0],
            "ur": np.full(n, path["ur"][t - t0]), "ur_yoy": np.full(n, path["ur_yoy"][t - t0]),
        }).fill_nan(None)   # NaN -> null so the model's own missing-value rules apply
        X, _ = design(pl.concat([hz_static, dyn], how="horizontal"), hcfg)
        h = hazard.hazards(X)
        hd, hp = h[:, 0], h[:, 1]
        w = _overlay_weight(k, ocfg["horizon_months"], ocfg["decay_months"]) if ocfg else 0.0
        hd = hd * (1.0 + (mult[seg] - 1.0) * w)
        if k == 1:
            hd = np.where(in_default, 1.0, hd)
            hp = np.where(in_default, 0.0, hp)
        tot = hd + hp
        scale = np.where(tot > 1.0, 1.0 / tot, 1.0)
        hd, hp = hd * scale * alive, hp * scale * alive

        md, mp = S * hd, S * hp
        loan_pd += md
        loan_pp += mp
        m_ex[k - 1] = np.bincount(seg, weights=S * b, minlength=nseg)
        m_du[k - 1] = np.bincount(seg, weights=md * b, minlength=nseg)
        m_pd[k - 1], m_pp[k - 1] = md.sum(), mp.sum()
        if not pd_only:
            hpi_l = hpi_mat[gidx, t - lhl - t0]
            hpi_f = hpi_mat[gidx, t - lhl + fwd - t0]
            ldf = pl.concat([lg_static, pl.DataFrame({
                "mtm_ltv_default": 100.0 * b / (ov * hpi_l / hpi0), "hpi_fwd": hpi_f / hpi_l - 1.0,
                "upb_default": np.maximum(b, 1.0), "age_default": age0 + k,
                "hpi_us_d": np.full(n, path["hpi"]["US"][t - lhl - t0]),
                "default_year": np.full(n, t // 12, dtype=np.int64)}).fill_nan(None)], how="horizontal")
            el = md * lgd_model.lgd(ldf) * b * ead_factor
            loan_el += el
            m_el[k - 1] = np.bincount(seg, weights=el, minlength=nseg)
        S = S * (1.0 - hd - hp)
        if not alive.any():
            break
    return Projection(loan_pd, loan_pp, loan_el, m_el, m_du, m_ex, m_pd, m_pp)
