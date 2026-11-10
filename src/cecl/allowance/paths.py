"""Monthly macro paths for projection: realized history, or a CECL/DFAST scenario.

A path is a dict of numpy arrays indexed by (t - t0), t = month index, covering
[t0, t1]: ur, ur_yoy (one-quarter lag, as in training), pmms (monthly mortgage rate),
and state HPI levels (dict geo -> array). The same quarterly->monthly mapping as the
hazard training (cecl.hazard.macro.quarterly_to_monthly) is reused, so definitions match.

Scenario construction (D-033 .. D-035):
  scaled scenario:  x = base + scale * (severely_adverse - base)   (HPI on log scale)
  R&S period:       first rs_quarters of the scaled scenario
  reversion:        straight-line from the last R&S value to the long-run value
  after reversion:  long-run value (UR, mortgage rate levels; HPI long-run log growth)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from cecl.hazard.macro import quarterly_to_monthly


def q_index(year: int, quarter: int) -> int:
    return year * 4 + quarter - 1


def long_run(fed_hist: pl.DataFrame, lcfg: dict) -> dict:
    def mean_between(col, years):
        h = fed_hist.filter(pl.col("q").is_between(q_index(years[0], 1), q_index(years[1], 4)))
        return h, float(h[col].mean())
    _, ur = mean_between("ur", lcfg["ur_years"])
    _, mr = mean_between("mort_rate", lcfg["mort_rate_years"])
    h, _ = mean_between("hpi_nat", lcfg["hpi_growth_years"])
    g = float(np.log(h["hpi_nat"]).diff().drop_nulls().mean())
    return {"ur": ur, "mort_rate": mr, "hpi_growth_q": g}


def scenario_quarterly(fed_hist: pl.DataFrame, base: pl.DataFrame, sa: pl.DataFrame, scale: float,
                       rs_q: int, rev_q: int, lr: dict, ur_floor: float, n_quarters: int) -> pl.DataFrame:
    """History through jump-off, then scaled scenario for rs_q quarters, reversion, long run."""
    b, s = base.sort("q"), sa.sort("q")
    ur = b["ur"].to_numpy() + scale * (s["ur"].to_numpy() - b["ur"].to_numpy())
    mr = b["mort_rate"].to_numpy() + scale * (s["mort_rate"].to_numpy() - b["mort_rate"].to_numpy())
    lh = np.log(b["hpi_nat"].to_numpy()) + scale * (np.log(s["hpi_nat"].to_numpy()) - np.log(b["hpi_nat"].to_numpy()))
    ur = np.maximum(ur, ur_floor)
    if rs_q > len(ur):
        raise ValueError(f"R&S period {rs_q}q exceeds the {len(ur)} scenario quarters available")
    q0 = int(b["q"][0])
    last_hist = fed_hist.filter(pl.col("q") == q0 - 1).row(0, named=True)
    g_rs = np.diff(np.concatenate([[np.log(last_hist["hpi_nat"])], lh]))[:rs_q]
    ur_out, mr_out, g_out = list(ur[:rs_q]), list(mr[:rs_q]), list(g_rs)
    for k in range(1, n_quarters - rs_q + 1):          # reversion then long run
        w = min(k / rev_q, 1.0) if rev_q > 0 else 1.0
        ur_out.append((1 - w) * ur[rs_q - 1] + w * lr["ur"])
        mr_out.append((1 - w) * mr[rs_q - 1] + w * lr["mort_rate"])
        g_out.append((1 - w) * g_rs[-1] + w * lr["hpi_growth_q"])
    hpi_path = last_hist["hpi_nat"] * np.exp(np.cumsum(g_out))
    proj = pl.DataFrame({"q": np.arange(q0, q0 + len(ur_out), dtype=np.int32), "ur": ur_out,
                         "mort_rate": mr_out, "hpi_nat": hpi_path})
    return pl.concat([fed_hist.filter(pl.col("q") < q0).select(proj.columns), proj])


def monthly_path(fed_q: pl.DataFrame, pmms_hist: pl.DataFrame, fmhpi: pl.DataFrame, betas: pl.DataFrame,
                 t0: int, t1: int, lag_q: int, hist_through: int, state_hpi_mode: str) -> dict:
    """Build arrays for months t0..t1.

    hist_through: last month index whose *actual* PMMS/FMHPI may be used; later months
    come from fed_q (scenario or history). state_hpi_mode: 'actual' (calibration on
    history: realized FMHPI) or 'betas' (projection: national path -> states via betas).
    """
    n = t1 - t0 + 1
    m = quarterly_to_monthly(fed_q, lag_q).filter(pl.col("t").is_between(t0, t1)).sort("t")
    assert m.height == n, "macro path does not cover the requested horizon"
    out = {"t0": t0, "ur": m["ur"].to_numpy(), "ur_yoy": m["ur_yoy"].to_numpy()}

    # Mortgage rate: actual monthly PMMS through hist_through, then the quarterly path (no lag).
    qm = fed_q.select(pl.col("q"), pl.col("mort_rate"))
    tt = np.arange(t0, t1 + 1)
    q_of_t = pl.DataFrame({"t": tt, "q": (tt // 3).astype(np.int32)}).join(qm, on="q", how="left")["mort_rate"].to_numpy()
    act = dict(zip(pmms_hist["t"].to_list(), pmms_hist["pmms"].to_list()))
    out["pmms"] = np.array([act.get(t, np.nan) if t <= hist_through else np.nan for t in tt])
    out["pmms"] = np.where(np.isnan(out["pmms"]), q_of_t, out["pmms"])

    # State HPI levels
    geos = fmhpi["geo"].unique().to_list()
    out["hpi"] = {}
    if state_hpi_mode == "actual":
        for g, grp in fmhpi.filter(pl.col("t").is_between(t0, t1)).group_by("geo"):
            arr = np.full(n, np.nan)
            arr[grp["t"].to_numpy() - t0] = grp["hpi"].to_numpy()
            out["hpi"][g[0]] = arr
        return out
    # betas: monthly log growth = (alpha + beta * national quarterly log growth) / 3
    nat = fed_q.sort("q").with_columns(dn=pl.col("hpi_nat").log().diff())
    dn_of_t = pl.DataFrame({"t": tt, "q": (tt // 3).astype(np.int32)}).join(nat.select("q", "dn"), on="q", how="left")["dn"].to_numpy()
    bmap = {r["geo"]: (r["alpha"], r["beta"]) for r in betas.iter_rows(named=True)}
    for g in geos:
        hist = fmhpi.filter((pl.col("geo") == g) & pl.col("t").is_between(t0, hist_through)).sort("t")
        arr = np.full(n, np.nan)
        arr[hist["t"].to_numpy() - t0] = hist["hpi"].to_numpy()
        a, b = bmap.get(g, (0.0, 1.0))
        start = hist_through - t0
        level = arr[start]
        for i in range(start + 1, n):
            level = level * np.exp((a + b * dn_of_t[i]) / 3.0)
            arr[i] = level
        out["hpi"][g] = arr
    return out
