"""Loan portfolio as of a month-end D, built only from data observed on or before D.

Segments at D (D-036):
  S0 clean      current, never in forbearance, never defaulted
  S1 history    current, but forbearance history or a prior (cured) default
  S2 dpd30      30 days past due, not in forbearance
  S3 dpd60      60 days past due, not in forbearance
  S5 forb_dq    delinquent while in forbearance
  S4 default    in default at D (90+ DPD outside forbearance, or REO): PD = 1
Optionally attaches the realized outcome over (D, D+H] for overlay calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cecl.data.panel import add_month_flags, ym_to_index

SEGMENTS = ["S0_clean", "S1_history", "S2_dpd30", "S3_dpd60", "S4_default", "S5_forb_dq"]


def build_portfolio(cfg, as_of: int, pmms_m: pl.DataFrame, hpi: pl.DataFrame,
                    outcome_months: int | None = None) -> pl.DataFrame:
    d = cfg["data"]
    t_d = (as_of // 100) * 12 + as_of % 100 - 1
    inter = cfg.path("interim") / "freddie"
    states = hpi.filter(pl.col("geo") != "US")["geo"].unique().to_list()
    frames = []
    for y in range(d["vintages"][0], as_of // 100 + 1):
        f = inter / f"perf_{y}.parquet"
        if not Path(f).exists():
            continue
        p = add_month_flags(pl.scan_parquet(f), d)
        hist = p.filter(pl.col("period") <= as_of)
        state = (hist.group_by("loan_id").agg(
            ever_forb=pl.col("forbearance_row").any(),
            prior_default=pl.col("dflt_row").any(),
        ))
        cur = hist.filter((pl.col("period") == as_of) & pl.col("zb_code").is_null()).select(
            "loan_id", "vintage", "cur_upb", "cur_rate", "rem_months", "loan_age", "delq", "delq_num",
            "forbearance_row", "dflt_row")
        df = cur.join(state, on="loan_id", how="left")
        if outcome_months:
            fut = (p.filter((pl.col("period") > as_of) & (pl.col("t") <= t_d + outcome_months))
                   .group_by("loan_id").agg(default_next=pl.col("dflt_row").any(),
                                            prepay_next=pl.col("prepay_row").any()))
            df = df.join(fut, on="loan_id", how="left").with_columns(
                pl.col("default_next").fill_null(False), pl.col("prepay_next").fill_null(False))
        o = pl.scan_parquet(inter / f"orig_{y}.parquet").select(
            "loan_id", "fico", "cltv", "ltv", "dti", "orig_upb", "orig_rate", "purpose", "occupancy",
            "n_borrowers", "state", "mi_pct", "prop_type", "units", "first_pay_date")
        frames.append(df.join(o, on="loan_id", how="left").collect())
    pf = pl.concat(frames, how="diagonal_relaxed")

    na = lambda c, v: pl.when(pl.col(c) == v).then(None).otherwise(pl.col(c))  # noqa: E731
    pf = pf.with_columns(
        fico=na("fico", 9999), cltv=na("cltv", 999), dti=na("dti", 999), mi_pct=na("mi_pct", 999),
        ltv=na("ltv", 999),
        hpi_geo=pl.when(pl.col("state").is_in(states)).then(pl.col("state")).otherwise(pl.lit("US")),
        t_orig=ym_to_index("first_pay_date") - 1,
    ).with_columns(orig_value=pl.col("orig_upb") / (pl.col("ltv") / 100.0))

    lag = cfg["hazard"]["hpi_lag_months"]
    pf = (pf.with_columns(t_h0=pl.col("t_orig") - lag)
          .join(hpi.rename({"t": "t_h0", "hpi": "hpi_0"}), left_on=["hpi_geo", "t_h0"], right_on=["geo", "t_h0"], how="left")
          .join(pmms_m.rename({"t": "t_orig", "pmms": "pmms_0"}), on="t_orig", how="left")
          .with_columns(sato=pl.col("orig_rate") - pl.col("pmms_0")))

    in_forb_dq = pl.col("forbearance_row") & (pl.col("delq_num").fill_null(0) >= 1)
    seg = (pl.when(pl.col("dflt_row")).then(pl.lit("S4_default"))
           .when(in_forb_dq).then(pl.lit("S5_forb_dq"))
           .when(pl.col("delq_num") == 2).then(pl.lit("S3_dpd60"))
           .when(pl.col("delq_num") == 1).then(pl.lit("S2_dpd30"))
           .when(pl.col("ever_forb") | pl.col("prior_default")).then(pl.lit("S1_history"))
           .otherwise(pl.lit("S0_clean")))
    return pf.with_columns(segment=seg, seg_idx=seg.replace_strict({s: i for i, s in enumerate(SEGMENTS)},
                                                                    return_dtype=pl.Int8))
