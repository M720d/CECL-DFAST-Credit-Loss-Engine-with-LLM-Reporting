"""Loan-month panel for the competing-risk hazard (D-021).

Outcome in month t: 0 survive, 1 default (first default month), 2 voluntary prepayment.
Risk set: months up to and including the terminal event; censored loans contribute
months before their zero-balance month. Loan-level dynamic covariates (balance, note
rate, forbearance) are lagged one month, so month t's hazard uses only information
from the end of month t-1. Macro inputs are lagged per config.

Loans are split by a seeded hash: 'fit' loans (sampled survive-months, for estimation)
and 'valid' loans (all months, for backtesting); the two sets never share a loan.
"""
from __future__ import annotations

import polars as pl

from cecl.data.panel import EVENT_DEFAULT, EVENT_PREPAY, add_month_flags, ym_to_index


def loan_bucket(seed: int) -> pl.Expr:
    return (pl.col("loan_id").hash(seed) % 10_000).cast(pl.Int32)


def build_vintage_panel(perf: pl.LazyFrame, orig: pl.LazyFrame, outcomes: pl.LazyFrame,
                        macro_m: pl.DataFrame, pmms_m: pl.DataFrame, hpi: pl.DataFrame,
                        dcfg: dict, hcfg: dict, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    fit_hi = int(hcfg["fit_loan_share"] * 10_000)
    val_hi = fit_hi + int(hcfg["valid_loan_share"] * 10_000)
    states = hpi.filter(pl.col("geo") != "US")["geo"].unique().to_list()

    p = (add_month_flags(perf, dcfg)
         .select("loan_id", "period", "t", "loan_age", "cur_upb", "cur_rate", "forbearance_row")
         .with_columns(bucket=loan_bucket(seed))
         .filter(pl.col("bucket") < val_hi)
         .sort("loan_id", "period")
         .with_columns(upb_l1=pl.col("cur_upb").shift(1).over("loan_id"),
                       rate_l1=pl.col("cur_rate").shift(1).over("loan_id"),
                       forb_l1=pl.col("forbearance_row").shift(1).over("loan_id").cast(pl.Int8))
         .filter(pl.col("upb_l1").is_not_null()))          # first observed month has no lag

    o = outcomes.select("loan_id", "event", "event_period", "zb_period", "default_period")
    p = p.join(o, on="loan_id", how="left").with_columns(
        y=pl.when((pl.col("event") == EVENT_DEFAULT) & (pl.col("period") == pl.col("default_period"))).then(1)
        .when((pl.col("event") == EVENT_PREPAY) & (pl.col("period") == pl.col("zb_period"))).then(2)
        .otherwise(0).cast(pl.Int8),
    ).filter(
        pl.when(pl.col("event").is_in([EVENT_DEFAULT, EVENT_PREPAY]))
        .then(pl.col("period") <= pl.col("event_period"))
        # censored: drop the zero-balance month itself (no event, loan leaves the risk set)
        .otherwise(pl.col("zb_period").is_null() | (pl.col("period") < pl.col("zb_period")))
    )

    og = orig.select(
        "loan_id", "fico", "cltv", "ltv", "dti", "orig_upb", "orig_rate", "purpose", "occupancy",
        "n_borrowers", "state",
        hpi_geo=pl.when(pl.col("state").is_in(states)).then(pl.col("state")).otherwise(pl.lit("US")),
        t_orig=ym_to_index("first_pay_date") - 1,
    )
    p = p.join(og, on="loan_id", how="left")

    hl, il = hcfg["hpi_lag_months"], hcfg["incentive_lag_months"]
    hp = hpi.lazy()
    p = (p.with_columns(t_h=pl.col("t") - hl, t_h0=pl.col("t_orig") - hl, t_i=pl.col("t") - il)
         .join(hp.rename({"t": "t_h", "hpi": "hpi_t"}), left_on=["hpi_geo", "t_h"], right_on=["geo", "t_h"], how="left")
         .join(hp.rename({"t": "t_h0", "hpi": "hpi_0"}), left_on=["hpi_geo", "t_h0"], right_on=["geo", "t_h0"], how="left")
         .join(pmms_m.lazy().rename({"t": "t_i", "pmms": "pmms_i"}), on="t_i", how="left")
         .join(pmms_m.lazy().rename({"t": "t_orig", "pmms": "pmms_0"}), on="t_orig", how="left")
         .join(macro_m.lazy().select("t", "ur", "ur_yoy"), on="t", how="left"))

    na = lambda c, v: pl.when(pl.col(c) == v).then(None).otherwise(pl.col(c))  # noqa: E731
    orig_value = pl.col("orig_upb") / (na("ltv", 999) / 100.0)
    p = p.with_columns(
        fico=na("fico", 9999), cltv=na("cltv", 999), dti=na("dti", 999),
        mtm_ltv=100.0 * pl.col("upb_l1") / (orig_value * pl.col("hpi_t") / pl.col("hpi_0")),
        incentive=pl.col("rate_l1") - pl.col("pmms_i"),
        sato=pl.col("orig_rate") - pl.col("pmms_0"),
    ).select("loan_id", "bucket", "period", "y", "loan_age", "fico", "cltv", "dti", "orig_upb", "sato",
             "mtm_ltv", "incentive", "ur", "ur_yoy", "forb_l1", "purpose", "occupancy", "n_borrowers", "state")

    df = p.collect()
    fit = df.filter(pl.col("bucket") < fit_hi)
    # Choice-based sampling: keep all event months, a seeded r-share of survive months.
    r = hcfg["survive_sample_rate"]
    keep = (pl.struct("loan_id", "period").hash(seed + 1) % 1_000_000) < int(r * 1_000_000)
    fit = fit.filter((pl.col("y") > 0) | keep)
    valid = df.filter(pl.col("bucket") >= fit_hi)
    return fit, valid
