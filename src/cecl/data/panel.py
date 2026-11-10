"""Loan-month panel, loan-level outcomes, and the 12-month PD snapshot dataset.

Definitions (all parameters from config.yaml -> data):
  default  first month with delinquency >= dpd_default_threshold, or status 'RA'
           (REO acquired), or a default-type zero-balance code (02/03/09/15).
  prepay   zero-balance code 01 while scheduled term remains (rem_months > 0 in the
           prior month). Code 01 at scheduled maturity is a payoff, not a prepayment.
  censor   codes 16/96, matured loans, and loans still active at the cutoff.
Competing risks: whichever event comes first wins; a loan has at most one terminal event.
"""
from __future__ import annotations

import polars as pl

EVENT_NONE, EVENT_DEFAULT, EVENT_PREPAY, EVENT_CENSOR = 0, 1, 2, 3


def ym_to_index(col: str) -> pl.Expr:
    """YYYYMM -> months since year 0, so month arithmetic is plain integer math."""
    return (pl.col(col) // 100) * 12 + (pl.col(col) % 100 - 1)


def add_month_flags(perf: pl.LazyFrame, dcfg: dict) -> pl.LazyFrame:
    """Per-row event flags on the raw loan-month panel."""
    covid_lo, covid_hi = dcfg["covid_window"]
    return perf.with_columns(
        delq_num=pl.when(pl.col("delq") == "RA").then(None)
        .otherwise(pl.col("delq").cast(pl.Int16, strict=False)),
        t=ym_to_index("period"),
    ).with_columns(
        # is_in() on a null returns null, not False: fill explicitly or filters drop rows.
        dpd_row=(pl.col("delq_num") >= dcfg["dpd_default_threshold"]).fill_null(False),
        hard_default_row=((pl.col("delq") == "RA").fill_null(False)
                          | pl.col("zb_code").is_in(dcfg["zb_default"]).fill_null(False)),
        prepay_row=pl.col("zb_code").is_in(dcfg["zb_prepay"]).fill_null(False),
        censor_row=pl.col("zb_code").is_in(dcfg["zb_censor"]).fill_null(False),
        forbearance_row=(
            (pl.col("borrower_assist") == "F")
            | (pl.col("payment_deferral") == "C")   # C = deferral granted this period
            | (pl.col("disaster_flag") == "Y")
        ).fill_null(False),
        covid_row=pl.col("period").is_between(covid_lo, covid_hi),
    ).with_columns(
        # 'spec': 90+ DPD in any month. 'ex_forbearance': 90+ DPD months while the loan is
        # in forbearance / payment deferral / disaster relief do not trigger default.
        # REO acquisition and default-type zero-balance codes always count.
        dflt_spec_row=pl.col("dpd_row") | pl.col("hard_default_row"),
        dflt_xf_row=(pl.col("dpd_row") & ~pl.col("forbearance_row")) | pl.col("hard_default_row"),
    ).with_columns(
        dflt_row=pl.col("dflt_xf_row" if dcfg["default_definition"] == "ex_forbearance" else "dflt_spec_row"),
    )


def loan_outcomes(perf: pl.LazyFrame, dcfg: dict) -> pl.LazyFrame:
    """One row per loan: first default / prepay / censor timing and terminal event."""
    p = add_month_flags(perf, dcfg).sort(["loan_id", "period"])
    # Remaining term in the month *before* a zero balance decides prepay vs matured.
    p = p.with_columns(prev_rem=pl.col("rem_months").shift(1).over("loan_id"))
    agg = p.group_by("loan_id").agg(
        vintage=pl.col("vintage").first(),
        first_period=pl.col("period").min(),
        last_period=pl.col("period").max(),
        first_age=pl.col("loan_age").min(),
        n_months=pl.len(),
        default_period=pl.col("period").filter(pl.col("dflt_row")).min(),
        default_age=pl.col("loan_age").filter(pl.col("dflt_row")).min(),
        default_in_forbearance=pl.col("forbearance_row").filter(pl.col("dflt_row")).first(),
        default_period_spec=pl.col("period").filter(pl.col("dflt_spec_row")).min(),
        default_period_xf=pl.col("period").filter(pl.col("dflt_xf_row")).min(),
        zb_code=pl.col("zb_code").drop_nulls().last(),
        zb_period=pl.col("period").filter(pl.col("zb_code").is_not_null()).last(),
        zb_prev_rem=pl.col("prev_rem").filter(pl.col("zb_code").is_not_null()).last(),
        zb_removal_upb=pl.col("zb_removal_upb").drop_nulls().last(),
        actual_loss=pl.col("actual_loss").drop_nulls().last(),
        mi_recoveries=pl.col("mi_recoveries").drop_nulls().last(),
        net_sales_proceeds=pl.col("net_sales_proceeds").drop_nulls().last(),
        non_mi_recoveries=pl.col("non_mi_recoveries").drop_nulls().last(),
        total_expenses=pl.col("total_expenses").drop_nulls().last(),
        delinq_accrued_int=pl.col("delinq_accrued_int").drop_nulls().last(),
        ever_mod=pl.col("mod_flag").is_in(["Y", "P"]).fill_null(False).any(),  # Y current, P prior
        last_delq=pl.col("delq_num").last(),
        ever_forbearance=pl.col("forbearance_row").any(),
    )
    is_prepay = pl.col("zb_code").is_in(dcfg["zb_prepay"]) & (pl.col("zb_prev_rem").fill_null(1) > 1)
    is_matured = pl.col("zb_code").is_in(dcfg["zb_prepay"]) & ~is_prepay
    return agg.with_columns(
        matured=is_matured,
        event=pl.when(pl.col("default_period").is_not_null()).then(EVENT_DEFAULT)
        .when(is_prepay).then(EVENT_PREPAY)
        .otherwise(EVENT_CENSOR).cast(pl.Int8),
        # Hit the default definition but later returned to current or paid off without loss.
        default_cured=(pl.col("default_period").is_not_null()
                       & ~pl.col("zb_code").is_in(dcfg["zb_default"]).fill_null(False)
                       & ((pl.col("last_delq") == 0) | pl.col("zb_code").is_in(dcfg["zb_prepay"])).fill_null(False)),
        event_period=pl.coalesce(pl.col("default_period"), pl.col("zb_period"), pl.col("last_period")),
    )


def pd_snapshots(perf: pl.LazyFrame, outcomes: pl.LazyFrame, dcfg: dict) -> pl.LazyFrame:
    """12-month PD dataset: one row per loan per snapshot (loan age multiple of N).

    Eligible snapshot: loan observed that month, not yet in default (delq < threshold),
    and the 12-month window closes on or before the performance cutoff, so the label
    is never truncated by the end of the data (no partially observed targets).
    Target = 1 if the first default falls in (snapshot, snapshot + H].
    A loan that prepays inside the window without defaulting gets target 0.
    """
    h, step = dcfg["default_horizon_months"], dcfg["snapshot_every_months"]
    cutoff_t = (dcfg["performance_cutoff"] // 100) * 12 + dcfg["performance_cutoff"] % 100 - 1
    p = add_month_flags(perf, dcfg)
    o = outcomes.select("loan_id", "default_period", "default_period_spec").with_columns(
        dt=ym_to_index("default_period"), dt_spec=ym_to_index("default_period_spec"))
    # Behavioural history up to and including the snapshot month.
    p = p.sort(["loan_id", "period"]).with_columns(
        ever_30dpd=(pl.col("delq_num") >= 1).fill_null(False).cum_max().over("loan_id"),
        months_obs=pl.int_range(1, pl.len() + 1).over("loan_id"),
    )
    snaps = p.filter(
        (pl.col("loan_age") % step == 0)
        & pl.col("zb_code").is_null()
        & (pl.col("t") + h <= cutoff_t)
    ).join(o, on="loan_id", how="left").filter(
        # First default is absorbing: a loan leaves the population in its default month
        # and never re-enters after curing (otherwise cures inflate the non-default class).
        pl.col("dt").is_null() | (pl.col("t") < pl.col("dt"))
    )
    return snaps.with_columns(
        target=(pl.col("dt").is_not_null() & (pl.col("dt") > pl.col("t")) & (pl.col("dt") <= pl.col("t") + h))
        .cast(pl.Int8),
        # Sensitivity label under the strict spec definition, with its own eligibility.
        eligible_spec=pl.col("dt_spec").is_null() | (pl.col("t") < pl.col("dt_spec")),
        target_spec=(pl.col("dt_spec").is_not_null() & (pl.col("dt_spec") > pl.col("t"))
                     & (pl.col("dt_spec") <= pl.col("t") + h)).cast(pl.Int8),
        snapshot_period=pl.col("period"),
    ).select(
        "loan_id", "vintage", "snapshot_period", "loan_age", "cur_upb", "delq_num", "cur_rate",
        "mod_flag", "ever_30dpd", "forbearance_row", "covid_row", "eltv", "target",
        "target_spec", "eligible_spec",
    )
