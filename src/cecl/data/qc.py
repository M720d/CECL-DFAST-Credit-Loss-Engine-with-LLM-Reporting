"""Data quality checks. Every check returns numbers; pass/fail thresholds live in the report."""
from __future__ import annotations

import polars as pl

from cecl.data.panel import ym_to_index
from cecl.data.schema import ORIG_SENTINELS


def loss_residual() -> pl.Expr:
    """Freddie stores recoveries as negatives, so UPB + interest + expenses + recoveries = loss."""
    return (
        pl.col("zb_removal_upb").fill_null(0) + pl.col("delinq_accrued_int").fill_null(0)
        + pl.col("total_expenses").fill_null(0) + pl.col("net_sales_proceeds").fill_null(0)
        + pl.col("mi_recoveries").fill_null(0) + pl.col("non_mi_recoveries").fill_null(0)
        - pl.col("actual_loss")
    )


def qc_vintage(orig: pl.LazyFrame, perf: pl.LazyFrame, outcomes: pl.DataFrame, dcfg: dict) -> dict:
    o = orig.collect()
    res: dict = {"orig_rows": o.height, "orig_unique_ids": o["loan_id"].n_unique()}

    pl_ids = perf.select("loan_id").unique().collect()["loan_id"]
    res["perf_ids_not_in_orig"] = int((~pl_ids.is_in(o["loan_id"].implode())).sum())
    res["orig_ids_without_perf"] = int((~o["loan_id"].is_in(pl_ids.implode())).sum())

    pstats = perf.select(
        rows=pl.len(),
        dup_loan_month=pl.len() - pl.struct("loan_id", "period").n_unique(),
        max_period=pl.col("period").max(),
        bad_delq=(~pl.col("delq").str.contains(r"^(\d+|RA|XX)$")).fill_null(False).sum(),
        negative_upb=(pl.col("cur_upb") < 0).sum(),
    ).collect().row(0, named=True)
    res.update(pstats)

    gaps = perf.group_by("loan_id").agg(
        span=ym_to_index("period").max() - ym_to_index("period").min() + 1, n=pl.len()
    ).select(loans_with_gaps=(pl.col("span") != pl.col("n")).sum(),
             missing_months=(pl.col("span") - pl.col("n")).sum()).collect().row(0, named=True)
    res.update(gaps)

    # Sentinel ("Not Available") share per origination field.
    res["sentinel_share"] = {
        c: round(float(o[c].is_in(v).mean()), 5) for c, v in ORIG_SENTINELS.items()
    }
    res["null_share"] = {c: round(float(o[c].is_null().mean()), 5)
                         for c in ("fico", "dti", "ltv", "cltv", "orig_rate", "orig_upb", "state")}
    valid = o.filter(~pl.col("fico").is_in([9999]))
    res["fico_out_of_range"] = int(((valid["fico"] < 300) | (valid["fico"] > 850)).sum())
    res["rate_nonpositive"] = int((o["orig_rate"] <= 0).sum())

    res["zb_code_counts"] = dict(outcomes.group_by("zb_code").len().sort("zb_code").iter_rows())
    known = set(dcfg["zb_prepay"] + dcfg["zb_default"] + dcfg["zb_censor"])
    res["unknown_zb_loans"] = int(outcomes.filter(
        pl.col("zb_code").is_not_null() & ~pl.col("zb_code").is_in(list(known))).height)

    disp = outcomes.filter(pl.col("zb_code").is_in(dcfg["zb_default"]))
    with_loss = disp.filter(pl.col("actual_loss").is_not_null())
    res["default_dispositions"] = disp.height
    res["dispositions_with_loss"] = with_loss.height
    res["loss_identity_pass"] = int((loss_residual().abs() < 1.0).cast(pl.Int32).sum()
                                    .pipe(lambda e: with_loss.select(e)).item()) if with_loss.height else 0
    return res
