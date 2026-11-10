"""One row per defaulted loan: resolution, loss, and drivers known at the default month.

Resolution of a default (D-027, D-028):
  liquidated  zero-balance code 02/03/09/15 with Actual Loss disclosed
  cured       returned to current or paid off after default; code 16 (reperforming-loan
              sale) per config
  unresolved  still in default at the cutoff (excluded: outcome unknown)
  excluded    code 96 (defect), liquidations without loss data
Severity (D-029) = (Actual Loss - delinquent accrued interest) / UPB at default, i.e. on
principal, because accrued interest on nonaccrual loans is reversed, not reserved.
"""
from __future__ import annotations

import polars as pl

from cecl.data.panel import ym_to_index


def build_lgd_vintage(perf: pl.LazyFrame, orig: pl.LazyFrame, outcomes: pl.LazyFrame,
                      hpi: pl.DataFrame, macro_m: pl.DataFrame, dcfg: dict, lcfg: dict) -> pl.DataFrame:
    states = hpi.filter(pl.col("geo") != "US")["geo"].unique().to_list()
    o = outcomes.filter(pl.col("default_period").is_not_null()).select(
        "loan_id", "vintage", "default_period", "zb_code", "zb_period", "default_cured", "actual_loss",
        "delinq_accrued_int", "mi_recoveries", "net_sales_proceeds", "total_expenses", "zb_removal_upb")
    ids = o.select("loan_id")

    p = perf.join(ids, on="loan_id", how="semi").join(o.select("loan_id", "default_period"), on="loan_id")
    # UPB at default: last positive balance at or before the default month
    at_def = (p.filter((pl.col("period") <= pl.col("default_period")) & (pl.col("cur_upb") > 0))
              .sort("loan_id", "period").group_by("loan_id")
              .agg(upb_default=pl.col("cur_upb").last(), age_default=pl.col("loan_age").last(),
                   rate_default=pl.col("cur_rate").last()))
    # EAD validation: balance, rate, remaining term L months before default
    L = lcfg["ead_lookback_months"]
    pre = (p.with_columns(lag=ym_to_index("default_period") - ym_to_index("period"))
           .filter(pl.col("lag") == L)
           .select("loan_id", upb_pre=pl.col("cur_upb"), rate_pre=pl.col("cur_rate"), rem_pre=pl.col("rem_months")))

    og = orig.select("loan_id", "fico", "dti", "ltv", "mi_pct", "orig_upb", "state", "prop_type", "occupancy",
                     "units", "purpose", "n_borrowers", "first_pay_date",
                     hpi_geo=pl.when(pl.col("state").is_in(states)).then(pl.col("state")).otherwise(pl.lit("US")),
                     t_orig=ym_to_index("first_pay_date") - 1)
    df = o.join(at_def, on="loan_id", how="left").join(pre, on="loan_id", how="left").join(og, on="loan_id", how="left")

    lag = lcfg["hpi_lag_months"]
    hp = hpi.lazy()
    df = (df.with_columns(t_def=ym_to_index("default_period"))
          .with_columns(t_h=pl.col("t_def") - lag, t_h0=pl.col("t_orig") - lag,
                        t_hf=pl.col("t_def") - lag + lcfg["hpi_forward_months"])
          .join(hp.rename({"t": "t_h", "hpi": "hpi_d"}), left_on=["hpi_geo", "t_h"], right_on=["geo", "t_h"], how="left")
          .join(hp.rename({"t": "t_h0", "hpi": "hpi_0"}), left_on=["hpi_geo", "t_h0"], right_on=["geo", "t_h0"], how="left")
          # National FMHPI at default: deflates loan size to real terms (D-041)
          .join(hp.filter(pl.col("geo") == "US").select(t_h="t", hpi_us_d="hpi"), on="t_h", how="left")
          # Forward HPI path after default (scenario-conditional input, D-030)
          .join(hp.rename({"t": "t_hf", "hpi": "hpi_f"}), left_on=["hpi_geo", "t_hf"], right_on=["geo", "t_hf"], how="left")
          .join(macro_m.lazy().select("t", "ur", "ur_yoy").rename({"t": "t_def"}), on="t_def", how="left"))

    liq_codes = dcfg["zb_default"]
    cure_16 = lcfg["zb16_treatment"] == "cure"
    num = pl.col("actual_loss") - (pl.col("delinq_accrued_int").fill_null(0.0) if lcfg["severity_basis"] == "principal" else 0.0)
    lo, hi = lcfg["clip"]
    na = lambda c, v: pl.when(pl.col(c) == v).then(None).otherwise(pl.col(c))  # noqa: E731
    orig_value = pl.col("orig_upb") / (na("ltv", 999) / 100.0)
    return df.with_columns(
        resolution=pl.when(pl.col("zb_code").is_in(liq_codes) & pl.col("actual_loss").is_not_null()).then(pl.lit("liquidated"))
        .when(pl.col("zb_code").is_in(liq_codes)).then(pl.lit("excluded"))
        .when(pl.col("zb_code") == "96").then(pl.lit("excluded"))
        .when(pl.col("zb_code") == "16").then(pl.lit("cured" if cure_16 else "excluded"))
        .when(pl.col("default_cured")).then(pl.lit("cured"))
        .otherwise(pl.lit("unresolved")),
        severity_raw=num / pl.col("upb_default"),
        mtm_ltv_default=100.0 * pl.col("upb_default") / (orig_value * pl.col("hpi_d") / pl.col("hpi_0")),
        months_to_disposition=ym_to_index("zb_period") - pl.col("t_def"),
        hpi_fwd=pl.col("hpi_f") / pl.col("hpi_d") - 1.0,
        fico=na("fico", 9999), dti=na("dti", 999), mi_pct=na("mi_pct", 999),
    ).with_columns(
        severity=pl.col("severity_raw").clip(lo, hi),
        default_year=pl.col("default_period") // 100,
    ).collect()
