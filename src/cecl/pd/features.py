"""Snapshot-level features for the 12-month PD scorecard.

Every feature is known at the snapshot month t (no lookahead):
  * origination attributes (fixed at origination),
  * loan state at t (age, balance, delinquency, modification),
  * market data at t: PMMS 30Y rate (weekly, published in real time) and the state
    FMHPI lagged `hpi_lag_months`, because an index for month t is not published at t.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from cecl.data.panel import ym_to_index

NUMERIC = ["fico", "ltv", "cltv", "dti", "orig_rate", "log_orig_upb", "orig_term", "mi_pct",
           "loan_age", "delq_num", "mtm_ltv", "hpi_change", "sato", "refi_incentive",
           "paydown", "n_borrowers", "units"]
CATEGORICAL = ["occupancy", "purpose", "channel", "prop_type", "first_time_buyer", "state",
               "ever_30dpd", "modified", "harp_flag"]


def load_pmms(path: Path) -> pl.DataFrame:
    """Monthly average of weekly PMMS 30Y fixed rate, keyed by month index."""
    w = pl.read_csv(path, try_parse_dates=True).rename({"MORTGAGE30US": "pmms"})
    w = w.filter(pl.col("pmms").cast(pl.Float64, strict=False).is_not_null())
    return (w.with_columns(pl.col("pmms").cast(pl.Float64),
                           t=pl.col("observation_date").dt.year() * 12 + pl.col("observation_date").dt.month() - 1)
            .group_by("t").agg(pl.col("pmms").mean()).sort("t"))


def load_hpi(path: Path) -> pl.DataFrame:
    """State + national FMHPI (NSA), keyed by (geo, month index). National geo = 'US'."""
    h = pl.read_csv(path, schema_overrides={"GEO_Code": pl.Utf8})
    h = h.filter(pl.col("GEO_Type").is_in(["State", "US"]))
    return h.select(
        geo=pl.when(pl.col("GEO_Type") == "US").then(pl.lit("US")).otherwise(pl.col("GEO_Name")),
        t=pl.col("Year") * 12 + pl.col("Month") - 1,
        hpi=pl.col("Index_NSA"),
    )


def build_features(snaps: pl.LazyFrame, orig: pl.LazyFrame, pmms: pl.DataFrame,
                   hpi: pl.DataFrame, hpi_lag: int) -> pl.LazyFrame:
    states = hpi.filter(pl.col("geo") != "US")["geo"].unique().to_list()
    o = orig.select(
        "loan_id", "fico", "ltv", "cltv", "dti", "orig_rate", "orig_upb", "orig_term", "mi_pct",
        "n_borrowers", "units", "occupancy", "purpose", "channel", "prop_type", "first_time_buyer",
        "state", "harp_flag", "first_pay_date",
        # Territories without a state FMHPI fall back to the national index.
        hpi_geo=pl.when(pl.col("state").is_in(states)).then(pl.col("state")).otherwise(pl.lit("US")),
        # Note date is typically ~1-2 months before first payment; use first payment - 1.
        t_orig=ym_to_index("first_pay_date") - 1,
    )
    s = snaps.with_columns(t=ym_to_index("snapshot_period")).join(o, on="loan_id", how="left")

    hp = hpi.lazy()
    s = (s.with_columns(t_hpi=pl.col("t") - hpi_lag, t_hpi0=pl.col("t_orig") - hpi_lag)
         .join(hp.rename({"t": "t_hpi", "hpi": "hpi_t"}), left_on=["hpi_geo", "t_hpi"], right_on=["geo", "t_hpi"], how="left")
         .join(hp.rename({"t": "t_hpi0", "hpi": "hpi_0"}), left_on=["hpi_geo", "t_hpi0"], right_on=["geo", "t_hpi0"], how="left")
         .join(pmms.lazy().rename({"pmms": "pmms_t"}), on="t", how="left")
         .join(pmms.lazy().rename({"t": "t_orig", "pmms": "pmms_0"}), on="t_orig", how="left"))

    # Sentinels to null (Not Available codes, User Guide R48).
    def na(c, codes):
        return pl.when(pl.col(c).is_in(codes)).then(None).otherwise(pl.col(c)).alias(c)

    s = s.with_columns(na("fico", [9999]), na("ltv", [999]), na("cltv", [999]), na("dti", [999]),
                       na("mi_pct", [999]), na("n_borrowers", [99]), na("units", [99]),
                       na("occupancy", ["9"]), na("purpose", ["9"]), na("channel", ["9"]),
                       na("prop_type", ["99"]), na("first_time_buyer", ["9"]))
    orig_value = pl.col("orig_upb") / (pl.col("ltv") / 100.0)
    return s.with_columns(
        log_orig_upb=pl.col("orig_upb").log(),
        hpi_change=pl.col("hpi_t") / pl.col("hpi_0") - 1.0,
        mtm_ltv=100.0 * pl.col("cur_upb") / (orig_value * pl.col("hpi_t") / pl.col("hpi_0")),
        sato=pl.col("orig_rate") - pl.col("pmms_0"),
        refi_incentive=pl.col("cur_rate") - pl.col("pmms_t"),
        paydown=1.0 - pl.col("cur_upb") / pl.col("orig_upb"),
        ever_30dpd=pl.col("ever_30dpd").cast(pl.Utf8),
        modified=pl.col("mod_flag").is_in(["Y", "P"]).fill_null(False).cast(pl.Utf8),
    ).select(["loan_id", "vintage", "snapshot_period", "target", "target_spec", "eligible_spec",
              *NUMERIC, *CATEGORICAL])
