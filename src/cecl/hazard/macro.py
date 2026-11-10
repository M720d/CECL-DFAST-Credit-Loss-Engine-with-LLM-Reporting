"""Macro inputs for the hazard model, defined so scenarios can be plugged in unchanged.

National unemployment, mortgage rate and HPI come from the Fed's DFAST historic file
(same definitions as the scenario files, D-009). State house prices come from FMHPI;
scenario paths reach states through state betas on the national (Z.1) HPI.
All tables are keyed by month index t = year*12 + month-1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

FED_COLS = {"Unemployment rate": "ur", "Mortgage rate": "mort_rate", "House Price Index (Level)": "hpi_nat"}


def load_fed_quarterly(path: Path) -> pl.DataFrame:
    """Fed historic or scenario CSV -> quarterly frame with q = year*4 + quarter-1."""
    df = pl.read_csv(path)
    return df.select(
        q=pl.col("Date").str.slice(0, 4).cast(pl.Int32) * 4 + pl.col("Date").str.slice(6, 1).cast(pl.Int32) - 1,
        **{v: pl.col(k).cast(pl.Float64) for k, v in FED_COLS.items()},
    ).sort("q")


def quarterly_to_monthly(q: pl.DataFrame, lag_q: int) -> pl.DataFrame:
    """Each month takes its quarter's value lagged `lag_q` quarters; adds 4-quarter UR change."""
    q = q.with_columns(ur_yoy=pl.col("ur") - pl.col("ur").shift(4))
    months = pl.DataFrame({"t": np.arange(q["q"].min() * 3, (q["q"].max() + 1 + lag_q) * 3, dtype=np.int32)})
    return (months.with_columns(q=(pl.col("t") // 3 - lag_q).cast(pl.Int32))
            .join(q, on="q", how="left").drop("q").drop_nulls(["ur"]))


def load_fmhpi_states(path: Path) -> pl.DataFrame:
    """Monthly NSA FMHPI by state plus 'US'. Columns: geo, t, hpi."""
    h = pl.read_csv(path, schema_overrides={"GEO_Code": pl.Utf8}).filter(pl.col("GEO_Type").is_in(["State", "US"]))
    return h.select(
        geo=pl.when(pl.col("GEO_Type") == "US").then(pl.lit("US")).otherwise(pl.col("GEO_Name")),
        t=(pl.col("Year") * 12 + pl.col("Month") - 1).cast(pl.Int32),
        hpi=pl.col("Index_NSA"),
    )


def state_hpi_betas(fmhpi: pl.DataFrame, fed_q: pl.DataFrame, q_from: int, q_to: int) -> pl.DataFrame:
    """OLS per state: dlog(state FMHPI, quarterly avg) = a + b * dlog(national Z.1 HPI) + e."""
    sq = (fmhpi.with_columns(q=pl.col("t") // 3).group_by("geo", "q").agg(pl.col("hpi").mean())
          .sort("geo", "q").with_columns(ds=pl.col("hpi").log().diff().over("geo")))
    nat = fed_q.select("q", dn=pl.col("hpi_nat").log().diff())
    d = sq.join(nat, on="q").filter(pl.col("q").is_between(q_from, q_to)).drop_nulls(["ds", "dn"])
    rows = []
    for geo, g in d.group_by("geo"):
        x, y = g["dn"].to_numpy(), g["ds"].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - resid.var() / y.var()
        rows.append({"geo": geo[0], "alpha": coef[0], "beta": coef[1], "resid_sd": resid.std(ddof=2),
                     "r2": r2, "n_quarters": len(y)})
    return pl.DataFrame(rows).sort("geo")
