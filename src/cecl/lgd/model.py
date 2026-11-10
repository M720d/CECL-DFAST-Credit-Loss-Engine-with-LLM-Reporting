"""Two-part LGD: P(cure | default) x severity given liquidation.

  E[LGD] = (1 - P(cure)) * E[severity | liquidation]
Cure: logistic regression. Severity: fractional logit (Papke & Wooldridge 1996), a
binomial-family GLM with logit link on a [0,1] response, with robust (sandwich)
standard errors because only the mean is assumed correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm

from cecl.hazard.model import hinge


def _common(lcfg: dict, state_timeline: dict[str, float]) -> dict[str, pl.Expr]:
    m = pl.col("mtm_ltv_default").cast(pl.Float64).fill_null(80.0).clip(0, 250) / 100.0
    tl_default = float(np.median(list(state_timeline.values())))
    return {
        "mtm_ltv": m,
        "hpi_fwd": pl.col("hpi_fwd").cast(pl.Float64).fill_null(0.0).clip(-0.5, 0.5),
        **{f"mtm_gt{int(k * 100)}": hinge(m, k) for k in lcfg["mtm_knots"]},
        "fico": (pl.col("fico").cast(pl.Float64).fill_null(700.0) - 700.0) / 100.0,
        # Real loan size: UPB deflated by national HPI (index ~300 in 2025). Nominal size was an era
        # proxy: loans grow with house prices, which made LGD drift down mechanically (D-041).
        "log_upb_real": (pl.col("upb_default").cast(pl.Float64).clip(1000, None)
                         / pl.col("hpi_us_d").cast(pl.Float64) * 300.0).log() - 12.0,
        # Data-driven foreclosure timeline: median months default->disposition in the state,
        # estimated on training liquidations only (proxy for judicial process and carrying costs).
        "state_timeline": pl.col("state").replace_strict(state_timeline, default=tl_default, return_dtype=pl.Float64) / 12.0,
        "occ_invest": (pl.col("occupancy") == "I").cast(pl.Float64).fill_null(0.0),
        "purp_cashout": (pl.col("purpose") == "C").cast(pl.Float64).fill_null(0.0),
    }


def severity_exprs(lcfg: dict, st: dict) -> dict[str, pl.Expr]:
    return {
        **_common(lcfg, st),
        "mi_pct": pl.col("mi_pct").cast(pl.Float64).fill_null(0.0) / 100.0,
        "condo": (pl.col("prop_type") == "CO").cast(pl.Float64).fill_null(0.0),
        "manufactured": (pl.col("prop_type") == "MH").cast(pl.Float64).fill_null(0.0),
        "multi_unit": (pl.col("units").cast(pl.Float64).fill_null(1.0) > 1).cast(pl.Float64),
    }


def cure_exprs(lcfg: dict, st: dict) -> dict[str, pl.Expr]:
    return {
        **_common(lcfg, st),
        "dti": pl.col("dti").cast(pl.Float64).fill_null(35.0) / 100.0,
        "age_years": pl.col("age_default").cast(pl.Float64).fill_null(36.0) / 12.0,
        # Unemployment deliberately excluded: wrong sign, confounded with 2009-10 modification
        # programs (D-031). LGD's macro channel runs through house prices.
        "servicing_post_regime": (pl.col("default_year") >= lcfg["servicing_regime_start_year"]).cast(pl.Float64),
        "single_borrower": (pl.col("n_borrowers") == 1).cast(pl.Float64).fill_null(0.0),
    }


def matrix(df: pl.DataFrame, ex: dict[str, pl.Expr]) -> tuple[np.ndarray, list[str]]:
    X = df.select(**ex).to_numpy().astype(float)
    return np.column_stack([np.ones(len(X)), X]), ["const", *ex]


@dataclass
class LGDModel:
    lcfg: dict
    state_timeline: dict
    cure: object          # statsmodels LogitResults
    sev: object           # statsmodels GLMResults
    cure_names: list[str]
    sev_names: list[str]

    def p_cure(self, df: pl.DataFrame) -> np.ndarray:
        X, _ = matrix(df, cure_exprs(self.lcfg, self.state_timeline))
        return np.asarray(self.cure.predict(X))

    def severity(self, df: pl.DataFrame) -> np.ndarray:
        X, _ = matrix(df, severity_exprs(self.lcfg, self.state_timeline))
        return np.asarray(self.sev.predict(X))

    def lgd(self, df: pl.DataFrame) -> np.ndarray:
        return (1.0 - self.p_cure(df)) * self.severity(df)

    def tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        def t(res, names):
            return pl.DataFrame({"feature": names, "coef": np.asarray(res.params), "se_robust": np.asarray(res.bse),
                                 "p_value": np.asarray(res.pvalues)})
        return t(self.cure, self.cure_names), t(self.sev, self.sev_names)


def state_timelines(liq: pl.DataFrame) -> dict[str, float]:
    return dict(liq.group_by("state").agg(pl.col("months_to_disposition").median()).iter_rows())


def fit_lgd(train: pl.DataFrame, lcfg: dict) -> LGDModel:
    res = train.filter(pl.col("resolution").is_in(["liquidated", "cured"]))
    liq = res.filter(pl.col("resolution") == "liquidated")
    st = state_timelines(liq)
    Xc, cn = matrix(res, cure_exprs(lcfg, st))
    yc = (res["resolution"] == "cured").to_numpy().astype(int)
    cure = sm.Logit(yc, Xc).fit(disp=0, cov_type="HC1")
    Xs, sn = matrix(liq, severity_exprs(lcfg, st))
    sev = sm.GLM(liq["severity"].to_numpy(), Xs, family=sm.families.Binomial()).fit(cov_type="HC1")
    return LGDModel(lcfg, st, cure, sev, cn, sn)
