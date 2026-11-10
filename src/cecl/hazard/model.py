"""Discrete-time competing-risk hazard: multinomial logit (survive / default / prepay).

h_k(t | x) = exp(x b_k) / (1 + exp(x b_1) + exp(x b_2)),  k in {1 default, 2 prepay}

Estimated on a choice-based sample (all event months, a share r of survive months).
Under that sampling only the outcome-specific constants are shifted, by ln(1/r)
(Manski & Lerman 1977; Prentice & Pyke 1979), so constants are corrected by + ln(r).
The same design() is used for fitting, backtesting and scenario projection (Phase 5/6),
so the model can never see different variable definitions in different uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm


def hinge(x: pl.Expr, k: float) -> pl.Expr:
    return pl.max_horizontal(x - k, pl.lit(0.0))


def design_exprs(hcfg: dict) -> dict[str, pl.Expr]:
    """Named regressors. Missing values are filled and flagged, never dropped."""
    k = hcfg["knots"]
    age = pl.col("loan_age").cast(pl.Float64) / 12.0          # years
    fico = pl.col("fico").cast(pl.Float64)
    cltv = pl.col("cltv").cast(pl.Float64)
    dti = pl.col("dti").cast(pl.Float64)
    m = pl.col("mtm_ltv").cast(pl.Float64).fill_null(70.0).clip(0, 250) / 100.0
    inc = pl.col("incentive").cast(pl.Float64).fill_null(0.0).clip(-5, 5)
    ex = {
        "age": age,
        **{f"age_gt{a}y": hinge(age, a) for a in k["age_years"]},
        "fico": (fico.fill_null(720.0) - 700.0) / 100.0,
        "fico_missing": fico.is_null().cast(pl.Float64),
        **{f"fico_gt{f}": hinge(fico.fill_null(720.0), f) / 100.0 for f in k["fico"]},
        "cltv": cltv.fill_null(80.0) / 100.0,
        "dti": dti.fill_null(35.0) / 100.0,
        "dti_missing": dti.is_null().cast(pl.Float64),
        "sato": pl.col("sato").cast(pl.Float64).fill_null(0.0).clip(-3, 3),
        "log_upb": pl.col("orig_upb").cast(pl.Float64).log() - 12.0,
        "mtm_ltv": m,
        **{f"mtm_gt{int(x * 100)}": hinge(m, x) for x in k["mtm_ltv"]},
        "incentive": inc,
        **{f"inc_gt{str(x).replace('-', 'm').replace('.', 'p')}": hinge(inc, x) for x in k["incentive"]},
        "ur": pl.col("ur").cast(pl.Float64),
        "ur_yoy": pl.col("ur_yoy").cast(pl.Float64),
        # forb_l1 deliberately excluded (D-025): not projectable, meaning changed in COVID.
        "purp_cashout": (pl.col("purpose") == "C").cast(pl.Float64).fill_null(0.0),
        "purp_refi": pl.col("purpose").is_in(["N", "R"]).cast(pl.Float64).fill_null(0.0),
        "occ_invest": (pl.col("occupancy") == "I").cast(pl.Float64).fill_null(0.0),
        "occ_second": (pl.col("occupancy") == "S").cast(pl.Float64).fill_null(0.0),
        "single_borrower": (pl.col("n_borrowers") == 1).cast(pl.Float64).fill_null(0.0),
    }
    return ex


def design(df: pl.DataFrame | pl.LazyFrame, hcfg: dict) -> tuple[np.ndarray, list[str]]:
    ex = design_exprs(hcfg)
    names = ["const", *ex]
    X = df.lazy().select(**ex).collect().to_numpy().astype(np.float64)
    return np.column_stack([np.ones(len(X)), X]), names


@dataclass
class HazardModel:
    names: list[str]
    beta: np.ndarray          # (n_features, 2): column 0 default, column 1 prepay; constants corrected
    se: np.ndarray            # same shape, loan-clustered
    pvalues: np.ndarray
    survive_rate: float
    n_obs: int
    n_events: tuple[int, int]

    def hazards(self, X: np.ndarray) -> np.ndarray:
        """(n, 2) monthly hazards [default, prepay]."""
        eta = X @ self.beta
        m = np.maximum(eta.max(axis=1), 0.0)[:, None]          # stable softmax with the 0 baseline
        e = np.exp(eta - m)
        return e / (np.exp(-m) + e.sum(axis=1, keepdims=True))

    def table(self) -> pl.DataFrame:
        return pl.DataFrame({
            "feature": self.names,
            "beta_default": self.beta[:, 0], "se_default": self.se[:, 0], "p_default": self.pvalues[:, 0],
            "beta_prepay": self.beta[:, 1], "se_prepay": self.se[:, 1], "p_prepay": self.pvalues[:, 1],
        })


def fit_hazard(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], r: float) -> HazardModel:
    res = sm.MNLogit(y, X).fit(method="newton", maxiter=50, disp=0,
                               cov_type="cluster", cov_kwds={"groups": groups})
    beta = np.asarray(res.params).reshape(len(names), 2).copy()
    beta[0, :] += np.log(r)            # undo choice-based sampling on the constants
    return HazardModel(names, beta, np.asarray(res.bse).reshape(len(names), 2),
                       np.asarray(res.pvalues).reshape(len(names), 2), r, len(y),
                       (int((y == 1).sum()), int((y == 2).sum())))
