"""Discrimination, calibration and stability metrics (no model-specific code)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return 2.0 * roc_auc_score(y, p) - 1.0


def ks(y: np.ndarray, p: np.ndarray) -> float:
    """Max distance between cumulative score distributions of defaults and non-defaults."""
    order = np.argsort(-p, kind="mergesort")
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (len(y) - y.sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Predicted vs actual default rate by predicted-PD quantile bucket (1 = lowest risk)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bucket").agg(n=("y", "size"), defaults=("y", "sum"),
                                 pd_pred=("p", "mean"), dr_actual=("y", "mean"),
                                 p_min=("p", "min"), p_max=("p", "max")).reset_index()
    t["ratio_actual_to_pred"] = t["dr_actual"] / t["pd_pred"]
    return t


def psi(expected: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    """Population stability index with bins fixed on the development sample."""
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_quantile(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(expected, actual, edges)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True).reindex(e.index).fillna(0)
    e, a = e.clip(lower=1e-6), a.clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def gini_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    """95% CI by resampling whole loans (snapshots of one loan are correlated)."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    idx_by_group = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[idx_by_group], np.arange(len(codes)))
    ends = np.append(starts[1:], len(inv))
    out = []
    for _ in range(reps):
        g = rng.integers(0, len(codes), len(codes))
        rows = np.concatenate([idx_by_group[starts[k]:ends[k]] for k in g])
        out.append(gini(y[rows], p[rows]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
