"""EAD for amortizing mortgages: scheduled balance from the standard annuity formula."""
from __future__ import annotations

import numpy as np


def scheduled_balance(b0, annual_rate_pct, n_remaining, k):
    """Balance after k scheduled payments on a level-payment loan.

    b0: current balance; annual_rate_pct: note rate in percent; n_remaining: payments left.
    B_k = B0 (1+r)^k - P ((1+r)^k - 1) / r,  P = B0 r / (1 - (1+r)^-n).  Zero-rate handled.
    """
    b0, rate, n, k = map(lambda a: np.asarray(a, dtype=float), (b0, annual_rate_pct, n_remaining, k))
    r = rate / 1200.0
    k = np.minimum(k, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 + r) ** k
        pay = b0 * r / (1 - (1 + r) ** (-n))
        bal = b0 * g - pay * (g - 1) / r
    linear = b0 * (1 - k / np.where(n > 0, n, 1))
    out = np.where(r == 0, linear, bal)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, None)
