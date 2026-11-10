"""DFAST quarterly aggregation and Fed-definition loss rate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_dfast import quarterly  # noqa: E402


def test_quarterly_aggregation_and_fed_loss_rate():
    class P:
        month_el = np.arange(1, 28, dtype=float)[:, None] * np.array([[1.0, 0.0]])
        month_def_upb = np.full((27, 2), 10.0)
        month_exposure = np.full((27, 2), 500.0)
    q = quarterly(P, 9)
    assert q["loss"][0] == pytest.approx(1 + 2 + 3)
    assert q["loss"].sum() == pytest.approx(sum(range(1, 28)))
    assert q["avg_balance"] == pytest.approx(1000.0)                 # two segments x 500
    assert q["loss_rate_fed_def"] == pytest.approx(sum(range(1, 28)) / 1000.0)
    assert q["lgd"] == pytest.approx(sum(range(1, 28)) / (27 * 20.0))
