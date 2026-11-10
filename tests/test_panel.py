"""Target definitions on hand-built loans whose correct answer is known."""
from __future__ import annotations

import polars as pl
import pytest

from cecl.data.panel import EVENT_CENSOR, EVENT_DEFAULT, EVENT_PREPAY, loan_outcomes, pd_snapshots
from cecl.data.schema import PERF_FIELDS

DCFG = {
    "zb_prepay": ["01"], "zb_default": ["02", "03", "09", "15"], "zb_censor": ["16", "96"],
    "dpd_default_threshold": 3, "default_definition": "ex_forbearance",
    "default_horizon_months": 12, "snapshot_every_months": 12,
    "performance_cutoff": 202212, "covid_window": [202003, 202112],
}


def months(start: int, n: int) -> list[int]:
    y, m, out = start // 100, start % 100, []
    for _ in range(n):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def loan(loan_id: str, start: int, delq: list[str], zb: dict[int, str] | None = None,
         forb: set[int] | None = None, term: int = 360) -> list[dict]:
    """One loan's monthly rows; zb/forb keyed by month index."""
    zb, forb = zb or {}, forb or set()
    rows = []
    for i, (per, d) in enumerate(zip(months(start, len(delq)), delq)):
        r = {name: None for name, _ in PERF_FIELDS}
        r.update(loan_id=loan_id, period=per, delq=d, loan_age=i, rem_months=term - i,
                 cur_upb=100_000.0, zb_code=zb.get(i), borrower_assist="F" if i in forb else None)
        rows.append(r)
    return rows


def frame(*loans: list[dict]) -> pl.LazyFrame:
    schema = dict(PERF_FIELDS) | {"vintage": pl.Int16}
    rows = [r | {"vintage": 2020} for l in loans for r in l]
    return pl.DataFrame(rows, schema=schema).lazy()


def outcomes(df: pl.LazyFrame, **over) -> dict[str, dict]:
    out = loan_outcomes(df, DCFG | over).collect()
    return {r["loan_id"]: r for r in out.iter_rows(named=True)}


def test_default_by_dpd_ra_and_zero_balance():
    df = frame(
        loan("A", 202001, ["00", "01", "02", "03", "04"]),            # 90+ DPD in month 3
        loan("B", 202001, ["00", "01", "RA"]),                          # REO acquisition
        loan("C", 202001, ["00", "01", "02"], zb={2: "03"}),           # short sale at 60 DPD
        loan("D", 202001, ["00"] * 6),                                  # performing, active
    )
    o = outcomes(df)
    assert o["A"]["event"] == EVENT_DEFAULT and o["A"]["default_period"] == 202004
    assert o["B"]["default_period"] == 202003
    assert o["C"]["default_period"] == 202003  # default-type ZB counts before 90 DPD
    assert o["D"]["event"] == EVENT_CENSOR and o["D"]["default_period"] is None


def test_forbearance_switch():
    # 90+ DPD only while in forbearance, then cures.
    rows = loan("F", 202004, ["00", "01", "02", "03", "04", "00"], forb={1, 2, 3, 4})
    assert outcomes(frame(rows))["F"]["default_period"] is None
    assert outcomes(frame(rows), default_definition="spec")["F"]["default_period"] == 202007
    # Still 90+ after forbearance ends -> default in the first non-forbearance 90+ month.
    rows2 = loan("G", 202004, ["00", "01", "02", "03", "04", "05"], forb={1, 2, 3})
    assert outcomes(frame(rows2))["G"]["default_period"] == 202008


def test_prepay_vs_matured_vs_censor():
    df = frame(
        loan("P", 202001, ["00", "00", "00"], zb={2: "01"}),                 # early payoff
        loan("M", 202001, ["00", "00", "00"], zb={2: "01"}, term=2),         # pays at maturity
        loan("X", 202001, ["00", "00"], zb={1: "96"}),                       # defect, censored
    )
    o = outcomes(df)
    assert o["P"]["event"] == EVENT_PREPAY
    assert o["M"]["event"] == EVENT_CENSOR and o["M"]["matured"]
    assert o["X"]["event"] == EVENT_CENSOR


def test_default_beats_later_prepay():
    o = outcomes(frame(loan("Q", 202001, ["00", "03", "00", "00"], zb={3: "01"})))
    assert o["Q"]["event"] == EVENT_DEFAULT and o["Q"]["default_cured"]


def snaps_for(df: pl.LazyFrame, **over) -> pl.DataFrame:
    d = DCFG | over
    o = loan_outcomes(df, d)
    return pd_snapshots(df, o, d).collect().sort(["loan_id", "loan_age"])


def test_snapshot_window_boundaries():
    # Snapshot at age 0 (201901). Default exactly 12 months later (202001) is inside the window.
    inside = ["00"] * 12 + ["03"]
    s = snaps_for(frame(loan("I", 201901, inside)))
    assert s.filter(pl.col("loan_age") == 0)["target"].item() == 1
    # Default 13 months later is outside the age-0 window.
    outside = ["00"] * 13 + ["03"]
    s = snaps_for(frame(loan("O", 201901, outside)))
    assert s.filter(pl.col("loan_age") == 0)["target"].item() == 0
    assert s.filter(pl.col("loan_age") == 12)["target"].item() == 1


def test_no_truncated_labels_near_cutoff():
    # Cutoff 202212: a snapshot in 202201 has a full window (202202..202301 > cutoff) -> excluded.
    s = snaps_for(frame(loan("T", 202101, ["00"] * 24)))
    assert s["snapshot_period"].to_list() == [202101]  # 202201 dropped, window not observed


def test_first_default_is_absorbing():
    # Defaults at age 3, cures, keeps performing: no snapshot after the default month.
    s = snaps_for(frame(loan("Z", 201801, ["00", "00", "00", "03"] + ["00"] * 30)))
    assert s["loan_age"].to_list() == [0]
    assert s["target"].to_list() == [1]


def test_loss_identity_on_real_reo_row():
    """Values from a 2007-vintage REO disposition in the raw sample file."""
    from cecl.data.qc import loss_residual
    row = pl.DataFrame({
        "zb_removal_upb": [60182.33], "delinq_accrued_int": [10273.63], "total_expenses": [17881.26],
        "net_sales_proceeds": [-15896.59], "mi_recoveries": [0.0], "non_mi_recoveries": [-763.33],
        "actual_loss": [71677.30],
    })
    assert row.select(loss_residual()).item() == pytest.approx(0.0, abs=0.01)
