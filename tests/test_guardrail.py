"""Number-check guardrail: passes figures that come from metrics, fails anything else."""
from __future__ import annotations

from cecl.llm.guardrail import Guardrail, extract_figures

METRICS = {"phase5": {"acl_weighted_pct_upb": 0.007020478, "acl_weighted": 573553523.7,
                      "portfolio": {"loans": 347613}},
           "phase2": {"gini": 0.7779, "psi": 0.072}, "hazard": {"ur": 4.6}}
G = Guardrail([METRICS], (1970, 2035))


def test_passes_correct_figures_at_displayed_precision():
    ok, bad = G.check("The weighted ACL is 0.70% of UPB ($573.6 million; $0.57 billion) on 347,613 loans; "
                      "out-of-time Gini 0.778, PSI 0.07, unemployment 4.6%.")
    assert ok, bad


def test_fails_invented_or_misrounded_figures():
    ok, bad = G.check("The weighted ACL is 0.75% of UPB and Gini is 0.80.")
    assert not ok and set(bad) == {"0.75%", "0.80"}


def test_identifiers_and_years_are_not_figures():
    text = ("Per SR 26-2 and ASC 326 (ASU 2016-13), see D-024, Table 9, Phase 3 and 2026Q1 [SR2602a1.pdf p.5]; "
            "data from 1999 to 2025, period 202512.")
    assert extract_figures(text, (1970, 2035)) == []


def test_units_parsed():
    figs = {f.raw: (f.value, f.decimals, f.unit) for f in extract_figures("0.70% and $1.5 billion and 25×", (1970, 2035))}
    assert figs["0.70%"] == (0.70, 2, "%")
    assert figs["$1.5 billion"] == (1.5, 1, "$B")
    assert figs["25×"] == (25.0, 0, "x")
