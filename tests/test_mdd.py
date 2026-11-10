"""MDD drafting control loop: invented figures and fake citations are rejected and retried."""
from __future__ import annotations

import polars as pl

from cecl.llm.mdd import draft_section

GCFG = {"max_attempts": 3, "year_range": [1970, 2035]}
METRICS = {"phase2": {"gini": 0.7779}}
SEC = {"title": "PD", "decisions": [], "query": "q", "facts": {"Out-of-time Gini": "0.778"}}


class FakeIndex:
    def search(self, q, k):
        return pl.DataFrame({"doc": ["SR2602a1.pdf"], "page": [6], "text": ["Testing may include out-of-time testing."]})


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, system, user):
        self.calls += 1
        return self.replies.pop(0) if self.replies else "The out-of-time Gini is 0.95."


def test_invented_figure_is_rejected_then_fixed():
    llm = ScriptedLLM(["The out-of-time Gini is 0.81 [SR2602a1.pdf p.6].",
                       "The out-of-time Gini is 0.778, consistent with out-of-time testing [SR2602a1.pdf p.6]."])
    r = draft_section(SEC, llm, FakeIndex(), METRICS, {}, "", GCFG, 1)
    assert r.passed and r.attempts == 2


def test_fake_citation_is_rejected():
    llm = ScriptedLLM(["Gini 0.778 [SR1107a1.pdf p.3]."] * 3)
    r = draft_section(SEC, llm, FakeIndex(), METRICS, {}, "", GCFG, 1)
    assert not r.passed and r.bad_citations


def test_persistent_invention_is_rejected_after_max_attempts():
    llm = ScriptedLLM([])
    r = draft_section(SEC, llm, FakeIndex(), METRICS, {}, "", GCFG, 1)
    assert not r.passed and r.attempts == 3 and "0.95" in r.rejected_figures and llm.calls == 3
