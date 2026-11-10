"""Number-check guardrail (D-043): every figure in LLM-written text must come from code.

A number in the text passes if, at the precision it is displayed, it equals a value in
metrics.json or config.yaml under one of the representations a writer would use
(fraction -> percent, dollars -> $M / $B, as-is). Identifiers (SR 26-2, ASC 326, D-012,
2026Q1, Table 9, Phase 3...) and bare years are not figures and are removed first.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

IDENTIFIER_PATTERNS = [
    r"\bSR\s*\d{1,2}-\d{1,2}\b", r"\bASC\s*\d{3}(-\d{2,3})*\b", r"\bASU\s*\d{4}-\d{2}\b", r"\bTopic\s*\d{3}\b",
    r"\bD-\d{3}\b", r"\b(?:Phase|Table|Section|Figure|Appendix|Release|Part|Subpart)\s+\d+[A-Za-z]?\b",
    r"\b\d{4}\s*:?\s*Q[1-4]\b", r"\bQ[1-4]\s*\d{4}\b", r"\b\d{1,2}\s*CFR\s*\d+(\.\d+)?\b",
    r"\bp\.\s*\d+\b", r"\bpp?\.\s*\d+(-\d+)?\b", r"\[[^\]]*\]",            # citations like [SR2602a1.pdf p.5]
    r"\b[A-Z]{1,3}\d+\b",                                                  # codes like A12, S4, R48
    r"\b\d{6}\b",                                                          # YYYYMM periods
]
NUM = re.compile(r"(?<![\w.])(\$)?\s?(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?\s?(%|percent|bps|basis points|bn|billion|B\b|mn|million|M\b|×|x\b)?",
                 re.I)


@dataclass
class Figure:
    raw: str
    value: float
    decimals: int
    unit: str          # '', '%', '$', '$M', '$B', 'x'


def strip_identifiers(text: str) -> str:
    for p in IDENTIFIER_PATTERNS:
        text = re.sub(p, " ", text)
    return text


def extract_figures(text: str, year_range: tuple[int, int]) -> list[Figure]:
    out = []
    for m in NUM.finditer(strip_identifiers(text)):
        dollar, whole, frac, unit = m.group(1), m.group(2), m.group(3) or "", (m.group(4) or "").lower()
        val = float(whole.replace(",", "") + frac)
        dec = len(frac) - 1 if frac else 0
        u = {"%": "%", "percent": "%", "bps": "bps", "basis points": "bps", "bn": "$B", "billion": "$B", "b": "$B",
             "mn": "$M", "million": "$M", "m": "$M", "×": "x", "x": "x"}.get(unit, "")
        if dollar and u == "":
            u = "$"
        if u == "" and dec == 0 and "," not in whole and year_range[0] <= val <= year_range[1]:
            continue                                  # bare year
        out.append(Figure(m.group(0).strip(), val, dec, u))
    return out


def flatten_numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_numbers(v))
    return out


class Guardrail:
    def __init__(self, sources: list, year_range: tuple[int, int]):
        vals = []
        for s in sources:
            vals.extend(flatten_numbers(s))
        self.values = sorted(set(vals))
        self.year_range = year_range

    def _candidates(self, unit: str) -> list[float]:
        v = self.values
        if unit == "%":
            return [x * 100 for x in v if abs(x) <= 50] + v      # fractions as percent, or already-percent values
        if unit == "bps":
            return [x * 1e4 for x in v if abs(x) <= 1]
        if unit == "$M":
            return [x / 1e6 for x in v]
        if unit == "$B":
            return [x / 1e9 for x in v]
        return v                                                 # plain, '$', 'x'

    def matches(self, f: Figure) -> bool:
        tol = 0.5 * 10 ** (-f.decimals) + 1e-9
        for c in self._candidates(f.unit):
            if abs(abs(c) - abs(f.value)) <= tol:
                return True
        return False

    def check(self, text: str) -> tuple[bool, list[str]]:
        bad = [f.raw for f in extract_figures(text, self.year_range) if not self.matches(f)]
        return (len(bad) == 0, bad)
