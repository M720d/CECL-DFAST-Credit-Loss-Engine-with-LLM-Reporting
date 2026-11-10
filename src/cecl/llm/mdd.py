"""LLM drafting of model development document (MDD) sections (D-043).

The LLM never computes or changes numbers. Code builds a FACTS block per section from
metrics.json (already formatted); the LLM writes narrative around those facts, cites the
supplied regulatory passages as [doc p.N], and every output passes two checks:
  1. number guardrail: each figure must match metrics/config, or a number in the supplied passages
  2. citation check: each [doc p.N] must be one of the supplied passages
Failures are fed back and regenerated up to max_attempts; a section that still fails is rejected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from cecl.llm.guardrail import Guardrail, extract_figures

SYSTEM = (
    "You are a model documentation writer in a US bank's credit risk model development team. "
    "Write formal, concise model development document prose in Markdown (no headings, 2-4 short paragraphs). "
    "Rules: (1) Use ONLY numbers that appear in FACTS, copied exactly as written; never compute, round, or "
    "introduce any other number, count, percentage, or date. (2) Describe decisions only as stated in DECISIONS. "
    "(3) Cite regulatory expectations only from SOURCES, using the exact tag given, e.g. [SR2602a1.pdf p.6]. "
    "(4) If something is not in FACTS, DECISIONS or SOURCES, do not claim it. Do not add a conclusion section. "
    "(5) Cite a SOURCE only for what that passage literally says; never claim that a regulation endorses or "
    "requires a specific modeling choice unless the passage says so. "
    "(6) SOURCES may describe the Federal Reserve's own supervisory models; never describe those as this model's "
    "practice. (7) Follow the READING NOTES when interpreting a metric."
)


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def build_sections(m: dict) -> list[dict]:
    p1, p2, p3, p4, p5, p6 = (m[k] for k in ("phase1", "phase2", "phase3", "phase4", "phase5", "phase6"))
    sc = {r["sample"]: r for r in p2["scorecard"]["discrimination"]}
    gb = {r["sample"]: r for r in p2["lgbm"]["discrimination"]}
    t = p1["totals"]
    return [
        {"title": "Data and target definition", "decisions": ["D-001", "D-003", "D-004", "D-007"],
         "query": "data quality relevance and inputs assessment in model development",
         "notes": ["Snapshots are loan-year observations, not loans; one loan contributes several snapshots."],
         "facts": {"Loans": f"{t['loans']:,}", "Loan-months": f"{t['loan_months']:,}",
                   "Lifetime defaults (definition used)": f"{t['defaults_lifetime']:,}",
                   "12-month PD snapshots": f"{t['snapshots']:,}", "Snapshot defaults": f"{t['snapshot_defaults']:,}",
                   "Duplicate loan-months": f"{t['dup_loan_months']:,}",
                   "Liquidations reconciling to the loss identity": f"{t['loss_identity_pass']:,} of {t['dispositions_with_loss']:,}"}},
        {"title": "12-month PD scorecard", "decisions": ["D-012", "D-015", "D-016", "D-019", "D-020"],
         "query": "out-of-time testing benchmarking to alternative models",
         "notes": ["Actual-to-predicted below 1 means the model predicted more defaults than occurred (over-prediction).",
                   "A PSI below 0.1 is conventionally read as a stable population.",
                   "Recalibration was tested as an analysis; the scorecard used for ranking was not recalibrated."],
         "facts": {"Scorecard features": str(p2["scorecard"]["n_features"]),
                   "Out-of-time Gini (scorecard)": f"{sc['oot']['gini']:.3f}",
                   "Out-of-time KS (scorecard)": f"{sc['oot']['ks']:.3f}",
                   "Out-of-time Gini (LightGBM challenger)": f"{gb['oot']['gini']:.3f}",
                   "Score PSI, development vs out-of-time": f"{p2['scorecard']['psi_score_fit_vs_oot']:.3f}",
                   "Out-of-time actual-to-predicted default ratio": f"{p2['scorecard']['oot_actual_to_pred']:.2f}"}},
        {"title": "Lifetime default and prepayment hazard", "decisions": ["D-021", "D-022", "D-024", "D-025"],
         "query": "outcomes analysis back-testing model performance",
         "notes": ["Actual-to-predicted below 1 means over-prediction; above 1 means under-prediction.",
                   "COVID policy months were excluded from estimation because forbearance and stimulus broke the "
                   "historical link between unemployment and default; this is a modeling judgment, not a regulatory rule."],
         "facts": {"In-time default actual-to-predicted": f"{p3['backtest_in_time']['default_actual_to_pred']:.2f}",
                   "Out-of-time prepayment actual-to-predicted": f"{p3['backtest_oot']['prepay_actual_to_pred']:.2f}",
                   "Out-of-time default actual-to-predicted, COVID policy months": f"{p3['backtest_covid_only']['default_actual_to_pred']:.2f}",
                   "Default multiplier for unemployment +2 points": f"{p3['sensitivities_prod']['ur_plus_2pt']['default_multiplier']:.2f}",
                   "Default multiplier for house prices -20%": f"{p3['sensitivities_prod']['hpi_minus_20pct']['default_multiplier']:.2f}"}},
        {"title": "Loss given default and exposure at default", "decisions": ["D-027", "D-030", "D-031", "D-032", "D-041"],
         "query": "collateral-dependent fair value of collateral costs to sell",
         "notes": ["Predicted above actual in the out-of-time period means the LGD model is conservative there.",
                   "LGD is a model estimate, not a loan-by-loan collateral valuation."],
         "facts": {"Out-of-time severity, actual": pct(p4["test_oot"]["severity_actual"], 1),
                   "Out-of-time severity, predicted": pct(p4["test_oot"]["severity_pred"], 1),
                   "Out-of-time LGD, actual (UPB-weighted)": pct(p4["test_oot"]["lgd_actual_upb_weighted"], 1),
                   "Out-of-time LGD, predicted (UPB-weighted)": pct(p4["test_oot"]["lgd_pred_upb_weighted"], 1),
                   "EAD actual-to-scheduled ratio": f"{p4['ead']['ead_factor_upb_weighted']:.3f}"}},
        {"title": "CECL allowance", "decisions": ["D-033", "D-034", "D-035", "D-036"],
         "query": "reasonable and supportable forecast period reversion to historical loss information",
         "notes": ["The upside scenario is the mirror image of the downside around the baseline (baseline minus half the "
                   "distance to severely adverse), with unemployment floored.",
                   "Weights and the upside construction are management judgment, not regulatory requirements."],
         "facts": {"Weighted ACL, % of UPB": pct(p5["acl_weighted_pct_upb"]),
                   "ACL, model only (input reversion), % of UPB": pct(p5["acl_weighted_model_only_pct_upb"]),
                   "Baseline / downside / upside ACL, % of UPB": " / ".join(pct(p5["scenarios"][s]["acl_pct_upb"]) for s in ("baseline", "downside", "upside")),
                   "Scenario weights baseline / downside / upside": " / ".join(f"{p5['scenarios'][s]['weight']:.0%}" for s in ("baseline", "downside", "upside")),
                   "Historical annual loss rate used for reversion": pct(p5["hist_annual_loss_rate"], 3),
                   "Reasonable and supportable period (quarters)": str(p5["method"]["rs_quarters"]),
                   "Reversion period (quarters)": str(p5["method"]["reversion_quarters"])}},
        {"title": "DFAST stress projection", "decisions": ["D-038", "D-039", "D-040"],
         "query": "first lien mortgage model projected losses severely adverse scenario",
         "notes": ["The Fed benchmark aggregates bank portfolios; it is a range check, not a target.",
                   "The crisis replay is in-sample for the models: it validates aggregation, not forecasting."],
         "facts": {"Severely adverse 9-quarter loss rate": pct(p6["results"]["severely_adverse"]["loss_rate_fed_def"]),
                   "Baseline 9-quarter loss rate": pct(p6["results"]["baseline"]["loss_rate_fed_def"]),
                   "Fed 2026 all-bank first-lien loss rate (benchmark)": pct(p6["fed_benchmark"]["first_lien_loss_rate_sa"], 1),
                   "Crisis replay predicted loss rate": pct(p6["crisis_replay"]["pred_loss_rate_fed_def"]),
                   "Crisis replay actual loss rate": pct(p6["crisis_replay"]["actual_loss_rate_fed_def"]),
                   "Severely adverse LGD": pct(p6["results"]["severely_adverse"]["lgd"], 1)}},
        {"title": "Limitations and model use", "decisions": ["D-004", "D-024", "D-035", "D-040", "D-041"],
         "query": "model limitations overlays ongoing monitoring generative AI scope",
         "notes": ["Actual-to-predicted below 1 means the model predicted MORE defaults than occurred (over-prediction, "
                   "conservative), not under-prediction.",
                   "No source in SOURCES discusses forbearance; do not attribute the forbearance treatment to regulation."],
         "facts": {"Clean-loan actual-to-predicted in overlay calibration years": f"{p5['overlay_s0_actual_to_pred']:.2f}",
                   "Jan-Jun 2026 defaults, actual": f"{p5['backtest_2026h1']['defaults_actual']:,}",
                   "Jan-Jun 2026 defaults, predicted": f"{p5['backtest_2026h1']['defaults_pred']:,.0f}"}},
    ]


def decision_excerpts(decisions_md: str, ids: list[str], max_chars: int = 700) -> str:
    out = []
    for d in ids:
        mt = re.search(rf"### {d}\b(.*?)(?=\n### |\n---|\Z)", decisions_md, flags=re.S)
        if mt:
            out.append(f"{d}:{mt.group(1).strip()[:max_chars]}")
    return "\n\n".join(out)


@dataclass
class SectionResult:
    title: str
    text: str
    passed: bool
    attempts: int
    rejected_figures: list = field(default_factory=list)
    bad_citations: list = field(default_factory=list)


def draft_section(sec: dict, llm, index, metrics: dict, cfg_dict: dict, decisions_md: str, gcfg: dict, k: int) -> SectionResult:
    hits = index.search(sec["query"], k)
    tags = [f"[{r['doc']} p.{r['page']}]" for r in hits.iter_rows(named=True)]
    sources = "\n\n".join(f"{tag} {r['text'][:800]}" for tag, r in zip(tags, hits.iter_rows(named=True)))
    facts = "\n".join(f"- {a}: {b}" for a, b in sec["facts"].items())
    notes = "\n".join(f"- {n}" for n in sec.get("notes", []))
    src_numbers = [f.value for f in extract_figures(sources, tuple(gcfg["year_range"]))]
    fact_numbers = [f.value for f in extract_figures(facts + "\n" + notes, tuple(gcfg["year_range"]))]
    # Strict mode for documentation (D-043, tightened after review): only this section's FACTS/notes and its
    # SOURCES are admissible; matching against all of metrics/config let small numbers pass by coincidence.
    guard = Guardrail([src_numbers, fact_numbers], tuple(gcfg["year_range"]))
    user = (f"SECTION: {sec['title']}\n\nFACTS (copy numbers exactly):\n{facts}\n\nREADING NOTES:\n{notes}\n\nDECISIONS:\n"
            f"{decision_excerpts(decisions_md, sec['decisions'])}\n\nSOURCES:\n{sources}")
    feedback, bad_fig, bad_cit = "", [], []
    for attempt in range(1, gcfg["max_attempts"] + 1):
        text = llm.chat(SYSTEM, user + feedback)
        ok_num, bad_fig = guard.check(text)
        cited = re.findall(r"\[[^\]]*?p\.\s*\d+\]", text)
        bad_cit = [c for c in cited if re.sub(r"\s+", " ", c) not in tags]
        if ok_num and not bad_cit:
            return SectionResult(sec["title"], text, True, attempt)
        feedback = ("\n\nYOUR PREVIOUS DRAFT WAS REJECTED. "
                    + (f"These figures are not in FACTS or SOURCES: {bad_fig}. " if bad_fig else "")
                    + (f"These citations are not in SOURCES: {bad_cit}. " if bad_cit else "")
                    + "Rewrite the section using only permitted figures and citations.")
    return SectionResult(sec["title"], text, False, gcfg["max_attempts"], bad_fig, bad_cit)
