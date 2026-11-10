"""Phase 8: render README.md and the model development document from metrics.json.

  docs/templates/README.tmpl.md -> README.md
  docs/templates/mdd.tmpl.md    -> docs/mdd/model_development_document.md (+ .pdf via pandoc + headless Chrome)
Also draws results/figures/pipeline_diagram.png and acl_by_scenario.png from metrics.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.reporting.render import render  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ---------------------------------------------------------------- figures
def pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    C = {"data": "#e8eef5", "model": "#eaf3ea", "use": "#fbf1e3", "rep": "#f3eaf5"}

    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=col, ec="#4a5a6a", lw=1))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=9.5, weight="bold", color="#1b2a3a")
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top", fontsize=7.6, color="#33424f", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1, color="#5a6a7a"))

    box(0.2, 4.3, 2.6, 1.7, "Inputs", "Freddie Mac SFLLD sample\n(manual, R48)\nFed DFAST CSVs, FRED,\nFMHPI, regulatory PDFs", C["data"])
    box(3.3, 4.3, 2.8, 1.7, "1. Data pipeline", "strict-schema parquet\nloan-month panel (76M)\ndefault / prepay / cure\nQC + 12m PD snapshots", C["data"])
    box(6.6, 4.9, 2.6, 1.1, "2. PD scorecard", "WoE logit + LightGBM\nchallenger, OOT tests", C["model"])
    box(6.6, 3.55, 2.6, 1.1, "3. Lifetime hazard", "default/prepay competing\nrisks, Fed macro drivers", C["model"])
    box(6.6, 2.2, 2.6, 1.1, "4. LGD / EAD", "cure x fractional-logit\nseverity; amortization EAD", C["model"])
    box(9.7, 4.0, 3.1, 1.6, "5. CECL allowance", "monthly engine to maturity\n3 weighted scenarios\nR&S + reversion, overlays", C["use"])
    box(9.7, 2.0, 3.1, 1.6, "6. DFAST", "9-quarter Fed paths\ncrisis replay 2008-10\nFed benchmark", C["use"])
    box(3.3, 0.3, 2.8, 1.5, "results/metrics.json", "single source of every\nreported number\n(+ tables, figures)", C["rep"])
    box(6.6, 0.3, 2.6, 1.5, "7. LLM reporting", "local RAG, MDD drafts,\nnumber guardrail,\ntraceability, eval", C["rep"])
    box(9.7, 0.3, 3.1, 1.5, "8. Packaging", "README + model\ndevelopment document\nrendered from metrics", C["rep"])
    arrow(2.8, 5.15, 3.3, 5.15)
    for y in (5.45, 4.1, 2.75):
        arrow(6.1, 5.0, 6.6, y)
    arrow(9.2, 4.1, 9.7, 4.8); arrow(9.2, 2.75, 9.7, 4.4); arrow(9.2, 2.75, 9.7, 2.8); arrow(9.2, 4.1, 9.7, 3.0)
    arrow(4.7, 4.3, 4.7, 1.8)
    arrow(6.1, 1.05, 6.6, 1.05); arrow(9.2, 1.05, 9.7, 1.05)
    ax.text(4.8, 2.9, "every phase\nwrites metrics", fontsize=7.5, color="#5a6a7a", ha="left")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def acl_chart(m: dict, path: Path) -> None:
    p5 = m["phase5"]
    names = ["baseline", "downside", "upside"]
    upb = p5["portfolio"]["upb"]
    with_rev = [p5["scenarios"][n]["acl_pct_upb"] * 100 for n in names]
    model = [p5["scenarios"][n]["acl_model_input_reversion"] / upb * 100 for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], model, 0.4, color="#9fb6cc", label="Model only (input reversion)")
    ax.bar([i + 0.2 for i in x], with_rev, 0.4, color="#1f4e79", label="Reversion to historical loss rate (main)")
    ax.axhline(p5["acl_weighted_pct_upb"] * 100, ls="--", color="#c0504d",
               label=f"Probability-weighted ACL ({p5['acl_weighted_pct_upb']:.2%})")
    ax.set_xticks(list(x), [f"{n}\n(weight {p5['scenarios'][n]['weight']:.0%})" for n in names])
    ax.set_ylabel("ACL, % of UPB"); ax.set_title(f"CECL allowance by scenario, as of {p5['as_of']}")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- tables
def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "|".join(["---"] + ["---:"] * (len(header) - 1))
    return "\n".join(["| " + " | ".join(header) + " |", "|" + align + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def make_tables(m: dict) -> dict:
    p2, p3, p4, p5, p6, p7e = m["phase2"], m["phase3"], m["phase4"], m["phase5"], m["phase6"], m["phase7_eval"]

    def vintage(c):
        rows = [[str(r["vintage"]), f"{r['loans']:,}", f"{r['default_lifetime_to_date']:.2%}", f"{r['default_lifetime_spec_def']:.2%}",
                 f"{r['snapshot_12m_default_rate'] or 0:.2%}"] for r in m["phase1"]["by_vintage"] if r["vintage"] % 3 == 1 or r["vintage"] in (2007, 2008)]
        return md_table(["Vintage", "Loans", "Lifetime default (used)", "Lifetime default (strict 90+ DPD)", "12m snapshot PD"], rows)

    def pd_disc(c):
        sc = {r["sample"]: r for r in p2["scorecard"]["discrimination"]}
        gb = {r["sample"]: r for r in p2["lgbm"]["discrimination"]}
        rows = [[k, f"{sc[k]['n']:,}", f"{sc[k]['defaults']:,}", f"{sc[k]['gini']:.3f}", f"{sc[k]['ks']:.3f}",
                 f"{gb[k]['gini']:.3f}", f"{sc[k]['default_rate'] / sc[k]['mean_pd']:.2f}"] for k in ("fit", "holdout", "oot")]
        return md_table(["Sample", "Snapshots", "Defaults", "Scorecard Gini", "Scorecard KS", "LightGBM Gini", "Actual/predicted (scorecard)"], rows)

    def hz_year(c):
        rows = [[str(r["period"]), f"{r['d_act']:,}", f"{r['d_hat']:,.0f}", f"{r['d_ratio']:.2f}", f"{r['p_ratio']:.2f}"] for r in p3["backtest_oot_by_year"]]
        return md_table(["Year", "Defaults actual", "Defaults predicted", "Default A/P", "Prepay A/P"], rows)

    def lgd_year(c):
        rows = [[str(r["default_year"]), f"{r['cure_rate_actual']:.0%}", f"{r['cure_rate_pred']:.0%}", f"{r['severity_actual']:.0%}",
                 f"{r['severity_pred']:.0%}", f"{r['lgd_actual_upb_weighted']:.1%}", f"{r['lgd_pred_upb_weighted']:.1%}"]
                for r in p4["by_default_year"] if r["default_year"] in (2003, 2006, 2008, 2009, 2010, 2012, 2015, 2017, 2019, 2021)]
        return md_table(["Default year", "Cure act.", "Cure pred.", "Severity act.", "Severity pred.", "LGD act.", "LGD pred."], rows)

    def cecl_scn(c):
        rows = [[n, f"{p5['scenarios'][n]['weight']:.0%}", f"{p5['scenarios'][n]['acl_pct_upb']:.2%}",
                 f"{p5['scenarios'][n]['acl_model_input_reversion'] / p5['portfolio']['upb']:.2%}",
                 f"{p5['scenarios'][n]['lifetime_pd_upb_wtd']:.1%}", f"{p5['scenarios'][n]['avg_lgd_on_defaults']:.1%}",
                 f"{p5['scenarios'][n]['expected_wal_years']:.1f}"] for n in ("baseline", "downside", "upside")]
        rows.append(["**weighted**", "100%", f"**{p5['acl_weighted_pct_upb']:.2%}**", f"{p5['acl_weighted_model_only_pct_upb']:.2%}", "", "", ""])
        return md_table(["Scenario", "Weight", "ACL (main)", "ACL (model only)", "Lifetime PD", "Avg LGD", "WAL (yrs)"], rows)

    def sens(c):
        return md_table(["Case", "ACL % of UPB", "vs main"], [[r["case"], f"{r['acl_pct_upb']:.2%}", f"{r['vs_main']:+.0%}"] for r in p5["sensitivity"]])

    def segs(c):
        bt = {r["segment"]: r for r in p5["backtest_2026h1_by_segment"]}
        rows = []
        for r in p5["acl_by_segment"]:
            b = bt.get(r["segment"])
            rows.append([r["segment"], f"{r['loans']:,}", f"{r['acl_pct']:.2%}", f"{b['def_actual']:,}" if b else "n/a",
                         f"{b['def_pred']:,.0f}" if b else "n/a (PD = 1)"])
        return md_table(["Segment at as-of", "Loans", "ACL % of UPB", "Jan-Jun 2026 defaults, actual", "predicted"], rows)

    def dfast(c):
        r = p6["results"]
        return md_table(["", "Baseline", "Severely adverse", "Severely adverse, no servicing regime"], [
            ["9-quarter loss rate (Fed definition)", f"{r['baseline']['loss_rate_fed_def']:.2%}", f"{r['severely_adverse']['loss_rate_fed_def']:.2%}",
             f"{p6['downturn_lgd']['sa_without_servicing_regime']['loss_rate_fed_def']:.2%}"],
            ["LGD on defaults", f"{r['baseline']['lgd']:.1%}", f"{r['severely_adverse']['lgd']:.1%}",
             f"{p6['downturn_lgd']['sa_without_servicing_regime']['lgd']:.1%}"],
            ["Cumulative default rate", f"{r['baseline']['cum_default_rate']:.1%}", f"{r['severely_adverse']['cum_default_rate']:.1%}", ""]])

    def trace(c):
        t = pl.read_csv(ROOT / "results" / "tables" / "traceability_matrix.csv")
        rows = [[r["id"], r["source"], r["requirement"], r["implementation"], "pass" if r["tests_pass"] else "FAIL",
                 "yes" if r["evidence_found"] else "no", r["assessment"]] for r in t.iter_rows(named=True)]
        return md_table(["ID", "Source", "Requirement", "Implementation", "Tests", "Evidence retrieved", "Assessment"], rows,
                        "---|---|---|---|---|---|---")

    def decisions(c):
        txt = (ROOT / "DECISIONS.md").read_text()
        rows = [[d, t.strip()] for d, t in re.findall(r"^### (D-\d{3}) (.+)$", txt, flags=re.M)]
        return md_table(["ID", "Decision"], rows, "---|---")

    def eval_tbl(c):
        a = p7e["retrieval_ablation"]
        return md_table(["Metric", "Value"], [
            ["Questions (facts verified on gold pages)", str(p7e["questions"])],
            [f"Retrieval hit@1 / hit@{p7e['top_k']} / MRR (dense)", f"{a['dense']['hit_at_1']:.0%} / {a['dense']['hit_at_k']:.0%} / {a['dense']['mrr']:.2f}"],
            [f"Retrieval hit@1 / hit@{p7e['top_k']} / MRR (hybrid BM25, rejected)", f"{a['hybrid']['hit_at_1']:.0%} / {a['hybrid']['hit_at_k']:.0%} / {a['hybrid']['mrr']:.2f}"],
            ["Key-fact recall (mean / all facts)", f"{p7e['answer_fact_recall_mean']:.0%} / {p7e['answers_all_facts']:.0%}"],
            ["Sentence support (faithfulness proxy)", f"{p7e['faithfulness_sentence_support_mean']:.0%}"],
            ["Answers with numbers grounded in context", f"{p7e['answers_numbers_grounded']:.0%}"],
            ["Answers with valid citations", f"{p7e['answers_citations_valid']:.0%}"]], "---|---:")

    return {"vintage_table": vintage, "pd_table": pd_disc, "hz_year_table": hz_year, "lgd_year_table": lgd_year,
            "cecl_table": cecl_scn, "sens_table": sens, "seg_table": segs, "dfast_table": dfast,
            "trace_table": trace, "decisions_table": decisions, "eval_table": eval_tbl}


def context(m: dict) -> dict:
    c = {"m": m}
    c.update({k: m[k] for k in m if not k.startswith("_")})
    c["sc"] = {r["sample"]: r for r in m["phase2"]["scorecard"]["discrimination"]}
    c["gb"] = {r["sample"]: r for r in m["phase2"]["lgbm"]["discrimination"]}
    import xml.etree.ElementTree as ET
    junit = ROOT / "data" / "processed" / "junit.xml"
    if junit.exists():
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        fails = sum(1 for t in cases if t.find("failure") is not None or t.find("error") is not None)
        c["tests"] = {"total": len(cases), "failed": fails, "passed": len(cases) - fails}
    else:
        c["tests"] = {"total": 0, "failed": 0, "passed": 0}
    c["cfg"] = dict(load_config())
    c["n_decisions"] = len(re.findall(r"^### D-\d{3}", (ROOT / "DECISIONS.md").read_text(), flags=re.M))
    return c


def to_pdf(md_path: Path, css: Path, title: str) -> Path:
    html, pdf = md_path.with_suffix(".html"), md_path.with_suffix(".pdf")
    shutil.copy(css, md_path.parent / css.name)
    subprocess.run(["pandoc", md_path.name, "-s", "--toc", "--toc-depth=2", "-c", css.name, "--metadata", f"title={title}",
                    "--resource-path", str(ROOT), "-o", html.name], check=True, cwd=md_path.parent)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pdf


def main() -> None:
    cfg = load_config()
    m = json.loads(cfg.path("metrics").read_text())
    figs = cfg.path("figures")
    pipeline_diagram(figs / "pipeline_diagram.png")
    acl_chart(m, figs / "acl_by_scenario.png")
    ctx, tables = context(m), make_tables(m)
    tdir = ROOT / "docs" / "templates"
    (ROOT / "README.md").write_text(render((tdir / "README.tmpl.md").read_text(), ctx, tables))
    mdd = ROOT / "docs" / "mdd" / "model_development_document.md"
    mdd.write_text(render((tdir / "mdd.tmpl.md").read_text(), ctx, tables))
    pdf = to_pdf(mdd, tdir / "mdd.css", "Model Development Document: Mortgage CECL & DFAST Credit Loss Engine")
    print(f"wrote README.md, {mdd.relative_to(ROOT)}, {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
