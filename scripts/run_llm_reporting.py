"""Phase 7: local LLM reporting layer.

  uv run python scripts/run_llm_reporting.py [index] [mdd] [trace] [eval]   (default: all)

index  build the FAISS index over regulatory documents (Ollama embeddings)
mdd    draft model development document sections; number guardrail + citation check
trace  requirement -> implementation -> test -> result matrix (runs pytest for test status)
eval   52 regulatory questions: retrieval hit@k / MRR, key-fact recall, faithfulness
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cecl.config import load_config  # noqa: E402
from cecl.llm.guardrail import Guardrail, extract_figures  # noqa: E402
from cecl.llm.mdd import build_sections, draft_section  # noqa: E402
from cecl.llm.ollama import Ollama  # noqa: E402
from cecl.llm.rag import build_index, load_index  # noqa: E402
from cecl.metrics import update_metrics  # noqa: E402

T0 = time.time()
NORM = re.compile(r"[\s\-‐–]+")


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def norm(s: str) -> str:
    return NORM.sub("", s.lower())


def dotted(d, path: str):
    cur = d
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def run_mdd(cfg, ix, llm, metrics) -> dict:
    lcfg = cfg["llm"]
    decisions = (ROOT / "DECISIONS.md").read_text()
    cfg_dict = dict(cfg)
    out_dir = ROOT / "docs" / "mdd"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sec in build_sections(metrics):
        r = draft_section(sec, llm, ix, metrics, cfg_dict, decisions, lcfg["guardrail"], lcfg["rag"]["top_k"])
        results.append(r)
        log(f"MDD '{r.title}': {'PASS' if r.passed else 'REJECTED'} after {r.attempts} attempt(s)")
    lines = ["# Model Development Document: draft narrative", "",
             "> Narrative drafted by a local LLM (" + lcfg["chat_model"] + ") from `results/metrics.json`. "
             "Every figure passed the number guardrail and every citation the source check; rejected sections are "
             "listed and excluded. Numbers are produced by code only.", ""]
    for r in results:
        lines += [f"## {r.title}", ""]
        lines += [r.text if r.passed else f"*Section rejected by guardrail after {r.attempts} attempts "
                  f"(unverified figures: {r.rejected_figures}; bad citations: {r.bad_citations}).*", ""]
    (out_dir / "mdd_draft.md").write_text("\n".join(lines))
    summary = {"sections": len(results), "sections_passed": sum(r.passed for r in results),
               "first_attempt_pass": sum(r.passed and r.attempts == 1 for r in results),
               "by_section": [{"title": r.title, "passed": r.passed, "attempts": r.attempts,
                               "rejected_figures": r.rejected_figures, "bad_citations": r.bad_citations} for r in results]}
    update_metrics(cfg.path("metrics"), "phase7_mdd", summary)
    return summary


def run_trace(cfg, ix, metrics) -> dict:
    reqs = yaml.safe_load((ROOT / "eval" / "requirements.yaml").read_text())
    junit = cfg.path("processed") / "junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        cls = tc.get("classname", "").replace(".", "/")
        node = f"{cls}.py::{tc.get('name')}"
        status[node] = "fail" if (tc.find("failure") is not None or tc.find("error") is not None) else (
            "skip" if tc.find("skipped") is not None else "pass")
    rows = []
    m_all = {**metrics, "phase7": {"traceability": {"requirements": len(reqs)},
                                   "mdd": metrics.get("phase7_mdd", {})}}
    for r in reqs:
        hits = ix.search(r["query"], 3)
        ev = [f"{h['doc']} p.{h['page']}" for h in hits.iter_rows(named=True)]
        found = any(h["doc"] == r["source"] and h["page"] in r["pages"] for h in hits.iter_rows(named=True))
        def st(t):  # parametrized tests appear as name[param]; all instances must pass
            hits_ = [v for n, v in status.items() if n == t or n.startswith(t + "[")]
            return "missing" if not hits_ else ("pass" if all(v == "pass" for v in hits_) else "fail")
        tests = {t: st(t) for t in r["tests"]}
        try:
            val = dotted(m_all, r["metric"])
        except (KeyError, IndexError, TypeError):
            val = None
        rows.append({"id": r["id"], "source": f"{r['source']} p.{','.join(map(str, r['pages']))}",
                     "requirement": r["requirement"], "implementation": r["implementation"],
                     "tests": "; ".join(f"{t.split('::')[-1]} ({s})" for t, s in tests.items()),
                     "tests_pass": all(s == "pass" for s in tests.values()),
                     "metric": r["metric"], "value": val, "evidence_top3": "; ".join(ev),
                     "evidence_found": found, "assessment": r["assessment"], "note": r.get("note", "")})
    df = pl.DataFrame(rows)
    df.with_columns(pl.col("value").cast(pl.Utf8)).write_csv(cfg.path("tables") / "traceability_matrix.csv")
    md = ["# Requirement-to-test traceability matrix", "",
          "Generated by `scripts/run_llm_reporting.py trace`. Test status from pytest; value from `results/metrics.json`; "
          "evidence = top-3 passages retrieved for the requirement (✓ if the cited source page is among them).", "",
          "| ID | Source | Requirement | Implementation | Tests | Result | Evidence | Assessment |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["value"]
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        md.append(f"| {r['id']} | {r['source']} | {r['requirement']} | {r['implementation']} | {r['tests']} | "
                  f"`{r['metric']}` = {vs} | {'✓' if r['evidence_found'] else '✗'} | {r['assessment']}"
                  f"{' (' + r['note'] + ')' if r['note'] else ''} |")
    (ROOT / "docs" / "mdd" / "traceability_matrix.md").write_text("\n".join(md))
    summary = {"requirements": len(rows), "tests_all_pass": int(df["tests_pass"].sum()),
               "evidence_found": int(df["evidence_found"].sum()),
               "assessed_met": int((df["assessment"] == "Met").sum()),
               "assessed_partial": int((df["assessment"] == "Partially met").sum())}
    update_metrics(cfg.path("metrics"), "phase7_traceability", summary)
    log(f"traceability {summary}")
    return summary


ANSWER_SYSTEM = ("Answer the question using only the CONTEXT passages. Be concise (1-3 sentences). "
                 "Cite the passage tag you used, e.g. [SR2602a1.pdf p.6]. If the context does not contain "
                 "the answer, say you cannot find it in the provided documents.")


def sentences(t: str) -> list[str]:
    t = re.sub(r"\[[^\]]*\]", "", t)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 20]


def retrieval_only(ix, qs, k: int, mode: str) -> dict:
    rr = []
    for q in qs:
        hl = list(ix.search(q["question"], k, mode=mode).iter_rows(named=True))
        rank = next((h["rank"] for h in hl if h["doc"] == q["doc"] and h["page"] in q["pages"]), None)
        rr.append((rank == 1, rank is not None, 1.0 / rank if rank else 0.0))
    a = np.array(rr, dtype=float)
    return {"hit_at_1": float(a[:, 0].mean()), "hit_at_k": float(a[:, 1].mean()), "mrr": float(a[:, 2].mean())}


def run_eval(cfg, ix, llm) -> dict:
    lcfg = cfg["llm"]
    k = lcfg["rag"]["top_k"]
    tau = lcfg["eval"].get("support_threshold", 0.75)
    qs = yaml.safe_load((ROOT / lcfg["eval"]["questions"]).read_text())
    ablation = {m: retrieval_only(ix, qs, k, m) for m in ("dense", "hybrid")}
    log(f"retrieval ablation {ablation}")
    rows = []
    for q in qs:
        hits = ix.search(q["question"], k)
        hl = list(hits.iter_rows(named=True))
        rank = next((h["rank"] for h in hl if h["doc"] == q["doc"] and h["page"] in q["pages"]), None)
        doc_rank = next((h["rank"] for h in hl if h["doc"] == q["doc"]), None)
        tags = [f"[{h['doc']} p.{h['page']}]" for h in hl]
        ctx = "\n\n".join(f"{t} {h['text']}" for t, h in zip(tags, hl))
        ans = llm.chat(ANSWER_SYSTEM, f"QUESTION: {q['question']}\n\nCONTEXT:\n{ctx}")
        recall = float(np.mean([norm(f) in norm(ans) for f in q["facts"]]))
        # Faithfulness: share of answer sentences semantically supported by some context sentence
        a_s = sentences(ans)
        c_s = [s for h in hl for s in sentences(h["text"])]
        if a_s and c_s:
            sim = llm.embed(a_s, "query") @ llm.embed(c_s, "document").T
            support = float((sim.max(1) >= tau).mean())
        else:
            support = 0.0
        g = Guardrail([[f.value for f in extract_figures(ctx, (1970, 2035))]], (1970, 2035))
        num_ok, num_bad = g.check(ans)
        cited = re.findall(r"\[[^\]]*?p\.\s*\d+\]", ans)
        cit_ok = all(re.sub(r"\s+", " ", c) in tags for c in cited) and len(cited) > 0
        rows.append({"id": q["id"], "hit_at_1": rank == 1, "hit_at_k": rank is not None,
                     "reciprocal_rank": 1.0 / rank if rank else 0.0, "doc_hit_at_k": doc_rank is not None,
                     "fact_recall": recall, "sentence_support": support, "numbers_grounded": num_ok,
                     "citations_valid": cit_ok, "answer": ans, "reference": q["answer"]})
        log(f"eval {q['id']}: rank={rank} recall={recall:.2f} support={support:.2f} nums={'ok' if num_ok else num_bad}")
    df = pl.DataFrame(rows)
    df.write_csv(cfg.path("tables") / "rag_eval.csv")
    df.write_parquet(cfg.path("processed") / "rag_eval.parquet")
    by_src = (df.with_columns(src=pl.col("id").str.split("-").list.first())
              .group_by("src").agg(n=pl.len(), hit_at_k=pl.col("hit_at_k").mean(), fact_recall=pl.col("fact_recall").mean())
              .sort("src"))
    summary = {"questions": df.height, "top_k": k, "support_threshold": tau, "retrieval_mode": ix.mode,
               "retrieval_ablation": ablation,
               "retrieval_hit_at_1": float(df["hit_at_1"].mean()), "retrieval_hit_at_k": float(df["hit_at_k"].mean()),
               "retrieval_mrr": float(df["reciprocal_rank"].mean()), "doc_hit_at_k": float(df["doc_hit_at_k"].mean()),
               "answer_fact_recall_mean": float(df["fact_recall"].mean()),
               "answers_all_facts": float((df["fact_recall"] == 1.0).mean()),
               "faithfulness_sentence_support_mean": float(df["sentence_support"].mean()),
               "answers_numbers_grounded": float(df["numbers_grounded"].mean()),
               "answers_citations_valid": float(df["citations_valid"].mean()),
               "by_source": by_src.to_dicts()}
    update_metrics(cfg.path("metrics"), "phase7_eval", summary)
    log(f"eval summary { {k: v for k, v in summary.items() if k != 'by_source'} }")
    return summary


def main() -> None:
    cfg = load_config()
    steps = [a for a in sys.argv[1:] if not a.startswith("--")] or ["index", "mdd", "trace", "eval"]
    lcfg = cfg["llm"]
    rag_dir = cfg.path("processed") / "rag"
    ix = build_index(ROOT, rag_dir, lcfg) if "index" in steps or not (rag_dir / "faiss.index").exists() \
        else load_index(rag_dir, lcfg)
    log(f"index ready: {ix.chunks.height} chunks")
    llm = Ollama(lcfg)
    metrics = json.loads(cfg.path("metrics").read_text())
    if "mdd" in steps:
        run_mdd(cfg, ix, llm, metrics)
        metrics = json.loads(cfg.path("metrics").read_text())
    if "trace" in steps:
        run_trace(cfg, ix, metrics)
    if "eval" in steps:
        run_eval(cfg, ix, llm)
    log("done")


if __name__ == "__main__":
    main()
