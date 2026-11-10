"""Regulatory-document RAG: extract -> chunk (page-aware) -> embed -> FAISS inner-product index."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from bs4 import BeautifulSoup
from pypdf import PdfReader

from cecl.llm.ollama import Ollama


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """(page number, text). HTML counts as a single page."""
    if path.suffix.lower() in (".htm", ".html"):
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        # Main content only: the Fed site wraps letters in navigation menus that pollute retrieval
        main = soup.find(id="article") or soup.find(id="content") or soup.find("main") or soup
        for tag in main.find_all(["nav", "script", "style", "header", "footer"]):
            tag.decompose()
        return [(1, main.get_text(" "))]
    reader = PdfReader(str(path))
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def clean(t: str) -> str:
    t = t.replace("­", "").replace("‐", "-")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)      # de-hyphenate line breaks
    return re.sub(r"\s+", " ", t).strip()


def chunk(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):                           # end on a sentence boundary when possible
            k = text.rfind(". ", i + size // 2, j)
            j = k + 1 if k > 0 else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


def build_chunks(root: Path, rcfg: dict) -> pl.DataFrame:
    rows = []
    for src in rcfg["sources"]:
        for p in sorted((root / src).glob("*")):
            if p.suffix.lower() not in (".pdf", ".htm", ".html") or p.name in rcfg["exclude"]:
                continue
            for page, txt in extract_pages(p):
                for c in chunk(clean(txt), rcfg["chunk_chars"], rcfg["overlap_chars"]):
                    if len(c) > 80:
                        rows.append({"doc": p.name, "page": page, "text": c})
    df = pl.DataFrame(rows)
    return df.with_row_index("chunk_id")


TOKEN = re.compile(r"[a-z0-9]+")


class BM25:
    """Okapi BM25 over chunk text (keyword channel of hybrid retrieval, D-046)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [TOKEN.findall(t.lower()) for t in texts]
        self.k1, self.b = k1, b
        self.avgdl = float(np.mean([len(d) for d in self.docs]))
        df: dict[str, int] = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self.docs)
        self.idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [{w: d.count(w) for w in set(d)} for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = [w for w in TOKEN.findall(query.lower()) if w in self.idf]
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out


@dataclass
class Index:
    chunks: pl.DataFrame
    index: faiss.Index
    client: Ollama
    mode: str = "hybrid"              # 'dense' or 'hybrid' (dense + BM25, reciprocal-rank fusion)
    rrf_k: int = 60
    _bm25: BM25 | None = None

    def search(self, query: str, k: int, mode: str | None = None) -> pl.DataFrame:
        mode = mode or self.mode
        q = self.client.embed([query], "query")
        depth = k if mode == "dense" else min(100, self.chunks.height)
        scores, ids = self.index.search(q, depth)
        if mode == "dense":
            sel, sc = ids[0].tolist(), scores[0]
        else:
            if self._bm25 is None:
                self._bm25 = BM25(self.chunks["text"].to_list())
            bm = self._bm25.scores(query)
            bm_rank = np.argsort(-bm)[:depth]
            fused: dict[int, float] = {}
            for r, i in enumerate(ids[0].tolist()):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
            for r, i in enumerate(bm_rank.tolist()):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
            top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            sel, sc = [i for i, _ in top], np.array([s for _, s in top])
        return self.chunks[sel].with_columns(score=pl.Series(sc), rank=pl.Series(range(1, len(sel) + 1)))


def build_index(root: Path, out_dir: Path, llm_cfg: dict) -> Index:
    client = Ollama(llm_cfg)
    ch = build_chunks(root, llm_cfg["rag"])
    vecs = client.embed(ch["text"].to_list(), "document")
    idx = faiss.IndexFlatIP(vecs.shape[1])          # exact search: small corpus, no approximation error
    idx.add(vecs)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(out_dir / "faiss.index"))
    ch.write_parquet(out_dir / "chunks.parquet")
    (out_dir / "meta.json").write_text(json.dumps({"embed_model": llm_cfg["embed_model"], "n_chunks": ch.height,
                                                   "dim": int(vecs.shape[1])}))
    return Index(ch, idx, client, mode=llm_cfg["rag"].get("mode", "hybrid"))


def load_index(out_dir: Path, llm_cfg: dict) -> Index:
    return Index(pl.read_parquet(out_dir / "chunks.parquet"), faiss.read_index(str(out_dir / "faiss.index")),
                 Ollama(llm_cfg), mode=llm_cfg["rag"].get("mode", "hybrid"))
