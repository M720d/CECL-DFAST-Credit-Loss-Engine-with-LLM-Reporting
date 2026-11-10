"""Minimal client for a local Ollama server. No data leaves the machine."""
from __future__ import annotations

import re

import numpy as np
import requests


class Ollama:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = cfg["host"].rstrip("/")

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """nomic-embed-text expects task prefixes; vectors are L2-normalized for cosine/IP search."""
        prefix = "search_document: " if kind == "document" else "search_query: "
        out = []
        for i in range(0, len(texts), 32):
            r = requests.post(f"{self.host}/api/embed", timeout=600,
                              json={"model": self.cfg["embed_model"], "input": [prefix + t for t in texts[i:i + 32]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def chat(self, system: str, user: str) -> str:
        r = requests.post(f"{self.host}/api/chat", timeout=1200, json={
            "model": self.cfg["chat_model"], "stream": False, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.cfg["temperature"], "seed": self.cfg["seed"], "num_ctx": self.cfg["num_ctx"]},
        })
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # drop any reasoning trace
