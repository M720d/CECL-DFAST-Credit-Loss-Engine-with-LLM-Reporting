"""Config loader: every path and parameter comes from config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    def path(self, key: str) -> Path:
        return ROOT / self["paths"][key]


def load_config(path: Path | None = None) -> Config:
    return Config(yaml.safe_load((path or ROOT / "config.yaml").read_text()))
