"""results/metrics.json: the only place reported numbers come from."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):  # numpy / polars scalars
        return _clean(x.item())
    return x


def update_metrics(path: Path, section: str, payload: dict) -> None:
    """Replace one top-level section, keep all others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = _clean(payload)
    data.setdefault("_meta", {})[section] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, sort_keys=False))
