"""Download every public (no-login) input listed in config.yaml.

Freddie Mac loan-level sample files need a Clarity login and are downloaded by hand
(see README). Files that already exist are skipped, so the script is safe to re-run.
A manifest with SHA-256 hashes is written next to the files for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Default requests User-Agent: FRED's fredgraph endpoint rejects custom UA strings.
HEADERS: dict[str, str] = {}


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        status = "cached"
    else:
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        status = "downloaded"
        time.sleep(0.5)  # be polite to public servers
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"file": str(dest.relative_to(ROOT)), "url": url, "sha256": sha, "status": status}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    p, d = cfg["paths"], cfg["downloads"]
    jobs: list[tuple[str, Path]] = []
    jobs += [(u, ROOT / p["raw_freddie"] / "docs" / n) for n, u in d["freddie_docs"].items()]
    jobs += [(u, ROOT / p["raw_dfast"] / n) for n, u in d["dfast"].items()]
    jobs += [(d["fred_url"].format(series=s), ROOT / p["raw_macro"] / f"{s}.csv") for s in d["fred_series"]]
    jobs += [(u, ROOT / p["raw_docs"] / n) for n, u in d["docs"].items()]
    jobs += [(u, ROOT / p["raw_macro"] / n) for n, u in d.get("macro_files", {}).items()]

    manifest, failed = [], []
    for url, dest in jobs:
        try:
            rec = fetch(url, dest)
            manifest.append(rec)
            print(f"[{rec['status']:>10}] {rec['file']}")
        except Exception as e:  # report and continue; summary decides exit code
            failed.append({"url": url, "error": str(e)})
            print(f"[    FAILED] {url}: {e}", file=sys.stderr)

    (ROOT / "data" / "raw" / "manifest_public.json").write_text(
        json.dumps({"files": manifest, "failed": failed}, indent=2))
    print(f"\n{len(manifest)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
