"""One-time conversion of pipe-delimited Freddie Mac sample files to typed parquet.

Streams each file with polars' lazy engine, so peak memory stays far below file size.
Casting is strict: a value that does not parse as the layout says stops the run
instead of silently becoming null.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from cecl.data.schema import ORIG_FIELDS, PERF_FIELDS, raw_schema, typed


def _scan(path: Path, fields) -> pl.LazyFrame:
    return pl.scan_csv(
        path, separator="|", has_header=False, schema=raw_schema(fields),
        quote_char=None,
    ).with_columns(typed(fields))


def convert_vintage(raw_dir: Path, out_dir: Path, year: int, overwrite: bool = False) -> dict:
    src = raw_dir / f"sample_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"vintage": year}
    for kind, fields, pattern in (("orig", ORIG_FIELDS, "sample_orig_{y}.txt"),
                                  ("perf", PERF_FIELDS, "sample_perf_{y}.txt")):
        dst = out_dir / f"{kind}_{year}.parquet"
        if overwrite or not dst.exists():
            _scan(src / pattern.format(y=year), fields).with_columns(
                pl.lit(year, pl.Int16).alias("vintage")
            ).sink_parquet(dst, compression="zstd")
        stats[f"{kind}_rows"] = pl.scan_parquet(dst).select(pl.len()).collect().item()
    return stats
