"""Readers for the VNDB database dump (https://vndb.org/d14).

The dump is a set of PostgreSQL ``COPY ... TO`` text files under ``db/``, each
with a sibling ``<table>.header`` holding the tab-separated column names.
NULL is written as ``\\N`` and tabs/newlines inside values are backslash-escaped,
so the files can be parsed without any quoting.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
_ESCAPE_RE = re.compile(r"\\(.)")


class Dump:
    def __init__(self, root: str | Path):
        root = Path(root)
        # Accept either the extraction root (containing db/) or db/ itself.
        self.dir = root / "db" if (root / "db").is_dir() else root
        if not self.dir.is_dir():
            raise FileNotFoundError(f"dump directory not found: {self.dir}")
        self.root = self.dir.parent

    def has(self, table: str) -> bool:
        return (self.dir / table).exists() and (self.dir / f"{table}.header").exists()

    def header(self, table: str) -> list[str]:
        return (self.dir / f"{table}.header").read_text(encoding="utf-8").strip().split("\t")

    def _kwargs(self, table: str, columns: Sequence[str] | None) -> dict:
        header = self.header(table)
        if columns is not None:
            missing = [c for c in columns if c not in header]
            if missing:
                raise KeyError(f"{table}: columns {missing} not in dump header {header}")
        return dict(
            sep="\t",
            header=None,
            names=header,
            usecols=list(columns) if columns is not None else None,
            dtype=str,
            quoting=csv.QUOTE_NONE,
            na_values=["\\N"],
            keep_default_na=False,
            engine="c",
        )

    def read(self, table: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        return pd.read_csv(self.dir / table, **self._kwargs(table, columns))

    def read_chunks(
        self, table: str, columns: Sequence[str] | None = None, chunksize: int = 2_000_000
    ) -> Iterator[pd.DataFrame]:
        yield from pd.read_csv(self.dir / table, chunksize=chunksize, **self._kwargs(table, columns))

    def optional_columns(self, table: str, wanted: Sequence[str]) -> list[str]:
        header = set(self.header(table))
        return [c for c in wanted if c in header]

    def line_count(self, table: str) -> int:
        n = 0
        with open(self.dir / table, "rb") as f:
            while chunk := f.read(1 << 24):
                n += chunk.count(b"\n")
        return n

    def snapshot_date(self) -> str:
        """Date of the dump (from its TIMESTAMP file), falling back to today."""
        for p in (self.root / "TIMESTAMP", self.dir / "TIMESTAMP"):
            if p.exists():
                text = p.read_text().strip()
                m = re.search(r"\d{4}-\d{2}-\d{2}", text)
                if m:
                    return m.group(0)
        return dt.date.today().isoformat()


def unescape(s: str | float) -> str | None:
    """Undo COPY text-format escaping (``\\n`` -> newline, ...)."""
    if not isinstance(s, str):
        return None
    if "\\" not in s:
        return s
    return _ESCAPE_RE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def strip_id(series: pd.Series) -> pd.Series:
    """'v123' / 'cv123' / 'u45' -> 123 / 123 / 45 (nullable Int64)."""
    return pd.to_numeric(series.str.lstrip("abcdefghijklmnopqrstuvwxyz"), errors="coerce").astype("Int64")
