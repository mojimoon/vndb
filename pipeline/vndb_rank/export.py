"""Write a snapshot as a self-contained SQL script for ``wrangler d1 execute --file``.

Design notes (see docs/architecture.md):

* The script owns the schema: it drops and recreates every table, writing
  each row exactly once. D1 runs an imported file atomically, so readers see
  either the old or the new snapshot, never a mix.
* Bulky data is packed (see storage.py): one row per VN, pair blocks, user
  shards. A full refresh is ~20k row writes, well under D1's free 100k/day.
* No BEGIN/COMMIT: D1 rejects explicit transactions in imported files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

MAX_STATEMENT_BYTES = 90_000  # D1 caps a single statement at 100 KB

# Single source of truth for the D1 schema (the worker only reads these columns).
SCHEMA: dict[str, str] = {
    "vn": """
  id           INTEGER PRIMARY KEY,   -- VNDB id without the "v" prefix
  idx          INTEGER NOT NULL,      -- position in this snapshot (voter shards / user votes refer to it)
  title        TEXT    NOT NULL,      -- title in the original language
  latin        TEXT,
  title_ja     TEXT,
  title_zh     TEXT,
  title_en     TEXT,
  olang        TEXT,
  released     INTEGER,               -- YYYYMMDD (MM/DD may be 99)
  dev_id       INTEGER,
  image        INTEGER,               -- cover id (cv...)
  image_sexual REAL,                  -- 0 safe .. 2 explicit
  length       INTEGER,
  votes        INTEGER NOT NULL,      -- VNDB vote count
  rating       REAL,                  -- VNDB Bayesian rating, 1-10
  average      REAL,
  search       TEXT    NOT NULL,
  trend        INTEGER,               -- rank change of the default method over ~7 days
  ranks        TEXT    NOT NULL,      -- JSON {method: [rank, score]}
  neighbors    TEXT    NOT NULL,      -- JSON [[vid, wins, losses, common], ...]
  relations    TEXT    NOT NULL,      -- JSON [[vid, relation], ...]
  analysis     TEXT    NOT NULL,      -- JSON, see analysis.py
  similar      TEXT    NOT NULL,      -- JSON [[vid, similarity, common, wins, losses], ...]
  history      TEXT    NOT NULL       -- JSON [[day, rank, vndb_rank], ...], day = days since 2000-01-01
""",
    "producer": """
  id     INTEGER PRIMARY KEY,
  name   TEXT NOT NULL,
  latin  TEXT
""",
    "user_block": """
  shard INTEGER NOT NULL,  -- uid % 2048
  part  INTEGER NOT NULL,
  data  TEXT    NOT NULL,  -- JSON {uid: {name, votes, similar, recs}}
  PRIMARY KEY (shard, part)
""",
    "user_name": """
  shard INTEGER NOT NULL,  -- fnv1a(lower(name)) % 64
  part  INTEGER NOT NULL,
  data  TEXT    NOT NULL,  -- JSON {lower(name): uid}
  PRIMARY KEY (shard, part)
""",
    "vn_voters": """
  shard INTEGER NOT NULL,  -- vn idx % 1024
  part  INTEGER NOT NULL,
  data  BLOB    NOT NULL,  -- segments (idx u16, n u32) + n x 13-byte voter records, see storage.VOTER_DTYPE
  PRIMARY KEY (shard, part)
""",
    "vn_notes": """
  shard INTEGER NOT NULL,  -- vn idx % 512
  part  INTEGER NOT NULL,
  data  TEXT    NOT NULL,  -- JSON [[idx, uid, name, vote, date, text, has_page, labels, user_votes, sp%], ...], newest first per VN
  PRIMARY KEY (shard, part)
""",
    "user_notes": """
  shard INTEGER NOT NULL,  -- uid % 1024
  part  INTEGER NOT NULL,
  data  TEXT    NOT NULL,  -- JSON [[uid, idx, vote, date, text, labels, sp%], ...]
  PRIMARY KEY (shard, part)
""",
    "doc": """
  key   TEXT    NOT NULL,  -- "catalogue", "ranks:<method>"
  part  INTEGER NOT NULL,
  data  TEXT    NOT NULL,  -- a prebuilt JSON response, split into parts
  PRIMARY KEY (key, part)
""",
    "meta": """
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL      -- JSON
""",
}

VN_COLUMNS = [
    "id", "idx", "title", "latin", "title_ja", "title_zh", "title_en", "olang", "released", "dev_id",
    "image", "image_sexual", "length", "votes", "rating", "average", "search", "trend",
    "ranks", "neighbors", "relations", "analysis", "similar", "history",
]


# Tables keyed by a composite or text key are stored clustered on that key.
WITHOUT_ROWID = {"user_block", "user_name", "vn_voters", "vn_notes", "user_notes", "doc", "meta"}


PRODUCER_COLUMNS = ["id", "name", "latin"]


def create_table(name: str) -> str:
    tail = " WITHOUT ROWID" if name in WITHOUT_ROWID else ""
    return f"CREATE TABLE {name} ({SCHEMA[name].rstrip()}\n){tail};"


def sql_literal(v: Any) -> str:
    if v is None or v is pd.NA:
        return "NULL"
    if isinstance(v, (bytes, bytearray)):
        return "X'" + bytes(v).hex() + "'"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NULL" if math.isnan(v) or math.isinf(v) else repr(round(float(v), 6))
    return "'" + str(v).replace("'", "''") + "'"


def insert_statements(table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n"
    stmts, buf, size = [], [], len(head)
    for row in rows:
        val = "(" + ", ".join(sql_literal(v) for v in row) + ")"
        n = len(val.encode("utf-8")) + 2
        if n + len(head) > MAX_STATEMENT_BYTES:
            raise ValueError(f"{table}: a single row is {n} bytes, over the D1 statement limit")
        if buf and size + n > MAX_STATEMENT_BYTES:
            stmts.append(head + ",\n".join(buf) + ";")
            buf, size = [], len(head)
        buf.append(val)
        size += n
    if buf:
        stmts.append(head + ",\n".join(buf) + ";")
    return stmts


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _num(x: float) -> float | int | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    f = float(f"{float(x):.6g}")
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


def rank_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Competition ranks (1 = best, ties share the lower rank, NaN last)."""
    return scores.fillna(-np.inf).rank(method="min", ascending=False).astype(int)


def ranks_json(scores: pd.DataFrame, ranks: pd.DataFrame) -> list[str]:
    cols = list(scores.columns)
    s, r = scores.to_numpy(dtype=np.float64), ranks.to_numpy()
    return [compact_json({c: [int(r[i, j]), _num(s[i, j])] for j, c in enumerate(cols)}) for i in range(len(scores))]


def _none(v: Any) -> Any:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_sql(path: Path, tables: dict[str, tuple[Sequence[str], Iterable[Sequence[Any]]]]) -> dict[str, int]:
    """tables: name -> (columns, rows). ``meta`` (holding the snapshot id) is written last.

    D1 imports a file atomically (the database is unavailable while it runs and
    rolls back if any statement fails), so each table is simply dropped and
    rebuilt in place. Dropping first lets the new rows reuse the old pages: the
    database never holds two copies of the data, which a 500 MB database cannot
    afford once the snapshot passes ~200 MB."""
    counts: dict[str, int] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(tables, key=lambda n: n == "meta")
    with open(path, "w", encoding="utf-8") as f:
        f.write("-- VNDB SciRanking snapshot (generated by pipeline/vndb_rank)\n")
        # Drop everything first (also leftovers of the older build-then-swap format).
        for name in order:
            f.write(f"DROP TABLE IF EXISTS {name}__next;\nDROP TABLE IF EXISTS {name};\n")
        for name in order:
            columns, rows = tables[name]
            f.write(f"{create_table(name)}\n")
            rows = list(rows)
            counts[name] = len(rows)
            for stmt in insert_statements(name, columns, rows):
                f.write(stmt + "\n")
    return counts


def vn_rows(vn: pd.DataFrame, extra: dict[str, list[Any]]) -> list[list[Any]]:
    """Rows in VN_COLUMNS order; ``extra`` supplies per-row values for JSON columns etc."""
    out = []
    for i, row in enumerate(vn.itertuples(index=False)):
        r = row._asdict()
        out.append([_none(r[c]) if c in r else extra[c][i] for c in VN_COLUMNS])
    return out
