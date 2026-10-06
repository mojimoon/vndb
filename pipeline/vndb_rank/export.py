"""Write a snapshot as a SQL script for ``wrangler d1 execute --file``.

Design notes (see docs/architecture.md):

* One row per ranked VN. All per-method ranks live in a JSON column and the
  head-to-head list in another, so a full refresh writes only ~N + producers
  rows instead of N x methods + pairs (which blew past Supabase's quota).
* Rows are upserted with the new ``snapshot`` id, stale rows are deleted, and
  ``meta.snapshot`` is flipped last, so readers never cache a half-import under
  the new id.
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

VN_COLUMNS = [
    "id", "title", "latin", "title_ja", "title_zh", "title_en", "olang", "released", "dev_id",
    "image", "image_sexual", "length", "votes", "rating", "average", "search",
    "ranks", "neighbors", "relations", "snapshot",
]
PRODUCER_COLUMNS = ["id", "name", "latin", "search", "snapshot"]


def sql_literal(v: Any) -> str:
    if v is None or v is pd.NA:
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NULL" if math.isnan(v) or math.isinf(v) else repr(round(float(v), 6))
    return "'" + str(v).replace("'", "''") + "'"


def insert_statements(table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]], upsert: bool = True) -> list[str]:
    head = f"INSERT {'OR REPLACE ' if upsert else ''}INTO {table} ({', '.join(columns)}) VALUES\n"
    stmts, buf, size = [], [], len(head)
    for row in rows:
        val = "(" + ", ".join(sql_literal(v) for v in row) + ")"
        n = len(val.encode("utf-8")) + 2
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


def write_sql(
    path: Path,
    snapshot: str,
    vn: pd.DataFrame,
    producers: pd.DataFrame,
    ranks_col: list[str],
    neighbors: dict[int, list],
    relations: dict[int, list],
    meta: dict[str, Any],
) -> dict[str, int]:
    vn_rows = []
    for i, row in enumerate(vn.itertuples(index=False)):
        r = row._asdict()
        vid = int(r["id"])
        vn_rows.append([
            vid, r["title"], _none(r["latin"]), _none(r["title_ja"]), _none(r["title_zh"]), _none(r["title_en"]),
            _none(r["olang"]), _none(r["released"]), _none(r["dev_id"]), _none(r["image"]),
            _none(r["image_sexual"]), _none(r["length"]), int(r["votes"]), _none(r["rating"]), _none(r["average"]),
            r["search"], ranks_col[i], compact_json(neighbors.get(vid, [])), compact_json(relations.get(vid, [])), snapshot,
        ])
    prod_rows = [
        [int(p.id), p.name, _none(p.latin), p.search, snapshot] for p in producers.itertuples(index=False)
    ]

    parts = [f"-- VNDB Ranking+ snapshot {snapshot}"]
    parts += insert_statements("vn", VN_COLUMNS, vn_rows)
    parts += insert_statements("producer", PRODUCER_COLUMNS, prod_rows)
    parts.append(f"DELETE FROM vn WHERE snapshot <> {sql_literal(snapshot)};")
    parts.append(f"DELETE FROM producer WHERE snapshot <> {sql_literal(snapshot)};")
    meta_rows = [[k, compact_json(v)] for k, v in meta.items() if k != "snapshot"]
    meta_rows.append(["snapshot", compact_json(snapshot)])  # last: flips the cache key
    parts += insert_statements("meta", ["key", "value"], meta_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {"vn": len(vn_rows), "producer": len(prod_rows), "meta": len(meta_rows)}
