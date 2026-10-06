"""Compact encodings for the bulky per-snapshot data.

* Pair blocks: all 7M+ comparable pairs, so any two VNs can be compared. Each
  pair (a < b) is stored once under ``a`` as 8 little-endian bytes
  ``b:u16 pv:u16 nv:u16 tv:u16``, sorted by b, split into parts of at most
  ``PART_BYTES`` (D1 caps a statement at 100 KB and a BLOB literal is hex).
* User shards: users grouped by ``uid % USER_SHARDS``; one JSON object per part.
* Name index: lowercased username -> uid, grouped by FNV-1a hash.
* Rank history: [[day, rank_a, rank_b], ...] per VN, daily for 90 days, then
  weekly for two years.

The worker (web/worker/index.ts) mirrors these formats; keep them in sync.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Iterator

import numpy as np

from .pairs import Pairs

PART_BYTES = 40_000
PART_JSON_CHARS = 80_000
USER_SHARDS = 2048
NAME_SHARDS = 64
EPOCH = dt.date(2000, 1, 1)


def pair_blocks(p: Pairs) -> Iterator[tuple[int, int, bytes]]:
    """Yield (a, part, blob)."""
    if len(p) and max(p.b.max(), p.tv.max()) > 0xFFFF:
        raise ValueError("pair values exceed uint16; widen the pair block format")
    rec = np.empty(len(p), dtype=[("b", "<u2"), ("pv", "<u2"), ("nv", "<u2"), ("tv", "<u2")])
    rec["b"], rec["pv"], rec["nv"], rec["tv"] = p.b, p.pv, p.nv, p.tv
    per_part = PART_BYTES // rec.itemsize
    starts = np.flatnonzero(np.r_[True, np.diff(p.a) != 0])
    ends = np.r_[starts[1:], len(p.a)]
    for s, e in zip(starts, ends):
        a = int(p.a[s])
        for part, off in enumerate(range(s, e, per_part)):
            yield a, part, rec[off:min(off + per_part, e)].tobytes()


def lookup_pair(blocks: dict[int, bytes], a: int, b: int) -> tuple[int, int, int] | None:
    """Reference implementation of the worker's lookup (used in tests)."""
    lo, hi, flip = (a, b, False) if a < b else (b, a, True)
    data = blocks.get(lo)
    if not data:
        return None
    rec = np.frombuffer(data, dtype=[("b", "<u2"), ("pv", "<u2"), ("nv", "<u2"), ("tv", "<u2")])
    i = int(np.searchsorted(rec["b"], hi))
    if i >= len(rec) or rec["b"][i] != hi:
        return None
    pv, nv, tv = int(rec["pv"][i]), int(rec["nv"][i]), int(rec["tv"][i])
    return (nv, pv, tv) if flip else (pv, nv, tv)


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _json_parts(items: list[tuple[str, object]]) -> Iterator[str]:
    buf: dict[str, object] = {}
    size = 2
    for k, v in items:
        n = len(json.dumps({k: v}, ensure_ascii=False, separators=(",", ":")))
        if buf and size + n > PART_JSON_CHARS:
            yield json.dumps(buf, ensure_ascii=False, separators=(",", ":"))
            buf, size = {}, 2
        buf[k] = v
        size += n
    if buf:
        yield json.dumps(buf, ensure_ascii=False, separators=(",", ":"))


def user_shards(records: dict[int, dict]) -> Iterator[tuple[int, int, str]]:
    shards: dict[int, list] = {}
    for uid in sorted(records):
        shards.setdefault(uid % USER_SHARDS, []).append((str(uid), records[uid]))
    for shard in sorted(shards):
        for part, text in enumerate(_json_parts(shards[shard])):
            yield shard, part, text


def name_shards(records: dict[int, dict]) -> Iterator[tuple[int, int, str]]:
    shards: dict[int, list] = {}
    for uid, r in records.items():
        name = (r.get("name") or "").lower()
        if name:
            shards.setdefault(fnv1a(name) % NAME_SHARDS, []).append((name, uid))
    for shard in sorted(shards):
        for part, text in enumerate(_json_parts(sorted(shards[shard]))):
            yield shard, part, text


def day_number(date: str) -> int:
    return (dt.date.fromisoformat(date) - EPOCH).days


def merge_history(prev: list | None, today: int, ranks: list[int]) -> list[list[int]]:
    """Append today's ranks and thin out old points (daily 90 days, weekly 2 years)."""
    pts = [p for p in (prev or []) if isinstance(p, list) and p and p[0] < today]
    pts.append([today, *ranks])
    out, seen_weeks = [], set()
    for p in pts:
        age = today - p[0]
        if age <= 90:
            out.append(p)
        elif age <= 730:
            week = p[0] // 7
            if week not in seen_weeks:
                seen_weeks.add(week)
                out.append(p)
    return out


def rank_trend(history: list[list[int]], today: int, days: int = 7) -> int | None:
    """Change of the first tracked rank vs. the closest point at least `days` ago
    (positive = moved up)."""
    old = [p for p in history if today - p[0] >= days]
    if not old:
        return None
    return old[-1][1] - history[-1][1]
