"""Compact encodings for the bulky per-snapshot data.

* Voter shards: every VN's voters as (uid, vote, sample-percentile decile),
  so the worker can compute exact head-to-head counts and joint vote
  distributions for any two VNs. Binary parts are at most ``PART_BYTES``
  (D1 caps a statement at 100 KB and a BLOB literal is hex).
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


PART_BYTES = 40_000        # binary parts (hex-encoded in SQL, so ~80 KB per statement)
PART_TEXT_BYTES = 80_000   # UTF-8 bytes per text part
USER_SHARDS = 2048
NAME_SHARDS = 64
VOTER_SHARDS = 1024        # vn idx % VOTER_SHARDS
VN_NOTE_SHARDS = 512       # vn idx % VN_NOTE_SHARDS
USER_NOTE_SHARDS = 1024    # uid % USER_NOTE_SHARDS
VOTER_DTYPE = np.dtype([("uid", "<u4"), ("vote", "u1"), ("spd", "u1")])
EPOCH = dt.date(2000, 1, 1)


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _nbytes(obj: object) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _json_parts(items: list[tuple[str, object]]) -> Iterator[str]:
    buf: dict[str, object] = {}
    size = 2
    for k, v in items:
        n = _nbytes({k: v})
        if buf and size + n > PART_TEXT_BYTES:
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


def json_array_parts(rows: list) -> Iterator[str]:
    """Split a list into JSON arrays of at most PART_TEXT_BYTES each."""
    buf: list = []
    size = 2
    for r in rows:
        n = _nbytes(r) + 1
        if buf and size + n > PART_TEXT_BYTES:
            yield json.dumps(buf, ensure_ascii=False, separators=(",", ":"))
            buf, size = [], 2
        buf.append(r)
        size += n
    if buf:
        yield json.dumps(buf, ensure_ascii=False, separators=(",", ":"))


def text_parts(text: str) -> Iterator[str]:
    """Split a string into pieces of at most PART_TEXT_BYTES UTF-8 bytes (on character boundaries)."""
    raw = text.encode("utf-8")
    pos = 0
    while pos < len(raw):
        piece = raw[pos:pos + PART_TEXT_BYTES].decode("utf-8", errors="ignore")
        pos += len(piece.encode("utf-8"))
        yield piece


def voter_shards(vidx: np.ndarray, uid: np.ndarray, vote: np.ndarray, spd: np.ndarray) -> Iterator[tuple[int, int, bytes]]:
    """Per-VN voter lists for exact joint distributions of any two VNs.

    Shard = vn idx % VOTER_SHARDS. A part is a sequence of segments
    ``idx:u16 n:u32`` followed by n records ``uid:u32 vote:u8 spd:u8`` sorted by
    uid; one VN may continue in several segments across parts."""
    order = np.lexsort((uid, vidx))
    recs = np.empty(len(order), dtype=VOTER_DTYPE)
    recs["uid"], recs["vote"], recs["spd"] = uid[order], vote[order], spd[order]
    v = vidx[order]
    starts = np.flatnonzero(np.r_[True, np.diff(v) != 0])
    ends = np.r_[starts[1:], len(v)]
    by_shard: dict[int, list[tuple[int, np.ndarray]]] = {}
    for s, e in zip(starts, ends):
        by_shard.setdefault(int(v[s]) % VOTER_SHARDS, []).append((int(v[s]), recs[s:e]))
    head = np.dtype([("idx", "<u2"), ("n", "<u4")])
    per_seg = (PART_BYTES - head.itemsize) // VOTER_DTYPE.itemsize
    for shard in sorted(by_shard):
        part, buf = 0, bytearray()
        for idx, r in by_shard[shard]:
            off = 0
            while off < len(r):
                room = (PART_BYTES - len(buf) - head.itemsize) // VOTER_DTYPE.itemsize
                if room <= 0:
                    yield shard, part, bytes(buf)
                    part, buf = part + 1, bytearray()
                    room = per_seg
                take = min(room, len(r) - off)
                h = np.array([(idx, take)], dtype=head)
                buf += h.tobytes() + r[off:off + take].tobytes()
                off += take
        if buf:
            yield shard, part, bytes(buf)


def read_voters(parts: list[bytes], idx: int) -> np.ndarray:
    """Reference reader for voter_shards (the worker mirrors this)."""
    out = []
    for data in parts:
        pos = 0
        while pos < len(data):
            i, n = np.frombuffer(data, dtype=[("idx", "<u2"), ("n", "<u4")], count=1, offset=pos)[0]
            pos += 6
            if int(i) == idx:
                out.append(np.frombuffer(data, dtype=VOTER_DTYPE, count=int(n), offset=pos))
            pos += int(n) * VOTER_DTYPE.itemsize
    return np.concatenate(out) if out else np.zeros(0, dtype=VOTER_DTYPE)


def note_shards(rows: list[list], key: int, shards: int) -> Iterator[tuple[int, int, str]]:
    """Group note rows by rows[key] % shards into JSON array parts."""
    groups: dict[int, list] = {}
    for r in rows:
        groups.setdefault(int(r[key]) % shards, []).append(r)
    for shard in sorted(groups):
        for part, text in enumerate(json_array_parts(groups[shard])):
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
