"""Daily history of every ranked VN: default-method rank, VNDB rank, VNDB rating and votes.

Storage: table ``history_day`` (day, part, data), one row group per recorded
day holding every VN ranked that day (HIST_DTYPE records, base64). Unlike the
other tables it is *not* rebuilt by each snapshot: the refresh fetches it,
and the snapshot only inserts the new day and deletes the days thinned out,
so a failed fetch can never wipe it (nothing is deleted that was not fetched).

Thinning (``keep_days``): every day for 90 days, then the first recorded day
of each week (and of each month) for a year, then the first recorded day of
each month.

The per-VN view (``vn.history``: [[day, rank, vndb_rank, rating, votes], ...])
is derived from these days on every run, so titles that drop out of the
ranking keep their history and get it back if they return.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 14 bytes per VN and day; the frontend (lib/api.ts: decodeHistoryDay) mirrors this layout.
HIST_DTYPE = np.dtype([
    ("vid", "<u4"),
    ("rank", "<u2"),     # default method (grand ranking)
    ("vndb", "<u2"),     # VNDB rank (Bayesian rating)
    ("rating", "<u2"),   # VNDB rating x 100, 0 = none
    ("votes", "<u4"),    # VNDB vote count, 0 = unknown (days recorded before votes were tracked)
])
DAILY = 90
WEEKLY = 365
EPOCH = dt.date(2000, 1, 1)


class HistoryError(RuntimeError):
    """The fetched history is incomplete or malformed: refuse to build on it."""


def encode_day(rec: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(rec, dtype=HIST_DTYPE).tobytes()).decode("ascii")


def decode_day(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) % HIST_DTYPE.itemsize:
        raise HistoryError(f"history day of {len(raw)} bytes is not a whole number of records")
    return np.frombuffer(raw, dtype=HIST_DTYPE).copy()


def day_records(ids: np.ndarray, rank: np.ndarray, vndb: np.ndarray, rating: np.ndarray, votes: np.ndarray) -> np.ndarray:
    rec = np.zeros(len(ids), dtype=HIST_DTYPE)
    rec["vid"] = ids
    rec["rank"] = np.clip(rank, 0, 0xFFFF)
    rec["vndb"] = np.clip(vndb, 0, 0xFFFF)
    r = np.asarray(rating, dtype=np.float64)
    rec["rating"] = np.where(np.isfinite(r), np.clip(np.rint(r * 100), 0, 0xFFFF), 0)
    rec["votes"] = np.clip(votes, 0, 0xFFFFFFFF)
    return rec[np.argsort(rec["rank"], kind="stable")]


def _month(day: int) -> tuple[int, int]:
    d = EPOCH + dt.timedelta(days=day)
    return d.year, d.month


def keep_days(days: list[int] | set[int], today: int) -> set[int]:
    """Days to keep: all of the last DAILY days, the first day of each week or
    month within WEEKLY days, the first day of each month before that."""
    first_week: dict[int, int] = {}
    first_month: dict[tuple[int, int], int] = {}
    for d in sorted(days):
        first_week.setdefault((d + 5) // 7, d)  # weeks start on Monday (day 0 is a Saturday)
        first_month.setdefault(_month(d), d)
    weekly, monthly = set(first_week.values()), set(first_month.values())
    out = set()
    for d in days:
        age = today - d
        if age < 0:
            continue
        if age <= DAILY or d in monthly or (age <= WEEKLY and d in weekly):
            out.add(d)
    return out


def _rows(data) -> list[dict]:
    """`wrangler d1 execute --json` output ([{results: [...]}, ...]) or a plain list of rows."""
    if isinstance(data, list) and data and isinstance(data[0], dict) and "results" in data[0]:
        return [r for block in data for r in block.get("results", [])]
    return data or []


def load_days(path: Path | None) -> dict[int, np.ndarray]:
    """Rows {day, part, data} of history_day. Missing parts or bad data raise HistoryError."""
    if not path or not path.exists():
        return {}
    try:
        rows = _rows(json.loads(path.read_text() or "[]"))
    except json.JSONDecodeError as e:
        raise HistoryError(f"{path} is not JSON") from e
    parts: dict[int, dict[int, str]] = {}
    for r in rows:
        parts.setdefault(int(r["day"]), {})[int(r["part"])] = r["data"]
    out = {}
    for day, p in parts.items():
        if sorted(p) != list(range(len(p))):
            raise HistoryError(f"history day {day}: parts {sorted(p)} are not contiguous")
        out[day] = decode_day("".join(p[i] for i in range(len(p))))
    log.info("history: %d days from %s", len(out), path)
    return out


def legacy_days(path: Path | None) -> dict[int, np.ndarray]:
    """Days rebuilt from the old per-VN column (`SELECT id, history FROM vn`,
    points [day, rank, vndb_rank]); rating and votes were not recorded then."""
    if not path or not path.exists():
        return {}
    try:
        rows = _rows(json.loads(path.read_text() or "[]"))
    except json.JSONDecodeError as e:
        raise HistoryError(f"{path} is not JSON") from e
    by_day: dict[int, list[tuple[int, int, int]]] = {}
    for r in rows:
        h = json.loads(r["history"]) if isinstance(r["history"], str) else r["history"]
        for p in h or []:
            by_day.setdefault(int(p[0]), []).append((int(r["id"]), int(p[1]), int(p[2])))
    out = {}
    for day, pts in by_day.items():
        a = np.array(pts, dtype=np.int64)
        out[day] = day_records(a[:, 0], a[:, 1], a[:, 2], np.full(len(a), np.nan), np.zeros(len(a), np.int64))
    log.info("history: %d days rebuilt from %d VNs' legacy history in %s", len(out), len(rows), path)
    return out


def vn_histories(days: dict[int, np.ndarray], ids: np.ndarray) -> list[list[list]]:
    """Per VN in `ids`: [[day, rank, vndb_rank, rating | None, votes | None], ...], oldest first."""
    pos = {int(v): i for i, v in enumerate(ids)}
    out: list[list[list]] = [[] for _ in ids]
    for day in sorted(days):
        rec = days[day]
        for vid, rank, vndb, rating, votes in zip(rec["vid"].tolist(), rec["rank"].tolist(), rec["vndb"].tolist(), rec["rating"].tolist(), rec["votes"].tolist()):
            i = pos.get(vid)
            if i is not None:
                out[i].append([day, rank, vndb, rating / 100 if rating else None, votes or None])
    return out


def rank_trend(history: list[list], today: int, days: int = 30) -> int | None:
    """Change of the default-method rank vs. the closest point at least `days`
    ago (positive = moved up)."""
    old = [p for p in history if today - p[0] >= days]
    if not old:
        return None
    return old[-1][1] - history[-1][1]
