"""Per-VN head-to-head lists ("which VNs is this one usually compared with?").

For each VN we keep the top-k opponents under several criteria and store the
union as ``[opponent_id, wins, losses, common_voters]`` rows. Every category's
top-k is contained in the union, so the frontend can re-derive each list.

Categories (computed from the VN's point of view, x wins / y losses / n common):

* popular   - n
* ahead     - x / n * log10(n)     (this VN usually rated higher)
* behind    - y / n * log10(n)     (this VN usually rated lower)
* contested - (1 - |x - y| / (x + y)) * log10(n)
* tied      - (n - x - y) / n * log10(n)
"""

from __future__ import annotations

import numpy as np

from .pairs import Pairs

CATEGORIES = ["popular", "ahead", "behind", "contested", "tied"]


def _top_k_per_group(group: np.ndarray, key: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest ``key`` values within each group."""
    order = np.lexsort((-key, group))
    g = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    pos = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    return order[pos < k]


def build_neighbors(p: Pairs, ids: np.ndarray, k: int) -> dict[int, list[list[int]]]:
    if len(p) == 0:
        return {}
    # Both directions: row r describes ``me`` versus ``other``.
    me = np.r_[p.a, p.b]
    other = np.r_[p.b, p.a]
    x = np.r_[p.pv, p.nv].astype(np.int64)
    y = np.r_[p.nv, p.pv].astype(np.int64)
    n = np.r_[p.tv, p.tv].astype(np.int64)
    logn = np.log10(n)
    decided = np.maximum(x + y, 1)
    keys = {
        "popular": n.astype(np.float64),
        "ahead": x / n * logn,
        "behind": y / n * logn,
        "contested": (1 - np.abs(x - y) / decided) * logn * (x + y > 0),
        "tied": (n - x - y) / n * logn,
    }
    keep = np.zeros(len(me), dtype=bool)
    for key in keys.values():
        keep[_top_k_per_group(me, key, k)] = True

    rows = np.flatnonzero(keep)
    rows = rows[np.lexsort((-n[rows], me[rows]))]
    out: dict[int, list[list[int]]] = {}
    for r in rows:
        out.setdefault(int(ids[me[r]]), []).append([int(ids[other[r]]), int(x[r]), int(y[r]), int(n[r])])
    return out
