"""Per-VN rating analysis (stored as ``vn.analysis`` JSON).

For each ranked VN, computed from the votes of non-ignored users:

* ``hist``   votes per point 1..10
* ``mean`` / ``std`` of the votes (1-10 scale)
* ``years``  [[year, count, mean], ...] by vote date
* ``labels`` list-status counts {1 playing, 2 finished, 3 stalled, 4 dropped, 5 wishlist, 6 blacklist}
* ``sp``     how its voters place it within their own lists: mean sample
             percentile and a 10-bucket histogram (0-10%, ..., 90-100%)
* ``bias``   mean of (vote - that voter's own mean vote): positive means its
             voters like it more than they usually like things
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extract import Votes


def per_user_percentiles(votes: Votes) -> np.ndarray:
    """Sample percentile of every vote within its user's list (see pairs.sample_percentile)."""
    df = pd.DataFrame({"u": votes.uid, "v": votes.vote})
    rank = df.groupby("u")["v"].rank(method="average").to_numpy()
    n = df.groupby("u")["v"].transform("size").to_numpy()
    return rank / (n + 1)


def sp_decile(sp: np.ndarray) -> np.ndarray:
    """Sample percentile (0, 1) -> bucket 0..9."""
    return np.minimum((np.asarray(sp) * 10).astype(np.int64), 9)


def per_user_means(votes: Votes) -> np.ndarray:
    """Each vote's user mean (10-100 scale), aligned with votes."""
    return pd.Series(votes.vote, dtype=np.float64).groupby(votes.uid).transform("mean").to_numpy()


def build_analysis(votes: Votes, n_items: int, vn_labels: np.ndarray, sp: np.ndarray, user_mean: np.ndarray) -> list[dict]:
    v10 = votes.vote / 10
    idx = votes.vidx
    bucket = np.clip(votes.vote // 10, 1, 10) - 1
    hist = np.zeros((n_items, 10), dtype=np.int64)
    np.add.at(hist, (idx, bucket), 1)
    cnt = np.bincount(idx, minlength=n_items)
    s1 = np.bincount(idx, weights=v10, minlength=n_items)
    s2 = np.bincount(idx, weights=v10 * v10, minlength=n_items)
    sp_sum = np.bincount(idx, weights=sp, minlength=n_items)
    sp_hist = np.zeros((n_items, 10), dtype=np.int64)
    np.add.at(sp_hist, (idx, sp_decile(sp)), 1)
    bias = np.bincount(idx, weights=(votes.vote - user_mean) / 10, minlength=n_items)

    yr = pd.DataFrame({"i": idx, "y": votes.year, "v": v10})
    yr = yr[yr["y"] > 0].groupby(["i", "y"])["v"].agg(["size", "mean"]).reset_index()
    years: dict[int, list] = {}
    for i, y, c, m in zip(yr["i"], yr["y"], yr["size"], yr["mean"]):
        years.setdefault(int(i), []).append([int(y), int(c), round(float(m), 2)])

    out = []
    for i in range(n_items):
        n = int(cnt[i])
        mean = s1[i] / n if n else None
        out.append({
            "n": n,
            "hist": hist[i].tolist(),
            "mean": round(mean, 3) if n else None,
            "std": round(float(np.sqrt(max(s2[i] / n - mean * mean, 0))), 3) if n else None,
            "years": years.get(i, []),
            "labels": {str(k): int(vn_labels[i, k]) for k in range(1, vn_labels.shape[1]) if vn_labels[i, k]},
            "sp": {"mean": round(float(sp_sum[i] / n), 4), "hist": sp_hist[i].tolist()} if n else None,
            "bias": round(float(bias[i] / n), 3) if n else None,
        })
    return out
