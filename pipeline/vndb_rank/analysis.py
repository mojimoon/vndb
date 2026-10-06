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


def pearson_by_user(uinv: np.ndarray, x: np.ndarray, y: np.ndarray, min_n: int = 3) -> np.ndarray:
    """Pearson r of x and y within each user (uinv = user index per vote);
    NaN in y is skipped, and users with fewer than min_n pairs or no variance get NaN."""
    m = ~np.isnan(y)
    u, x, y = uinv[m], x[m], y[m]
    k = int(uinv.max()) + 1 if len(uinv) else 0
    n = np.bincount(u, minlength=k).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.bincount(u, weights=x, minlength=k) / n
        my = np.bincount(u, weights=y, minlength=k) / n
        cov = np.bincount(u, weights=x * y, minlength=k) / n - mx * my
        vx = np.bincount(u, weights=x * x, minlength=k) / n - mx * mx
        vy = np.bincount(u, weights=y * y, minlength=k) / n - my * my
        r = cov / np.sqrt(vx * vy)
    r[(n < min_n) | (vx <= 1e-9) | (vy <= 1e-12)] = np.nan
    return np.clip(r, -1, 1)


def user_attributes(votes: Votes, user_total: pd.Series, targets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """User-level attributes copied onto each of their votes (for filtering voters):

    * ``nvotes`` the user's votes on any VN (uint16, saturating)
    * ``umean``  the user's mean ranked vote on the 10-100 scale (uint8)
    * ``corr``   one int8 per target: Pearson r x 100 between the user's votes and
                 a per-VN value (NaN = skip), -128 when undefined
    """
    users, uinv = np.unique(votes.uid, return_inverse=True)
    x = votes.vote.astype(np.float64)
    mean = np.bincount(uinv, weights=x) / np.bincount(uinv)
    total = user_total.reindex(users).fillna(0).to_numpy()
    out = {
        "nvotes": np.minimum(total, 65535).astype(np.uint16)[uinv],
        "umean": np.rint(mean).astype(np.uint8)[uinv],
        "corr": [],
    }
    for t in targets:
        r = pearson_by_user(uinv, x, np.asarray(t, dtype=np.float64)[votes.vidx])
        enc = np.where(np.isnan(r), -128, np.rint(np.nan_to_num(r) * 100)).astype(np.int8)
        out["corr"].append(enc[uinv])
    return out


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
