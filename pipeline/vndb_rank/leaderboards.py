"""User leaderboards (stored in meta.leaderboards).

* ``most_votes``       most votes on any VN (VNDB's own count of a user's votes)
* ``most_votes_year``  most votes cast in the dump's calendar year
* ``highest_mean`` / ``lowest_mean``      mean vote, users with >= MIN_VOTES ranked votes
* ``most_mainstream`` / ``most_contrarian`` Pearson r between a user's votes and
  each VN's mean vote, users with >= MIN_VOTES ranked votes

Every entry is ``[uid, name, value, ranked_votes, has_user_page]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extract import Extras, Votes

MIN_VOTES = 100
SIZE = 50


def build_leaderboards(votes: Votes, extras: Extras, names: dict[int, str], vn_mean: np.ndarray, pages: set[int]) -> dict:
    df = pd.DataFrame({"u": votes.uid, "v": votes.vote / 10, "m": vn_mean[votes.vidx]})
    g = df.groupby("u")
    n = g.size()
    mean = g["v"].mean()

    big = n[n >= MIN_VOTES].index
    sub = df[df["u"].isin(big)]
    # Vectorized Pearson per user.
    sg = sub.groupby("u")
    mv, mm = sg["v"].transform("mean"), sg["m"].transform("mean")
    dv, dm = sub["v"] - mv, sub["m"] - mm
    num = (dv * dm).groupby(sub["u"]).sum()
    den = np.sqrt((dv * dv).groupby(sub["u"]).sum() * (dm * dm).groupby(sub["u"]).sum())
    r = (num / den).replace([np.inf, -np.inf], np.nan).dropna()

    def entries(series: pd.Series, ascending: bool, digits: int) -> list[list]:
        s = series.sort_values(ascending=ascending).head(SIZE)
        return [
            [int(u), names.get(int(u), ""), round(float(v), digits) if digits else int(v), int(n.get(u, 0)), int(u) in pages]
            for u, v in s.items()
        ]

    big_mean = mean.loc[big]
    return {
        "min_votes": MIN_VOTES,
        "most_votes": entries(extras.user_total, False, 0),
        "most_votes_year": entries(extras.user_year, False, 0),
        "highest_mean": entries(big_mean, False, 2),
        "lowest_mean": entries(big_mean, True, 2),
        "most_mainstream": entries(r, False, 3),
        "most_contrarian": entries(r, True, 3),
    }
