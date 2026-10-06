"""Build the partial order network.

For every pair of ranked VNs (a < b) we accumulate, over all users who voted on
both:

* ``pv`` users who rated a higher than b, ``nv`` lower, ``tv`` in total
* arithmetic / geometric means of both raw votes and of the users'
  *sample percentiles* (where the vote sits within that user's own list)

Users' items are sorted by matrix index so that every pair lands in the upper
triangle. (The legacy script indexed by rating order but iterated in vid order,
so roughly half of all comparisons fell into the lower triangle and were
silently dropped on export.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from .extract import Votes

log = logging.getLogger(__name__)


@dataclass
class Pairs:
    a: np.ndarray   # matrix index, a < b
    b: np.ndarray
    pv: np.ndarray  # users preferring a
    nv: np.ndarray  # users preferring b
    tv: np.ndarray  # users who voted on both
    # mean scores of a / b among the common voters, on a 0-1 scale
    ari: np.ndarray     # shape (P, 2): arithmetic mean of vote / 100
    geo: np.ndarray     # geometric mean of vote / 100
    sp_ari: np.ndarray  # arithmetic mean of sample percentile
    sp_geo: np.ndarray  # geometric mean of sample percentile

    def __len__(self) -> int:
        return len(self.a)


def sample_percentile(votes: np.ndarray) -> np.ndarray:
    """Percentile of each vote within one user's list: n items split the line
    into n+1 equally likely intervals; ties share the midpoint."""
    return rankdata(votes, method="average") / (len(votes) + 1)


def build_pairs(votes: Votes, n_items: int, min_common: int) -> Pairs:
    N = n_items
    pv = np.zeros((N, N), dtype=np.int32)
    nv = np.zeros((N, N), dtype=np.int32)
    tv = np.zeros((N, N), dtype=np.int32)
    # float32 keeps this at 32 bytes per cell (~3 GB for 10k VNs); the means are
    # only used as game scores for the rankit methods, so the precision is ample.
    acc = np.zeros((8, N, N), dtype=np.float32)

    starts = np.flatnonzero(np.r_[True, np.diff(votes.uid) != 0])
    ends = np.r_[starts[1:], len(votes.uid)]
    log.info("pairs: %d users, %d items", len(starts), N)

    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        idx = votes.vidx[s:e]  # ascending, see extract.load_votes
        v = votes.vote[s:e].astype(np.float64)
        sp = sample_percentile(v)
        i, j = np.triu_indices(n, k=1)
        a, b = idx[i], idx[j]
        gt, lt = v[i] > v[j], v[i] < v[j]
        pv[a[gt], b[gt]] += 1
        nv[a[lt], b[lt]] += 1
        tv[a, b] += 1
        lv, lsp = np.log(v / 100), np.log(sp)
        acc[0, a, b] += v[i]
        acc[1, a, b] += v[j]
        acc[2, a, b] += lv[i]
        acc[3, a, b] += lv[j]
        acc[4, a, b] += sp[i]
        acc[5, a, b] += sp[j]
        acc[6, a, b] += lsp[i]
        acc[7, a, b] += lsp[j]

    a, b = np.nonzero(np.triu(tv >= min_common, k=1))
    t = tv[a, b].astype(np.float64)
    m = acc[:, a, b] / t  # (8, P)
    pairs = Pairs(
        a=a.astype(np.int32),
        b=b.astype(np.int32),
        pv=pv[a, b],
        nv=nv[a, b],
        tv=tv[a, b],
        ari=np.c_[m[0], m[1]] / 100,
        geo=np.exp(np.c_[m[2], m[3]]),
        sp_ari=np.c_[m[4], m[5]],
        sp_geo=np.exp(np.c_[m[6], m[7]]),
    )
    log.info("pairs: %d comparable pairs (>= %d common voters)", len(pairs), min_common)
    return pairs
