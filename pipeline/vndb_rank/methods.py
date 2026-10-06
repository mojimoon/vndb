"""Ranking methods over the partial order network.

Every method maps the pair table to one score per VN, higher = better.
VNs that ended up without any comparable pair get NaN and rank last.

Changes from the legacy research script (``research/legacy_main.py``):

* ``po_elo`` plays one fractional-outcome match per pair (the count-weighted
  updates diverged) and keeps ratings as floats (they were truncated to ints).
* ``po_entropy`` normalizes by the *sum* of entropies (one side was subtracted).
* Several rankit variants were dropped (see RANKIT_RANKERS).
* ``po_rw`` (random walk) is gone: on real data it was uncorrelated or
  negatively correlated with every other method (Kendall tau -0.25..0.19).
* ``po_vi`` (unseeded gradient steps whose fancy-indexed updates dropped
  duplicates) is replaced by a proper Bradley-Terry MM fit, ``po_bt``.
* The rankit Markov variants transformed the shared input frame in place, so
  every ranker after them saw altered scores; each ranker now gets a copy.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .pairs import Pairs

log = logging.getLogger(__name__)

PO_METHODS = ["po_total", "po_percent", "po_simple", "po_weighted", "po_elo", "po_entropy", "po_bt"]
RANKIT_VARIABLES = ["prob", "ari", "geo", "sp_ari", "sp_geo"]
# Dropped after checking them on the 2026-10 dump (Kendall tau vs. the median
# rank of all methods; Spearman rho vs. log vote count):
#   keener_*      tau 0.39-0.56, rho ~0.75   -> a popularity contest
#   markov_rv_*   tau 0.42-0.62, rho ~0.77   -> same
#   difference_*  tau 0.36-0.63; top lists are simply the most-voted titles
#   massey_prob   tau 0.41; raw preference counts make big pairs dominate
RANKIT_RANKERS = ["massey", "colley", "markov_rdv", "markov_sdv", "od"]
RANKIT_EXCLUDE = {"massey_prob"}
GRAND_VARIABLES = ["prob", "sp_ari", "sp_geo"]


def _bincount2(a: np.ndarray, b: np.ndarray, wa: np.ndarray, wb: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(a, weights=wa, minlength=n) + np.bincount(b, weights=wb, minlength=n)


def po_classical(p: Pairs, n: int) -> dict[str, np.ndarray]:
    """Average pairwise score against every comparable opponent."""
    d = (p.pv - p.nv).astype(np.float64)
    sign = np.sign(d)
    appear = np.bincount(p.a, minlength=n) + np.bincount(p.b, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = lambda w: _bincount2(p.a, p.b, w, -w, n) / np.where(appear > 0, appear, np.nan)
        return {
            "po_total": avg(d),                         # (x - y)
            "po_percent": avg(d / p.tv),                # (x - y) / n
            "po_simple": avg(sign),                     # sgn(x - y)
            "po_weighted": avg(sign * np.sqrt(p.tv)),   # sgn(x - y) * sqrt(n)
        }


def po_elo(p: Pairs, n: int, K: float = 32, base: float = 1500, divisor: float = 400,
           epochs: int = 5, seed: int = 0) -> np.ndarray:
    """Elo where each pair is one match whose outcome is the share of common
    voters preferring a (ties count half). Several shuffled passes with a
    decaying K make the result far less dependent on match order.

    (The legacy version multiplied each update by the raw win count, which
    diverges for popular pairs with hundreds of common voters.)"""
    score = ((p.pv + (p.tv - p.pv - p.nv) / 2) / p.tv).tolist()
    a_list, b_list = p.a.tolist(), p.b.tolist()
    rating = [float(base)] * n
    order = np.arange(len(score))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        k = K / (1 + epoch)
        for t in order.tolist():
            a, b = a_list[t], b_list[t]
            e = 1 / (1 + 10 ** ((rating[b] - rating[a]) / divisor))
            d = k * (score[t] - e)
            rating[a] += d
            rating[b] -= d
    out = np.array(rating)
    out[_isolated(p, n)] = np.nan
    return out


def po_entropy(p: Pairs, n: int) -> np.ndarray:
    """Net preference weighted by how decisive each pair is (binary entropy)."""
    decided = (p.pv + p.nv).astype(np.float64)
    ok = decided > 0
    a, b = p.a[ok], p.b[ok]
    x, y = p.pv[ok] / decided[ok], p.nv[ok] / decided[ok]
    ent = -(x * np.log2(x + 1e-10) + y * np.log2(y + 1e-10))
    s = x - y
    num = _bincount2(a, b, s * ent, -s * ent, n)
    den = _bincount2(a, b, ent, ent, n)
    out = num / (den + 1e-10)
    out[_isolated(p, n)] = np.nan
    return out


def po_bradley_terry(p: Pairs, n: int, max_iter: int = 500, eps: float = 1e-8) -> np.ndarray:
    """Bradley-Terry strengths via Hunter's MM algorithm. Ties count as half a
    win for each side; half a pseudo-win keeps unbeaten / winless VNs finite."""
    ties = (p.tv - p.pv - p.nv).astype(np.float64)
    wins = _bincount2(p.a, p.b, p.pv + ties / 2, p.nv + ties / 2, n) + 0.5
    games = p.tv.astype(np.float64)
    strength = np.ones(n)
    for _ in range(max_iter):
        w = games / (strength[p.a] + strength[p.b])
        denom = _bincount2(p.a, p.b, w, w, n) + 1e-12
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(strength))) < eps:
            strength = new
            break
        strength = new
    out = np.log(strength)
    out[_isolated(p, n)] = np.nan
    return out


def _isolated(p: Pairs, n: int) -> np.ndarray:
    return (np.bincount(p.a, minlength=n) + np.bincount(p.b, minlength=n)) == 0


def rankit_scores(p: Pairs, n: int, variable: str, ranker: str) -> np.ndarray:
    from rankit.Ranker import (
        ColleyRanker, DifferenceRanker, KeenerRanker, MarkovRanker, MasseyRanker, ODRanker,
    )
    from rankit.Table import Table

    if variable == "prob":
        h, v = p.pv.astype(np.float64), p.nv.astype(np.float64)
    else:
        m = getattr(p, variable)
        h, v = m[:, 0].copy(), m[:, 1].copy()

    if ranker.startswith("markov"):
        if ranker == "markov_rv":
            tot = h + v + 1e-10
            h, v = h / tot, v / tot
        elif ranker == "markov_rdv":
            r = (h - v) / (h + v + 1e-10)
            h, v = np.maximum(0, r), np.maximum(0, -r)
        elif ranker == "markov_sdv":
            d = h - v
            h, v = np.maximum(0, d), np.maximum(0, -d)
        worker = MarkovRanker()
    else:
        worker = {
            "massey": MasseyRanker, "colley": ColleyRanker, "keener": KeenerRanker,
            "od": ODRanker, "difference": DifferenceRanker,
        }[ranker]()

    df = pd.DataFrame({"host": p.a, "visit": p.b, "hscore": h, "vscore": v})
    res = worker.rank(Table(df))
    out = np.full(n, np.nan)
    out[res["name"].to_numpy().astype(int)] = res["rating"].to_numpy(dtype=np.float64)
    return out


def borda(scores: pd.DataFrame) -> np.ndarray:
    """Borda count over several score columns (higher = better, NaN = last)."""
    filled = scores.fillna(-np.inf)
    ranks = filled.rank(method="min", ascending=False)
    n_items, n_methods = ranks.shape
    return (n_items * n_methods - ranks.sum(axis=1)).to_numpy(dtype=np.float64)


def compute_all(p: Pairs, n: int, skip_rankit: bool = False) -> pd.DataFrame:
    """Return a DataFrame (n rows, one column per method code)."""
    out: dict[str, np.ndarray] = {}
    steps: list[tuple[str, Callable[[], np.ndarray | dict]]] = [
        ("po_classical", lambda: po_classical(p, n)),
        ("po_elo", lambda: po_elo(p, n)),
        ("po_entropy", lambda: po_entropy(p, n)),
        ("po_bt", lambda: po_bradley_terry(p, n)),
    ]
    for name, fn in steps:
        log.info("method %s", name)
        res = fn()
        out.update(res if isinstance(res, dict) else {name: res})

    if not skip_rankit:
        for var in RANKIT_VARIABLES:
            for rk in RANKIT_RANKERS:
                code = f"{rk}_{var}"
                if code in RANKIT_EXCLUDE:
                    continue
                log.info("method %s", code)
                out[code] = rankit_scores(p, n, var, rk)

    df = pd.DataFrame(out)
    df["borda_po"] = borda(df[PO_METHODS])
    if not skip_rankit:
        for var in RANKIT_VARIABLES:
            df[f"borda_{var}"] = borda(df[[c for rk in RANKIT_RANKERS if (c := f"{rk}_{var}") in df]])
        df["borda_sci"] = borda(df[["borda_prob", "borda_ari", "borda_geo"]])
        df["borda_grand"] = borda(df[[c for var in GRAND_VARIABLES for rk in RANKIT_RANKERS if (c := f"{rk}_{var}") in df]])
    return df
