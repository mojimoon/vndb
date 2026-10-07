"""Reproduce the grand-ranking study in docs/ranking-experiments.md.

    cd pipeline
    PYTHONPATH=. python ../research/grand_experiment.py --dump ../db --out ../research/out

For every rankit algorithm x game-score variable and every PONet method it
computes scores on all voters and on two disjoint random halves of the voters,
then reports for single methods and for candidate Borda combinations:

* tau   Kendall tau against the consensus (median rank over all methods)
* rho   Spearman rho against log(vote count)
* rel   Spearman rho between the two halves (split-half reliability)
* prel  the same with log(vote count) partialled out (reliability beyond popularity)
* top-100 stability (overlap of the two halves' top 100) and how many of the
  full-data top 100 have < 100 votes

Takes ~25 minutes and ~6 GB of memory on the full dump (three pair tables).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from vndb_rank.config import Config
from vndb_rank.dump import Dump
from vndb_rank.extract import Votes, load_catalogue, load_users, load_votes
from vndb_rank.methods import GRAND_INPUTS, po_bradley_terry, po_classical, po_elo, po_entropy, rankit_scores
from vndb_rank.pairs import build_pairs

RANKERS = ["massey", "colley", "keener", "markov_rv", "markov_rdv", "markov_sdv", "od", "difference"]
VARIABLES = ["prob", "ari", "geo", "sp_ari", "sp_geo"]
PO = ["po_total", "po_percent", "po_simple", "po_weighted", "po_elo", "po_entropy", "po_bt"]


def all_scores(votes: Votes, n: int, min_common: int) -> pd.DataFrame:
    p = build_pairs(votes, n, min_common)
    out = dict(po_classical(p, n))
    out["po_elo"], out["po_entropy"], out["po_bt"] = po_elo(p, n), po_entropy(p, n), po_bradley_terry(p, n)
    for var in VARIABLES:
        for rk in RANKERS:
            logging.info("%s_%s", rk, var)
            out[f"{rk}_{var}"] = rankit_scores(p, n, var, rk)
    return pd.DataFrame(out)


def ranks(s: pd.Series) -> np.ndarray:
    return s.fillna(-np.inf).rank(ascending=False, method="min").to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    dump = Dump(args.dump)
    cat = load_catalogue(dump, cfg)
    n = len(cat.vn)
    ignored, _ = load_users(dump)
    votes, *_ = load_votes(dump, cat, ignored, with_notes=False)
    half = ((votes.uid.astype(np.int64) * 2654435761) >> 7) % 2  # fixed pseudo-random split by user

    frames = {}
    for key, mask in [("full", np.ones(len(votes), bool)), ("half0", half == 0), ("half1", half == 1)]:
        v = Votes(uid=votes.uid[mask], vidx=votes.vidx[mask], vote=votes.vote[mask], year=votes.year[mask])
        path = args.out / f"scores_{key}.csv"
        if path.exists():
            frames[key] = pd.read_csv(path)
        else:
            frames[key] = all_scores(v, n, cfg.min_common_vote)
            frames[key].to_csv(path, index=False)

    lv = np.log(cat.vn["votes"].to_numpy())
    lv_rank = rankdata(lv)
    R = {k: {c: ranks(f[c]) for c in f.columns} for k, f in frames.items()}
    consensus = np.median(np.vstack(list(R["full"].values())), axis=0)
    sp = lambda a, b: np.corrcoef(rankdata(a), rankdata(b))[0, 1]

    def evaluate(label: str, cols: list[str]) -> dict:
        s, a, b = (-sum(R[k][c] for c in cols) for k in ("full", "half0", "half1"))
        r01, r0v, r1v = sp(a, b), sp(a, lv_rank), sp(b, lv_rank)
        top = np.argsort(s)[::-1][:100]
        return {
            "label": label, "n": len(cols),
            "tau": kendalltau(s, -consensus)[0], "rho": spearmanr(s, lv)[0],
            "rel": r01, "prel": (r01 - r0v * r1v) / np.sqrt((1 - r0v**2) * (1 - r1v**2)),
            "top100_halves": len(set(np.argsort(a)[::-1][:100]) & set(np.argsort(b)[::-1][:100])),
            "top100_lt100": int((cat.vn["votes"].to_numpy()[top] < 100).sum()),
        }

    grid = lambda rks, vs, ex=(): [f"{r}_{v}" for v in vs for r in rks if f"{r}_{v}" not in ex]
    v3 = ["prob", "sp_ari", "sp_geo"]
    combos = {
        "legacy / v1 / v2 grand": grid(["massey", "colley", "markov_rdv", "markov_sdv", "od", "difference"], v3),
        "v3 / v4 grand": grid(["massey", "colley", "markov_rdv", "markov_sdv", "od"], v3, {"massey_prob"}),
        "A  Massey+Colley": grid(["massey", "colley"], v3, {"massey_prob"}),
        "C  A + BT + Elo": grid(["massey", "colley"], v3, {"massey_prob"}) + ["po_bt", "po_elo"],
        "F  Massey+Colley+OD": grid(["massey", "colley", "od"], v3, {"massey_prob"}),
        "G  C + Difference x3": grid(["massey", "colley", "difference"], v3, {"massey_prob"}) + ["po_bt", "po_elo"],
        "G2 (current grand)": list(GRAND_INPUTS),
    }
    singles = pd.DataFrame([evaluate(c, [c]) for c in frames["full"].columns])
    table = pd.DataFrame([evaluate(k, c) for k, c in combos.items()])
    singles.to_csv(args.out / "single_methods.csv", index=False)
    table.to_csv(args.out / "combinations.csv", index=False)
    pd.set_option("display.width", 200)
    print(singles.round(3).to_string(index=False))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
