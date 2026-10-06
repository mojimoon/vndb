"""CLI: python -m vndb_rank --dump ../db --out out/"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import __version__
from .config import Config
from .dump import Dump
from .export import rank_table, ranks_json, write_sql
from .extract import load_catalogue, load_votes
from .methods import compute_all
from .neighbors import build_neighbors
from .pairs import build_pairs

log = logging.getLogger("vndb_rank")

# Shown first in the UI and used for the method-agreement matrix.
FEATURED = [
    "borda_grand", "borda_sci", "borda_po", "po_percent", "po_simple", "po_weighted",
    "po_bt", "po_elo", "po_rw", "po_entropy", "massey_prob", "colley_prob", "markov_rdv_sp_geo", "vndb",
]


def kendall_matrix(scores: pd.DataFrame, methods: list[str]) -> dict:
    m = len(methods)
    mat = np.eye(m)
    filled = scores[methods].fillna(-np.inf)
    for i in range(m):
        for j in range(i):
            tau = kendalltau(filled.iloc[:, i], filled.iloc[:, j])[0]
            mat[i, j] = mat[j, i] = round(float(tau), 4) if not np.isnan(tau) else 0.0
    return {"methods": methods, "matrix": mat.tolist()}


def run(dump_dir: Path, out_dir: Path, cfg: Config) -> dict:
    t0 = time.time()
    dump = Dump(dump_dir)
    cat = load_catalogue(dump, cfg)
    n = len(cat.vn)
    votes, stats = load_votes(dump, cat)
    pairs = build_pairs(votes, n, cfg.min_common_vote)
    del votes

    scores = compute_all(pairs, n, skip_rankit=cfg.skip_rankit)
    # VNDB's own Bayesian rating as a reference method (average breaks ties).
    scores["vndb"] = cat.vn["rating"].to_numpy(dtype=np.float64)
    vndb_order = cat.vn["rating"].fillna(-1) * 1e6 + cat.vn["average"].fillna(0)
    ranks = rank_table(scores)
    ranks["vndb"] = rank_table(pd.DataFrame({"v": vndb_order.to_numpy()}))["v"].to_numpy()

    neighbors = build_neighbors(pairs, cat.ids, cfg.neighbors_per_category)

    for table in ["vn", "releases", "producers", "users", "chars", "staff", "tags", "traits"]:
        if dump.has(table):
            stats.setdefault("tables", {})[table] = dump.line_count(table)
    stats["ranked_vns"] = n
    stats["pairs"] = len(pairs)

    methods = list(scores.columns)
    featured = [m for m in FEATURED if m in methods]
    date = dump.snapshot_date()
    snapshot = f"{date.replace('-', '')}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
    meta = {
        "info": {
            "dump_date": date,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": __version__,
            "config": cfg.as_dict(),
            "methods": methods,
            "featured": featured,
            "default_method": featured[0],
        },
        "stats": stats,
        "kendall": kendall_matrix(scores, featured),
    }
    counts = write_sql(
        out_dir / "snapshot.sql", snapshot, cat.vn, cat.producers,
        ranks_json(scores, ranks), neighbors, cat.relations, meta,
    )
    summary = {"snapshot": snapshot, "rows": counts, "pairs": len(pairs), "seconds": round(time.time() - t0, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="vndb_rank", description=__doc__)
    ap.add_argument("--dump", type=Path, default=Path("../db"), help="extracted VNDB dump (folder containing db/)")
    ap.add_argument("--out", type=Path, default=Path("out"), help="output folder for snapshot.sql")
    ap.add_argument("--min-vote", type=int, default=Config.min_vote)
    ap.add_argument("--min-common-vote", type=int, default=Config.min_common_vote)
    ap.add_argument("--neighbors", type=int, default=Config.neighbors_per_category)
    ap.add_argument("--skip-rankit", action="store_true", help="only compute the PONet methods (fast)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config(
        min_vote=args.min_vote, min_common_vote=args.min_common_vote,
        neighbors_per_category=args.neighbors, skip_rankit=args.skip_rankit,
    )
    run(args.dump, args.out, cfg)


if __name__ == "__main__":
    main()
