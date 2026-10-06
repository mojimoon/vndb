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
from .analysis import build_analysis, per_user_means, per_user_percentiles
from .cf import CFConfig, build_users, similar_items
from .export import PRODUCER_COLUMNS, VN_COLUMNS, compact_json, rank_table, ranks_json, vn_rows, write_sql
from .extract import load_catalogue, load_users, load_votes
from .methods import compute_all
from .neighbors import build_neighbors
from .pairs import build_pairs
from .storage import day_number, merge_history, name_shards, pair_blocks, rank_trend, user_shards

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


def load_previous_history(path: Path | None) -> dict[int, list]:
    """Accept `wrangler d1 execute --json` output or a plain list of {id, history} rows."""
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "[]")
    except json.JSONDecodeError:
        log.warning("history: %s is not JSON, starting fresh", path)
        return {}
    rows = data[0].get("results", []) if data and isinstance(data[0], dict) and "results" in data[0] else data
    out = {}
    for r in rows:
        try:
            out[int(r["id"])] = json.loads(r["history"]) if isinstance(r["history"], str) else r["history"]
        except (KeyError, TypeError, ValueError):
            continue
    log.info("history: %d VNs from %s", len(out), path)
    return out


def run(dump_dir: Path, out_dir: Path, cfg: Config, history_path: Path | None = None) -> dict:
    t0 = time.time()
    dump = Dump(dump_dir)
    cat = load_catalogue(dump, cfg)
    n = len(cat.vn)
    ids = cat.ids
    ignored, names = load_users(dump)
    votes, stats, vn_labels = load_votes(dump, cat, ignored)

    analysis = build_analysis(votes, n, vn_labels, per_user_percentiles(votes), per_user_means(votes))
    pairs = build_pairs(votes, n, cfg.min_common_vote)

    scores = compute_all(pairs, n, skip_rankit=cfg.skip_rankit)
    # VNDB's own Bayesian rating as a reference method (average breaks ties).
    scores["vndb"] = cat.vn["rating"].to_numpy(dtype=np.float64)
    vndb_order = cat.vn["rating"].fillna(-1) * 1e6 + cat.vn["average"].fillna(0)
    ranks = rank_table(scores)
    ranks["vndb"] = rank_table(pd.DataFrame({"v": vndb_order.to_numpy()}))["v"].to_numpy()

    neighbors = build_neighbors(pairs, ids, cfg.neighbors_per_category)

    cf = CFConfig(min_user_votes=cfg.min_user_votes)
    if cfg.skip_users:
        user_records, item_sims = {}, [[] for _ in range(n)]
    else:
        user_records = build_users(votes, n, names, cf).records
        item_sims = similar_items(votes, n, cf)
    del votes

    methods = list(scores.columns)
    featured = [m for m in FEATURED if m in methods]
    default = featured[0]
    date = dump.snapshot_date()
    today = day_number(date)
    prev = load_previous_history(history_path)
    history = [merge_history(prev.get(int(vid)), today, [int(ranks[default].iloc[i]), int(ranks["vndb"].iloc[i])]) for i, vid in enumerate(ids)]

    for table in ["vn", "releases", "producers", "users", "chars", "staff", "tags", "traits"]:
        if dump.has(table):
            stats.setdefault("tables", {})[table] = dump.line_count(table)
    stats["ranked_vns"] = n
    stats["pairs"] = len(pairs)
    stats["user_pages"] = len(user_records)
    stats["ignored_users"] = len(ignored)

    snapshot = f"{date.replace('-', '')}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
    meta = {
        "info": {
            "dump_date": date,
            "day": today,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": __version__,
            "config": cfg.as_dict(),
            "methods": methods,
            "featured": featured,
            "default_method": default,
            "history_methods": [default, "vndb"],
        },
        "stats": stats,
        "kendall": kendall_matrix(scores, featured),
        "snapshot": snapshot,
    }

    extra = {
        "idx": list(range(n)),
        "trend": [rank_trend(h, today) for h in history],
        "ranks": ranks_json(scores, ranks),
        "neighbors": [compact_json(neighbors.get(int(v), [])) for v in ids],
        "relations": [compact_json(cat.relations.get(int(v), [])) for v in ids],
        "analysis": [compact_json(a) for a in analysis],
        "similar": [compact_json([[int(ids[j]), s, c] for j, s, c in sims]) for sims in item_sims],
        "history": [compact_json(h) for h in history],
    }
    tables = {
        "vn": (VN_COLUMNS, vn_rows(cat.vn, extra)),
        "producer": (PRODUCER_COLUMNS, [[int(p.id), p.name, None if pd.isna(p.latin) else p.latin] for p in cat.producers.itertuples(index=False)]),
        "pair_block": (["a", "part", "data"], pair_blocks(pairs)),
        "user_block": (["shard", "part", "data"], user_shards(user_records)),
        "user_name": (["shard", "part", "data"], name_shards(user_records)),
        "meta": (["key", "value"], [[k, compact_json(v)] for k, v in meta.items()]),
    }
    counts = write_sql(out_dir / "snapshot.sql", tables)
    size = (out_dir / "snapshot.sql").stat().st_size
    summary = {
        "snapshot": snapshot, "rows": counts, "rows_total": sum(counts.values()), "pairs": len(pairs),
        "user_pages": len(user_records), "sql_mb": round(size / 1e6, 1), "seconds": round(time.time() - t0, 1),
    }
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
    ap.add_argument("--min-user-votes", type=int, default=Config.min_user_votes)
    ap.add_argument("--skip-rankit", action="store_true", help="only compute the PONet methods (fast)")
    ap.add_argument("--skip-users", action="store_true", help="skip user pages, recommendations and similar VNs")
    ap.add_argument("--history", type=Path, default=None, help="previous `SELECT id, history FROM vn` as JSON (wrangler --json output)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config(
        min_vote=args.min_vote, min_common_vote=args.min_common_vote,
        neighbors_per_category=args.neighbors, skip_rankit=args.skip_rankit,
        min_user_votes=args.min_user_votes, skip_users=args.skip_users,
    )
    run(args.dump, args.out, cfg, args.history)


if __name__ == "__main__":
    main()
