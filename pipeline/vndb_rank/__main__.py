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
from .analysis import build_analysis, per_user_means, per_user_percentiles, user_attributes
from .cf import CFConfig, build_users, similar_items
from .export import PRODUCER_COLUMNS, VN_COLUMNS, _num, compact_json, rank_table, ranks_json, vn_rows, write_sql
from .extract import load_catalogue, load_users, load_votes
from .methods import GRAND_INPUTS, compute_all
from .neighbors import build_neighbors
from .pairs import build_pairs
from .leaderboards import build_leaderboards
from .tags import load_tags, tags_doc, vn_tag_json, vn_tags_blob, vote_idf
from .history import day_records, encode_day, keep_days, legacy_days, load_days, rank_trend, vn_histories
from .storage import (
    USER_NOTE_SHARDS, VN_NOTE_SHARDS, day_number, name_shards, note_shards,
    text_parts, user_shards, voter_records, voter_shards,
)

log = logging.getLogger("vndb_rank")

# Shown first in the UI and used for the method-agreement matrix.
FEATURED = ["borda_grand", "borda_sci", "borda_po", "vndb", *GRAND_INPUTS]


def kendall_matrix(scores: pd.DataFrame, methods: list[str]) -> dict:
    m = len(methods)
    mat = np.eye(m)
    filled = scores[methods].fillna(-np.inf)
    for i in range(m):
        for j in range(i):
            tau = kendalltau(filled.iloc[:, i], filled.iloc[:, j])[0]
            mat[i, j] = mat[j, i] = round(float(tau), 4) if not np.isnan(tau) else 0.0
    return {"methods": methods, "matrix": mat.tolist()}


def _clean(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def pair_lookup(pairs, n: int):
    """(i, j) -> [voters preferring i, voters preferring j] from the pair table."""
    key = np.minimum(pairs.a, pairs.b).astype(np.int64) * n + np.maximum(pairs.a, pairs.b)
    order = np.argsort(key)
    key = key[order]

    def get(i: int, j: int) -> list[int]:
        k = min(i, j) * n + max(i, j)
        p = np.searchsorted(key, k)
        if p >= len(key) or key[p] != k:
            return [0, 0]
        q = order[p]
        x, y = int(pairs.pv[q]), int(pairs.nv[q])  # pv: a preferred
        return [x, y] if int(pairs.a[q]) == i else [y, x]

    return get


def run(dump_dir: Path, out_dir: Path, cfg: Config, history_path: Path | None = None, legacy_history_path: Path | None = None) -> dict:
    t0 = time.time()
    dump = Dump(dump_dir)
    cat = load_catalogue(dump, cfg)
    n = len(cat.vn)
    ids = cat.ids
    ignored, names = load_users(dump)
    date = dump.snapshot_date()
    votes, stats, vn_labels, extras = load_votes(dump, cat, ignored, year=int(date[:4]), with_notes=not cfg.skip_notes)
    sp = per_user_percentiles(votes)
    tags = load_tags(dump)
    votes_per_vn = np.bincount(votes.vidx, minlength=n)

    analysis = build_analysis(votes, n, vn_labels, sp, per_user_means(votes))
    pairs = build_pairs(votes, n, cfg.min_common_vote)

    scores = compute_all(pairs, n, skip_rankit=cfg.skip_rankit)
    # VNDB's own Bayesian rating as a reference method (average breaks ties).
    scores["vndb"] = cat.vn["rating"].to_numpy(dtype=np.float64)
    vndb_order = cat.vn["rating"].fillna(-1) * 1e6 + cat.vn["average"].fillna(0)
    ranks = rank_table(scores)
    ranks["vndb"] = rank_table(pd.DataFrame({"v": vndb_order.to_numpy()}))["v"].to_numpy()

    # Statistics listed with the methods (own ranking docs) but not part of
    # vn.ranks: mean sample percentile and mean (vote - voter's own mean).
    stat = pd.DataFrame({
        "stat_sp": [a["sp"]["mean"] if a["sp"] else np.nan for a in analysis],
        "stat_bias": [a["bias"] if a["bias"] is not None else np.nan for a in analysis],
    })
    stat_ranks = rank_table(stat)

    neighbors = build_neighbors(pairs, ids, cfg.neighbors_per_category)
    methods = list(scores.columns)
    featured = [m for m in FEATURED if m in methods]
    default = featured[0]
    # Position under the default ranking as a 0-1 percentile (1 = top).
    sci_pct = 1 - (ranks[default].to_numpy(dtype=np.float64) - 1) / max(n - 1, 1)

    cf = CFConfig(min_user_votes=cfg.min_user_votes)
    if cfg.skip_users:
        user_records, item_sims = {}, [[] for _ in range(n)]
    else:
        user_records = build_users(votes, n, names, cf).records
        item_sims = similar_items(votes, n, cf)
    h2h = pair_lookup(pairs, n)
    item_sims = [[[j, s, c, *h2h(i, j)] for j, s, c in sims] for i, sims in enumerate(item_sims)]
    vn_mean = np.array([a["mean"] if a["mean"] is not None else np.nan for a in analysis])
    leaderboards = build_leaderboards(votes, extras, names, vn_mean, set(user_records))
    attrs = user_attributes(votes, extras.user_total, [cat.vn["rating"].to_numpy(dtype=np.float64), sci_pct])
    recs = voter_records(votes.uid, votes.vote.astype(np.uint8), sp, votes.labels, votes.year, attrs)
    voters = list(voter_shards(votes.vidx, recs))
    vote_sp = pd.DataFrame({"uid": votes.uid.astype(np.int64), "vidx": votes.vidx.astype(np.int64), "sp": np.rint(sp * 100)})
    del votes, sp, recs, attrs

    # Notes: per VN (newest first) and per user (only users with a page), with
    # the note's list labels, the author's vote count and the vote's sample percentile.
    nt = extras.notes.merge(vote_sp, on=["uid", "vidx"], how="left")
    nt["nvotes"] = nt["uid"].map(extras.user_total).fillna(0).astype(np.int64)
    nt = nt.sort_values(["vidx", "date"], ascending=[True, False])
    pct = lambda x: None if pd.isna(x) else int(x)
    vn_note_rows = [
        [int(i), int(u), names.get(int(u), ""), int(v), int(d), t, int(u) in user_records, int(lb), int(nv), pct(p)]
        for i, u, v, d, t, lb, nv, p in zip(nt["vidx"], nt["uid"], nt["vote"], nt["date"], nt["text"], nt["labels"], nt["nvotes"], nt["sp"])
    ]
    nt = nt[nt["uid"].isin(list(user_records))].sort_values(["uid", "date"], ascending=[True, False])
    user_note_rows = [
        [int(u), int(i), int(v), int(d), t, int(lb), pct(p)]
        for u, i, v, d, t, lb, p in zip(nt["uid"], nt["vidx"], nt["vote"], nt["date"], nt["text"], nt["labels"], nt["sp"])
    ]
    note_count = np.bincount(extras.notes["vidx"].to_numpy(dtype=np.int64), minlength=n) if len(extras.notes) else np.zeros(n, int)
    for i, a in enumerate(analysis):
        a["notes"] = int(note_count[i])
    del extras

    today = day_number(date)
    # history_day as fetched from D1; the old per-VN column only before the first migration.
    stored = load_days(history_path)
    days = dict(stored) if stored else legacy_days(legacy_history_path)
    if cfg.skip_rankit:
        log.info("history: --skip-rankit, today is not recorded (the default method differs)")
    else:
        days[today] = day_records(ids, ranks[default].to_numpy(), ranks["vndb"].to_numpy(), cat.vn["rating"].to_numpy(dtype=np.float64), cat.vn["votes"].to_numpy())
    kept = keep_days(days.keys(), today)
    days = {d: days[d] for d in sorted(kept)}
    history = vn_histories(days, ids)
    new_days = [d for d in days if d not in stored or d == today]
    history_update = (
        "day",
        [d for d in stored if d not in kept] + new_days,
        ["day", "part", "data"],
        [[d, part, piece] for d in new_days for part, piece in enumerate(text_parts(encode_day(days[d])))],
    )

    for table in ["vn", "releases", "producers", "users", "chars", "staff", "tags", "traits"]:
        if dump.has(table):
            stats.setdefault("tables", {})[table] = dump.line_count(table)
    stats["ranked_vns"] = n
    stats["pairs"] = len(pairs)
    stats["user_pages"] = len(user_records)
    stats["ignored_users"] = len(ignored)
    stats["notes"] = len(vn_note_rows)

    snapshot = f"{date.replace('-', '')}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
    meta = {
        "info": {
            "dump_date": date,
            "day": today,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": __version__,
            "config": cfg.as_dict(),
            "methods": methods + list(stat.columns),
            "featured": featured,
            "default_method": default,
            "history_methods": [default, "vndb"],
            "history_days": list(days),
        },
        "stats": stats,
        "kendall": kendall_matrix(scores, featured),
        "leaderboards": leaderboards,
        "snapshot": snapshot,
    }

    extra = {
        "idx": list(range(n)),
        "trend": [rank_trend(h, today) for h in history],
        "ranks": ranks_json(scores, ranks),
        "neighbors": [compact_json(neighbors.get(int(v), [])) for v in ids],
        "relations": [compact_json(cat.relations.get(int(v), [])) for v in ids],
        "analysis": [compact_json(a) for a in analysis],
        "similar": [compact_json([[int(ids[j]), s, c, w, l] for j, s, c, w, l in sims]) for sims in item_sims],
        "history": [compact_json(h) for h in history],
        "tags": [compact_json(t) for t in vn_tag_json(tags, ids)],
    }
    # Prebuilt API payloads, served by the worker without parsing.
    vndb_rank = ranks["vndb"].to_numpy()
    sci_rank = ranks[default].to_numpy()
    cat_cols = ["id", "idx", "title", "latin", "title_ja", "title_zh", "title_en", "olang", "released",
                "dev_id", "dev", "dev_latin", "votes", "rating", "length", "trend", "vndb_rank", "sci_rank", "search"]
    dev_name = cat.producers.set_index("id")["name"].to_dict()
    dev_latin = cat.producers.set_index("id")["latin"].to_dict()
    cat_rows = []
    for i, r in enumerate(cat.vn.itertuples(index=False)):
        d = None if pd.isna(r.dev_id) else int(r.dev_id)
        lat = dev_latin.get(d) if d is not None else None
        cat_rows.append([
            int(r.id), i, r.title, _clean(r.latin), _clean(r.title_ja), _clean(r.title_zh), _clean(r.title_en), _clean(r.olang),
            None if pd.isna(r.released) else int(r.released), d, dev_name.get(d) if d is not None else None,
            None if lat is None or pd.isna(lat) else lat, int(r.votes), _clean(r.rating), _clean(r.length),
            extra["trend"][i], int(vndb_rank[i]), int(sci_rank[i]), r.search,
        ])
    docs = {"catalogue": compact_json({"columns": cat_cols, "rows": cat_rows})}
    for m, s_col, r_col in [*((m, scores[m], ranks[m]) for m in methods), *((m, stat[m], stat_ranks[m]) for m in stat.columns)]:
        order = np.lexsort((ids, r_col.to_numpy()))
        sc = s_col.to_numpy(dtype=np.float64)
        docs[f"ranks:{m}"] = compact_json({"method": m, "ranks": [[int(ids[j]), int(r_col.iloc[j]), _num(sc[j])] for j in order]})
    if tags is not None:
        docs["tags"] = compact_json(tags_doc(tags, vote_idf(tags, ids, votes_per_vn)))
        docs["vntags"] = vn_tags_blob(tags, ids)
    doc_rows = [[k, part, piece] for k, text in docs.items() for part, piece in enumerate(text_parts(text))]

    tables = {
        "vn": (VN_COLUMNS, vn_rows(cat.vn, extra)),
        "producer": (PRODUCER_COLUMNS, [[int(p.id), p.name, None if pd.isna(p.latin) else p.latin] for p in cat.producers.itertuples(index=False)]),
        "user_block": (["shard", "part", "data"], user_shards(user_records)),
        "user_name": (["shard", "part", "data"], name_shards(user_records)),
        "vn_voters": (["shard", "part", "data"], voters),
        "vn_notes": (["shard", "part", "data"], note_shards(vn_note_rows, 0, VN_NOTE_SHARDS)),
        "user_notes": (["shard", "part", "data"], note_shards(user_note_rows, 0, USER_NOTE_SHARDS)),
        "doc": (["key", "part", "data"], doc_rows),
        "meta": (["key", "value"], [[k, compact_json(v)] for k, v in meta.items()]),
    }
    counts = write_sql(out_dir / "snapshot.sql", tables, {"history_day": history_update})
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
    ap.add_argument("--skip-notes", action="store_true", help="do not export users' list notes")
    ap.add_argument("--history", type=Path, default=None, help="`SELECT day, part, data FROM history_day` as JSON (wrangler --json output, blocks concatenated)")
    ap.add_argument("--legacy-history", type=Path, default=None, help="`SELECT id, history FROM vn` from before history_day existed (used only when --history is empty)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config(
        min_vote=args.min_vote, min_common_vote=args.min_common_vote,
        neighbors_per_category=args.neighbors, skip_rankit=args.skip_rankit,
        min_user_votes=args.min_user_votes, skip_users=args.skip_users, skip_notes=args.skip_notes,
    )
    run(args.dump, args.out, cfg, args.history, args.legacy_history)


if __name__ == "__main__":
    main()
