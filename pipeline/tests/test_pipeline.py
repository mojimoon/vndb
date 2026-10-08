import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vndb_rank.__main__ import run
from vndb_rank.config import Config
from vndb_rank.cf import decode_votes
from vndb_rank.export import insert_statements, MAX_STATEMENT_BYTES
from vndb_rank.storage import fnv1a, merge_history, rank_trend
from vndb_rank.extract import Votes, normalize_search, to_ten_scale
from vndb_rank.methods import po_bradley_terry, po_classical, po_elo, po_entropy
from vndb_rank.neighbors import build_neighbors
from vndb_rank.pairs import build_pairs, sample_percentile

from fake_dump import make_fake_dump



def brute_force(rows, n):
    """rows: list of (uid, item, vote). Returns {(a, b): (pv, nv, tv)} for a < b."""
    by_user = {}
    for u, i, v in rows:
        by_user.setdefault(u, {})[i] = v
    out = {}
    for items in by_user.values():
        for a, b in combinations(sorted(items), 2):
            pv, nv, tv = out.get((a, b), (0, 0, 0))
            va, vb = items[a], items[b]
            out[(a, b)] = (pv + (va > vb), nv + (va < vb), tv + 1)
    return out


def make_votes(rows):
    arr = np.array(rows)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return Votes(uid=arr[:, 0].astype(np.int32), vidx=arr[:, 1].astype(np.int32), vote=arr[:, 2].astype(np.int16),
                 year=np.full(len(arr), 2020, dtype=np.int16))


def test_pairs_match_brute_force():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(60):
        for i in rng.choice(8, size=rng.integers(1, 8), replace=False):
            rows.append((u, int(i), int(rng.integers(1, 11) * 10)))
    p = build_pairs(make_votes(rows), 8, min_common=1)
    expected = brute_force(rows, 8)
    got = {(int(a), int(b)): (int(x), int(y), int(t)) for a, b, x, y, t in zip(p.a, p.b, p.pv, p.nv, p.tv)}
    assert got == expected
    assert (p.a < p.b).all()


def test_pairs_mean_scores():
    # Two users, both voted 0 and 1.
    rows = [(1, 0, 80), (1, 1, 60), (2, 0, 40), (2, 1, 100)]
    p = build_pairs(make_votes(rows), 2, min_common=1)
    assert np.allclose(p.ari[0], [0.6, 0.8])
    assert np.allclose(p.geo[0], [np.sqrt(0.8 * 0.4), np.sqrt(0.6 * 1.0)])
    assert np.allclose(p.sp_ari[0], [0.5, 0.5])


def test_sample_percentile_formula():
    # README: sp(x_k) = (#{x_i < x_k} + 0.5 * #{x_i = x_k, i != k} + 1) / (n + 1)
    v = np.array([50, 70, 70, 90])
    expected = [(0 + 0 + 1) / 5, (1 + 0.5 + 1) / 5, (1 + 0.5 + 1) / 5, (3 + 0 + 1) / 5]
    assert np.allclose(sample_percentile(v), expected)


def test_methods_prefer_dominant_item():
    # Item 0 beats everyone, item 2 loses to everyone.
    rows = []
    for u in range(20):
        rows += [(u, 0, 90), (u, 1, 70), (u, 2, 50)]
    p = build_pairs(make_votes(rows), 3, min_common=5)
    for name, s in po_classical(p, 3).items():
        assert s[0] > s[1] > s[2], name
    for fn in (po_bradley_terry, po_elo, po_entropy):
        s = fn(p, 3)
        assert s[0] > s[1] > s[2], fn.__name__


def test_grand_ranking_is_borda_of_g2_inputs():
    from vndb_rank.methods import GRAND_INPUTS, borda, compute_all, rankit_scores
    rng = np.random.default_rng(3)
    quality = rng.normal(size=12)
    rows = []
    for u in range(80):
        bias = rng.normal()
        for i in rng.choice(12, size=8, replace=False):
            rows.append((u, int(i), int(np.clip(np.rint(6 + 1.5 * quality[i] + bias + rng.normal()), 1, 10) * 10)))
    p = build_pairs(make_votes(rows), 12, min_common=5)
    df = compute_all(p, 12)
    assert len(GRAND_INPUTS) == 9 and set(GRAND_INPUTS) <= set(df.columns)
    assert not {"difference_prob", "difference_ari", "difference_geo"} & set(df.columns)
    assert np.allclose(df["borda_grand"], borda(df[GRAND_INPUTS]))
    assert np.allclose(df["difference_sp_ari"], rankit_scores(p, 12, "sp_ari", "difference"), equal_nan=True)
    assert spearmanr(df["borda_grand"], quality)[0] > 0.7


def test_neighbors_union_contains_each_category_top():
    rows = []
    rng = np.random.default_rng(3)
    for u in range(200):
        for i in rng.choice(30, size=10, replace=False):
            rows.append((u, int(i), int(rng.integers(1, 11) * 10)))
    p = build_pairs(make_votes(rows), 30, min_common=1)
    nb = build_neighbors(p, np.arange(100, 130), k=3)
    assert set(nb) <= set(range(100, 130))
    for vid, lst in nb.items():
        opp = [r[0] for r in lst]
        assert len(opp) == len(set(opp)) and vid not in opp
        assert all(r[1] + r[2] <= r[3] for r in lst)
        # the 3 most co-voted opponents must be present
        mine = p.tv[(p.a == vid - 100) | (p.b == vid - 100)]
        assert sorted(r[3] for r in lst)[-3:] == sorted(mine)[-3:]


def test_chinese_titles():
    from vndb_rank.extract import chinese_alias, clean_release_title
    # Latin-script original: aliases only if unambiguous or confirmed by a release.
    assert chinese_alias("リライト\nRiraito\n罚抄\n改写", "Rewrite") is None
    assert chinese_alias("リライト\n罚抄\n改写", "Rewrite", "改写 体验版") == "改写"
    assert chinese_alias("fsn\nフェイト／ステイナイト\n命运之夜", "Fate/stay night") == "命运之夜"
    # Kana original: the first Chinese alias is better than nothing.
    assert chinese_alias("dc\n初音岛\n初音島", "D.C.～ダ・カーポ～") == "初音岛"
    assert chinese_alias("なつくる\nNatsukuru", "なつくもゆるる") is None
    assert clean_release_title("花冠之淚") == "花冠之淚"
    assert clean_release_title("心靈判官: 無法抉擇的幸福 (中文版)") == "心靈判官: 無法抉擇的幸福"
    assert clean_release_title("网球王牌 0.62") == "网球王牌"
    assert clean_release_title("我的爱人是霸凌女 - 第一章第四集") == "我的爱人是霸凌女"
    assert clean_release_title("Fate/Extella 下載版") is None  # mixed product names are skipped
    assert clean_release_title("Rewrite") is None


def test_helpers():
    assert normalize_search("Steins;Gate", "シュタインズ・ゲート", None, float("nan")) == "steinsgate シュタインズゲート"
    assert to_ten_scale(pd.Series(["850", "\\N", "100"])).round(2).tolist()[::2] == [8.5, 1.0]
    stmts = insert_statements("t", ["a"], [["x" * 1000] for _ in range(500)])
    assert len(stmts) > 1 and all(len(s.encode()) < MAX_STATEMENT_BYTES + 2000 for s in stmts)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    dump = make_fake_dump(tmp / "dump", n_vns=150, n_users=1500, seed=7)
    summary = run(dump, tmp / "out", Config())
    db = sqlite3.connect(":memory:")
    db.executescript((tmp / "out" / "snapshot.sql").read_text())
    return tmp / "out" / "snapshot.sql", summary, db


def test_end_to_end_loads_into_sqlite(snapshot):
    sql, summary, db = snapshot
    n = db.execute("SELECT count(*) FROM vn").fetchone()[0]
    assert n == summary["rows"]["vn"] > 50
    assert db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE '%__next'").fetchone()[0] == 0
    snap = json.loads(db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0])
    assert snap == summary["snapshot"]
    info = json.loads(db.execute("SELECT value FROM meta WHERE key='info'").fetchone()[0])
    assert "borda_grand" in info["methods"] and info["default_method"] == "borda_grand"
    assert sorted(r[0] for r in db.execute("SELECT idx FROM vn")) == list(range(n))

    # The query the worker runs for a method's ranks.
    rows = db.execute("SELECT id, json_extract(ranks, '$.po_percent[0]') AS r FROM vn ORDER BY r").fetchall()
    assert [r[1] for r in rows][:3] == [1, 2, 3]
    assert all(db.execute("SELECT count(*) FROM producer WHERE id = ?", (d,)).fetchone()[0] == 1
               for (d,) in db.execute("SELECT DISTINCT dev_id FROM vn WHERE dev_id IS NOT NULL"))

    # Re-importing over an existing database replaces the tables cleanly, and
    # reuses the freed pages instead of holding two copies at once.
    pages = db.execute("PRAGMA page_count").fetchone()[0]
    db.executescript(sql.read_text())
    assert db.execute("SELECT count(*) FROM vn").fetchone()[0] == n
    assert db.execute("PRAGMA page_count").fetchone()[0] < pages * 1.15


def test_rankings_track_latent_quality(snapshot):
    _, _, db = snapshot
    rows = db.execute("SELECT rating, ranks FROM vn").fetchall()
    rating = [r[0] for r in rows]
    for m in ["po_percent", "po_bt", "massey_sp_ari", "borda_grand"]:
        rank = [json.loads(r[1])[m][0] for r in rows]
        rho = spearmanr(rating, rank)[0]
        assert rho < -0.6, (m, rho)  # better rating -> smaller rank number


def test_vn_analysis_and_history(snapshot):
    _, _, db = snapshot
    info = json.loads(db.execute("SELECT value FROM meta WHERE key='info'").fetchone()[0])
    for analysis, history, ranks in db.execute("SELECT analysis, history, ranks FROM vn LIMIT 20"):
        a = json.loads(analysis)
        assert sum(a["hist"]) == a["n"] > 0 and 1 <= a["mean"] <= 10
        assert sum(a["sp"]["hist"]) == a["n"] and 0 < a["sp"]["mean"] < 1
        h = json.loads(history)
        assert h[-1] == [info["day"], json.loads(ranks)["borda_grand"][0], json.loads(ranks)["vndb"][0]]


def test_users_and_recommendations(snapshot):
    _, summary, db = snapshot
    assert summary["user_pages"] > 100
    n_vn = db.execute("SELECT count(*) FROM vn").fetchone()[0]
    users = {}
    for (data,) in db.execute("SELECT data FROM user_block"):
        users.update(json.loads(data))
    assert len(users) == summary["user_pages"]
    with_recs = 0
    for uid, rec in users.items():
        idx, vote = decode_votes(rec["votes"])
        assert len(idx) >= 5 and (np.diff(idx) > 0).all() and ((10 <= vote) & (vote <= 100)).all() and idx.max() < n_vn
        assert all(r[0] not in set(idx.tolist()) for r in rec["recs"])  # never recommend what they voted on
        assert all(s[0] != int(uid) for s in rec["similar"])
        # head-to-head over the common titles: higher + equal + lower == common
        assert all(s[4] + s[5] + s[6] == s[3] and s[7] >= s[3] for s in rec["similar"])
        with_recs += bool(rec["recs"])
    assert with_recs > len(users) * 0.5
    # name index resolves every user
    names = {}
    for (data,) in db.execute("SELECT data FROM user_name"):
        names.update(json.loads(data))
    some = next(iter(users.items()))
    assert names[some[1]["name"].lower()] == int(some[0])


def test_similar_items_are_valid(snapshot):
    _, _, db = snapshot
    ids = {r[0] for r in db.execute("SELECT id FROM vn")}
    nonempty = 0
    for vid, sim in db.execute("SELECT id, similar FROM vn"):
        s = json.loads(sim)
        nonempty += bool(s)
        assert all(o in ids and o != vid and 0 < x <= 1 and w + l <= c for o, x, c, w, l in s)
    assert nonempty > len(ids) * 0.5


def test_storage_helpers():
    assert fnv1a("a") == 0xE40C292C
    today = 10_000
    hist = [[today - d, d, d] for d in range(800, 0, -1)]
    merged = merge_history(hist, today, [1, 2])
    assert merged[-1] == [today, 1, 2]
    assert len([p for p in merged if today - p[0] <= 90]) == 91
    assert all(today - p[0] <= 730 for p in merged)
    assert rank_trend([[today - 8, 10, 0], [today, 4, 0]], today) == 6


def _doc(db, key):
    return json.loads("".join(r[0] for r in db.execute("SELECT data FROM doc WHERE key = ? ORDER BY part", (key,))))


def test_prebuilt_docs(snapshot):
    _, summary, db = snapshot
    cat = _doc(db, "catalogue")
    assert len(cat["rows"]) == summary["rows"]["vn"]
    row = dict(zip(cat["columns"], cat["rows"][0]))
    assert row["idx"] == 0 and row["title"] and "search" in row
    info = json.loads(db.execute("SELECT value FROM meta WHERE key='info'").fetchone()[0])
    removed = {"po_rw", "massey_prob", "keener_prob", "markov_rv_sp_geo", "difference_ari", "difference_prob"}
    assert not removed & set(info["methods"]) and len(info["methods"]) == 42  # 41 methods + vndb
    assert {"difference_sp_ari", "difference_sp_geo"} <= set(info["methods"])
    for m in ["borda_grand", "vndb"]:
        r = _doc(db, f"ranks:{m}")
        assert r["method"] == m and [x[1] for x in r["ranks"]] == sorted(x[1] for x in r["ranks"])
        sql = dict(db.execute(f"SELECT id, json_extract(ranks, '$.{m}[0]') FROM vn"))
        assert all(sql[i] == rank for i, rank, _ in r["ranks"])


def test_voter_shards_give_exact_joint_counts(snapshot):
    _, _, db = snapshot
    from vndb_rank.storage import VOTER_SHARDS, read_voters
    parts = lambda idx: [bytes(r[0]) for r in db.execute("SELECT data FROM vn_voters WHERE shard = ? ORDER BY part", (idx % VOTER_SHARDS,))]
    nb = json.loads(db.execute("SELECT neighbors FROM vn WHERE idx = 0").fetchone()[0])
    idx = dict(db.execute("SELECT id, idx FROM vn"))
    a = read_voters(parts(0), 0)
    n_votes = json.loads(db.execute("SELECT analysis FROM vn WHERE idx = 0").fetchone()[0])["n"]
    assert len(a) == n_votes and (np.diff(a["uid"].astype(np.int64)) > 0).all()
    assert (a["sp"] <= 199).all() and ((a["vote"] >= 10) & (a["vote"] <= 100)).all()
    assert (a["labels"] < 64).all() and (a["nvotes"] >= 1).all() and ((a["umean"] >= 10) & (a["umean"] <= 100)).all()
    assert ((a["cv"] >= -100) | (a["cv"] == -128)).all() and (a["cv"] <= 100).all() and (a["cs"] != -128).any()
    # every voter's ranked votes share the same user attributes across VNs
    a_attr = dict(zip(a["uid"].tolist(), zip(a["nvotes"].tolist(), a["cs"].tolist())))
    other, wins, losses, common = nb[0]
    b = read_voters(parts(idx[other]), idx[other])
    _, ia, ib = np.intersect1d(a["uid"], b["uid"], return_indices=True)
    assert len(ia) == common
    assert (a["vote"][ia] > b["vote"][ib]).sum() == wins and (a["vote"][ia] < b["vote"][ib]).sum() == losses
    assert all(a_attr[u] == (n, c) for u, n, c in zip(b["uid"].tolist(), b["nvotes"].tolist(), b["cs"].tolist()) if u in a_attr)


def test_notes_and_leaderboards(snapshot):
    _, summary, db = snapshot
    rows = [r for (d,) in db.execute("SELECT data FROM vn_notes") for r in json.loads(d)]
    assert rows and all(len(r[5]) >= 20 and len(r) == 10 for r in rows)
    assert all(0 <= r[7] < 64 and r[8] >= 0 and (r[9] is None or 0 <= r[9] <= 100) for r in rows)
    assert any(r[7] for r in rows) and any(r[9] is not None for r in rows)
    assert all(r[9] is None for r in rows if r[3] == 0)  # no vote, no percentile
    users = [r for (d,) in db.execute("SELECT data FROM user_notes") for r in json.loads(d)]
    assert 0 < len(users) <= len(rows) and all(len(r) == 7 for r in users)
    lb = json.loads(db.execute("SELECT value FROM meta WHERE key='leaderboards'").fetchone()[0])
    assert lb["most_votes"][0][2] >= lb["most_votes"][-1][2]
    assert lb["highest_mean"][0][2] >= lb["lowest_mean"][0][2]
    assert all(e[3] >= lb["min_votes"] for e in lb["highest_mean"])
    assert lb["most_mainstream"][0][2] >= lb["most_contrarian"][0][2]


def test_text_parts_split_on_utf8_boundaries():
    from vndb_rank.storage import PART_TEXT_BYTES, text_parts
    s = "天" * (PART_TEXT_BYTES // 2) + "abc"
    parts = list(text_parts(s))
    assert "".join(parts) == s and all(len(p.encode()) <= PART_TEXT_BYTES for p in parts)
