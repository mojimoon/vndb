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
from vndb_rank.export import insert_statements, MAX_STATEMENT_BYTES
from vndb_rank.extract import Votes, normalize_search, to_ten_scale
from vndb_rank.methods import po_bradley_terry, po_classical, po_elo, po_entropy, po_random_walk
from vndb_rank.neighbors import build_neighbors
from vndb_rank.pairs import build_pairs, sample_percentile

from fake_dump import make_fake_dump

MIGRATION = Path(__file__).resolve().parents[2] / "web" / "migrations" / "0001_init.sql"


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
    return Votes(uid=arr[:, 0].astype(np.int32), vidx=arr[:, 1].astype(np.int32), vote=arr[:, 2].astype(np.int16))


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
    for fn in (po_bradley_terry, po_elo, po_entropy, po_random_walk):
        s = fn(p, 3)
        assert s[0] > s[1] > s[2], fn.__name__


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
    return tmp / "out" / "snapshot.sql", summary


def test_end_to_end_loads_into_sqlite(snapshot):
    sql, summary = snapshot
    db = sqlite3.connect(":memory:")
    db.executescript(MIGRATION.read_text())
    db.executescript(sql.read_text())
    n = db.execute("SELECT count(*) FROM vn").fetchone()[0]
    assert n == summary["rows"]["vn"] > 50
    snap = json.loads(db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0])
    assert snap == summary["snapshot"]
    info = json.loads(db.execute("SELECT value FROM meta WHERE key='info'").fetchone()[0])
    assert "borda_grand" in info["methods"] and info["default_method"] == "borda_grand"

    # The query the worker runs for the ranking list.
    rows = db.execute(
        "SELECT id, json_extract(ranks, '$.po_percent[0]') AS r FROM vn ORDER BY r"
    ).fetchall()
    assert [r[1] for r in rows][:3] == [1, 2, 3]
    assert all(db.execute("SELECT count(*) FROM producer WHERE id = ?", (d,)).fetchone()[0] == 1
               for (d,) in db.execute("SELECT DISTINCT dev_id FROM vn WHERE dev_id IS NOT NULL"))

    # Re-importing a newer snapshot replaces everything.
    db.executescript(sql.read_text().replace(summary["snapshot"], "next"))
    assert db.execute("SELECT count(DISTINCT snapshot) FROM vn").fetchone()[0] == 1


def test_rankings_track_latent_quality(snapshot):
    sql, _ = snapshot
    db = sqlite3.connect(":memory:")
    db.executescript(MIGRATION.read_text())
    db.executescript(sql.read_text())
    rows = db.execute("SELECT rating, ranks FROM vn").fetchall()
    rating = [r[0] for r in rows]
    for m in ["po_percent", "po_bt", "massey_prob", "borda_grand"]:
        rank = [json.loads(r[1])[m][0] for r in rows]
        rho = spearmanr(rating, rank)[0]
        assert rho < -0.6, (m, rho)  # better rating -> smaller rank number
