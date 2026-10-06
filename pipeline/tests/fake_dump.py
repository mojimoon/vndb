"""Generate a small synthetic dump in VNDB's format (db/<table> + .header).

Used by the tests and to produce seed data for local frontend development:

    python tests/fake_dump.py /tmp/fake-dump --vns 300 --users 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SYLLABLES = ["sora", "no", "hana", "yume", "koi", "hoshi", "kimi", "natsu", "aoi", "tsuki", "kaze", "uta", "mirai", "shiro"]
KANA = ["そら", "の", "はな", "ゆめ", "こい", "ほし", "きみ", "なつ", "あおい", "つき", "かぜ", "うた"]
HAN = ["天", "空", "花", "梦", "恋", "星", "君", "夏", "青", "月", "风", "歌"]


def _write(root: Path, table: str, header: list[str], rows: list[list[str]]) -> None:
    (root / f"{table}.header").write_text("\t".join(header) + "\n", encoding="utf-8")
    with open(root / table, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")


def make_fake_dump(out: Path, n_vns: int = 300, n_users: int = 2000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    db = out / "db"
    db.mkdir(parents=True, exist_ok=True)
    (out / "TIMESTAMP").write_text("2026-10-05T04:00:00Z\n")

    n_total = int(n_vns * 1.3)  # some VNs will fall below min_vote
    quality = rng.normal(7, 1.1, n_total)
    popularity = rng.pareto(1.2, n_total) + 0.05
    popularity /= popularity.sum()
    olangs = rng.choice(["ja", "ja", "ja", "en", "zh-Hans", "ko"], n_total)

    # --- votes ---------------------------------------------------------------
    ulist, counts = [], np.zeros(n_total, int)
    for u in range(1, n_users + 1):
        k = min(int(rng.lognormal(2.3, 1.0)) + 1, n_total)
        items = np.sort(rng.choice(n_total, size=k, replace=False, p=popularity))
        bias, spread = rng.normal(0, 0.8), rng.uniform(0.6, 1.4)
        for v in items:
            if rng.random() < 0.2:  # on the list but not voted
                vote, vdate = "\\N", "\\N"
            else:
                raw = 7 + (quality[v] - 7) * spread + bias + rng.normal(0, 0.9)
                # Mostly whole points (70), sometimes VNDB's decimal votes (75).
                step = 10 if rng.random() < 0.8 else 1
                vote = str(int(np.clip(round(raw * 10 / step) * step, 10, 100)))
                vdate = f"{rng.integers(2008, 2027)}-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"
                counts[v] += 1
            labels = "{" + ",".join(sorted({str(rng.integers(1, 6)), "7"} if vote != "\\N" else {str(rng.integers(1, 6))})) + "}"
            notes = "\\N" if rng.random() < 0.8 else "great\\tstory\\nwould read again"
            ulist.append([f"u{u}", f"v{v + 1}", "2020-01-01", "2020-01-02", vdate, "\\N", "\\N", vote, notes, labels])
    _write(db, "ulist_vns", ["uid", "vid", "added", "lastmod", "vote_date", "started", "finished", "vote", "notes", "labels"], ulist)

    # --- catalogue -----------------------------------------------------------
    vn_rows, title_rows, rel_rows, rv_rows, rp_rows, img_rows = [], [], [], [], [], []
    n_prod = max(n_total // 4, 5)
    rid = 0
    for v in range(n_total):
        vid = f"v{v + 1}"
        name = " ".join(rng.choice(SYLLABLES, rng.integers(2, 4)))
        kana = "".join(rng.choice(KANA, rng.integers(2, 4)))
        han = "".join(rng.choice(HAN, rng.integers(2, 4)))
        alias = f"{han}\\n{name.upper()}" if rng.random() < 0.5 else "\\N"
        rating = int(round(np.clip(quality[v], 1, 10) * 100)) if counts[v] else "\\N"
        img = f"cv{v + 1000}" if rng.random() < 0.9 else "\\N"
        vn_rows.append([vid, img, "\\N", olangs[v], "\\N", str(counts[v]), str(rating), str(rating), str(rng.integers(0, 6)), "0", alias, "\\N", "A story.\\nWith lines."])
        if img != "\\N":
            img_rows.append([img, "256", "400", "10", str(int(rng.choice([0, 0, 0, 40, 120, 190]))), "\\N", "0", "\\N", "\\N", "\\N"])
        if olangs[v] == "ja":
            title_rows.append([vid, "ja", "t", kana, name.title()])
        else:
            title_rows.append([vid, olangs[v], "t", name.title() if olangs[v] == "en" else han, name.title()])
        if rng.random() < 0.7 and olangs[v] != "en":
            title_rows.append([vid, "en", "t", name.title() + " Tale", "\\N"])
        if rng.random() < 0.3 and olangs[v] != "zh-Hans":
            title_rows.append([vid, "zh-Hans", "f", han + "物语", "\\N"])
        if v > 0 and rng.random() < 0.1:
            rel_rows.append([vid, f"v{v}", "preq", "t"])
            rel_rows.append([f"v{v}", vid, "seq", "t"])
        dev = f"p{rng.integers(1, n_prod + 1)}"
        for k in range(rng.integers(1, 4)):
            rid += 1
            rtype = "trial" if k == 0 and rng.random() < 0.3 else "complete"
            rv_rows.append([f"r{rid}", vid, rtype])
            rp_rows.append([f"r{rid}", dev, "t", "t"])
            if rng.random() < 0.3:
                rp_rows.append([f"r{rid}", f"p{rng.integers(1, n_prod + 1)}", "f", "t"])
    rel_dates = [[f"r{i}", "\\N", "ja", str(rng.choice([int(f"{rng.integers(1995, 2026)}{rng.integers(1, 13):02d}{rng.integers(1, 29):02d}"), 99999999, 20200099]))] for i in range(1, rid + 1)]

    _write(db, "vn", ["id", "image", "c_image", "olang", "l_wikidata", "c_votecount", "c_rating", "c_average", "length", "devstatus", "alias", "l_renai", "description"], vn_rows)
    _write(db, "vn_titles", ["id", "lang", "official", "title", "latin"], title_rows)
    _write(db, "vn_relations", ["id", "vid", "relation", "official"], rel_rows)
    _write(db, "releases", ["id", "gtin", "olang", "released"], rel_dates)
    _write(db, "releases_vn", ["id", "vid", "rtype"], rv_rows)
    _write(db, "releases_producers", ["id", "pid", "developer", "publisher"], rp_rows)
    _write(db, "images", ["id", "width", "height", "c_votecount", "c_sexual_avg", "c_sexual_stddev", "c_violence_avg", "c_violence_stddev", "c_weight", "c_uids"], img_rows)
    _write(db, "producers", ["id", "type", "lang", "name", "latin", "alias", "description"], [
        [f"p{p}", "co", "ja", f"Studio {''.join(rng.choice(KANA, 2))}", f"Studio {p}", "\\N", "\\N"] for p in range(1, n_prod + 1)
    ])
    _write(db, "users", ["id", "username"], [[f"u{u}", f"user{u}"] for u in range(1, n_users + 1)])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--vns", type=int, default=300)
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_fake_dump(a.out, a.vns, a.users, a.seed)
    print(f"fake dump written to {a.out}")
