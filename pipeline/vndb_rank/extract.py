"""Extract the VN catalogue and the vote matrix from the dump."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .dump import Dump, strip_id, unescape

log = logging.getLogger(__name__)

_SEARCH_DROP = re.compile("[^0-9a-z\u3040-\u30fa\u30fc-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_HAN = re.compile("[\u3400-\u9fff]")
_KANA = re.compile("[\u3040-\u30ff]")
_CJK = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def normalize_search(*parts: str | None) -> str:
    """Lowercase and keep only alphanumerics, CJK and kana; joined with spaces."""
    seen: list[str] = []
    for p in parts:
        if not isinstance(p, str) or not p:
            continue
        for piece in p.split("\n"):
            key = _SEARCH_DROP.sub("", piece.lower())
            if key and key not in seen:
                seen.append(key)
    return " ".join(seen)


def to_ten_scale(series: pd.Series) -> pd.Series:
    """VNDB stores ratings as integers on a 10-100 vote scale, possibly multiplied
    by 10 or 100 for extra precision. Normalize to a 1-10 float."""
    values = pd.to_numeric(series, errors="coerce")
    m = values.max()
    if pd.isna(m):
        return values
    div = 10 if m <= 100 else 100 if m <= 1000 else 1000
    return values / div


@dataclass
class Catalogue:
    vn: pd.DataFrame  # one row per ranked VN, ordered by id; row position == matrix index
    producers: pd.DataFrame
    relations: dict[int, list[list]] = field(default_factory=dict)

    @property
    def ids(self) -> np.ndarray:
        return self.vn["id"].to_numpy()


def load_catalogue(dump: Dump, cfg: Config) -> Catalogue:
    cols = dump.optional_columns(
        "vn", ["id", "image", "olang", "c_votecount", "c_rating", "c_average", "length", "alias"]
    )
    vn = dump.read("vn", cols)
    vn["votes"] = pd.to_numeric(vn["c_votecount"], errors="coerce").fillna(0).astype(int)
    vn = vn[vn["votes"] >= cfg.min_vote].copy()
    vn["id"] = strip_id(vn["id"]).astype(int)
    vn["rating"] = to_ten_scale(vn["c_rating"]) if "c_rating" in vn else np.nan
    vn["average"] = to_ten_scale(vn["c_average"]) if "c_average" in vn else np.nan
    vn["length"] = pd.to_numeric(vn.get("length"), errors="coerce") if "length" in vn else np.nan
    vn["alias"] = vn["alias"].map(unescape) if "alias" in vn else None
    vn = vn.sort_values("id").reset_index(drop=True)
    ids = set(vn["id"])
    log.info("catalogue: %d VNs with >= %d votes", len(vn), cfg.min_vote)

    _attach_titles(dump, vn)
    _attach_images(dump, vn)
    producers = _attach_releases(dump, vn, ids)
    relations = _load_relations(dump, ids)

    vn["search"] = [
        normalize_search(*row)
        for row in zip(vn["title"], vn["latin"], vn["title_ja"], vn["title_zh"], vn["title_en"], vn["all_titles"], vn["alias"])
    ]
    keep = [
        "id", "title", "latin", "title_ja", "title_zh", "title_en", "olang", "released",
        "dev_id", "image", "image_sexual", "image_violence", "length", "votes", "rating", "average", "search",
    ]
    return Catalogue(vn=vn[keep].copy(), producers=producers, relations=relations)


def _attach_titles(dump: Dump, vn: pd.DataFrame) -> None:
    t = dump.read("vn_titles", dump.optional_columns("vn_titles", ["id", "lang", "official", "title", "latin"]))
    t["id"] = strip_id(t["id"]).astype(int)
    t = t[t["id"].isin(vn["id"])]
    t["title"] = t["title"].map(unescape)
    t["latin"] = t["latin"].map(unescape)
    if "official" in t:  # official titles win over unofficial translations of the same language
        t = t.sort_values("official", ascending=False, kind="stable")
    by_lang = {lang: g.drop_duplicates("id").set_index("id") for lang, g in t.groupby("lang")}

    def pick(lang: str, col: str = "title") -> pd.Series:
        g = by_lang.get(lang)
        return vn["id"].map(g[col]) if g is not None else pd.Series(None, index=vn.index, dtype=object)

    orig = t.merge(vn[["id", "olang"]], left_on=["id", "lang"], right_on=["id", "olang"]).drop_duplicates("id").set_index("id")
    vn["title"] = vn["id"].map(orig["title"])
    vn["latin"] = vn["id"].map(orig["latin"])
    vn["title_ja"] = pick("ja")
    vn["title_en"] = pick("en")
    vn["title_zh"] = pick("zh-Hans").fillna(pick("zh-Hant")).fillna(pick("zh"))
    vn["all_titles"] = vn["id"].map(t.groupby("id")["title"].agg(lambda s: "\n".join(x for x in s if x)))

    # No Chinese title on VNDB: try an official Chinese release, then aliases.
    missing = vn["title_zh"].isna()
    rel_title, rel_text = _chinese_release_titles(dump, set(vn.loc[missing, "id"]))
    vn.loc[missing, "title_zh"] = [rel_title.get(int(i)) for i in vn.loc[missing, "id"]]
    missing = vn["title_zh"].isna()
    vn.loc[missing, "title_zh"] = [
        chinese_alias(a, o, rel_text.get(i, ""))
        for i, a, o in zip(vn.loc[missing, "id"], vn.loc[missing, "alias"], vn.loc[missing, "title"])
    ]

    # Every VN must have a display title.
    fallback = vn["title_en"].fillna(vn["title_ja"]).fillna(vn["id"].map(lambda i: f"v{i}"))
    vn["title"] = vn["title"].fillna(fallback)
    vn["latin"] = vn["latin"].where(vn["latin"] != vn["title"])


ZH_LANGS = ["zh-Hans", "zh-Hant", "zh"]


def chinese_alias(alias: str | None, original: str | None, release_text: str = "") -> str | None:
    """A Chinese name from VNDB's free-form aliases, or None if it isn't trustworthy.

    Aliases with Han characters and no kana count as Chinese. If the original
    title is in kana / hangul, Chinese readers can't read it, so the first such
    alias is better than nothing. If it is in Latin script (Rewrite, Fate/stay
    night) the original is readable and often what Chinese players use, so an
    alias is only used when it is the only Chinese one or a Chinese release title
    contains it (Rewrite's 罚抄 / 改写 are a joke and a literal translation)."""
    if not isinstance(alias, str):
        return None
    cands = [a.strip() for a in alias.split("\n") if _HAN.search(a) and not _KANA.search(a)]
    cands = list(dict.fromkeys(c for c in cands if len(c) >= 2))
    if not cands:
        return None
    if isinstance(original, str) and _CJK.search(original):
        return cands[0]
    if len(cands) == 1:
        return cands[0]
    confirmed = [c for c in cands if c in release_text]
    return confirmed[0] if confirmed else None


# Edition / format suffixes on Chinese release titles ("下載版", "(中文版)", ...).
_RELEASE_SUFFIX = re.compile(
    r"\s*([（(][^（()）]*[)）]"                      # (中文版)
    r"|[-－~～]?\s*(下[载載]|通常|中文|限定|初回限定|豪[华華]|完全|普通|特典|中文特典|DL|体验|體驗|試玩|试玩)版"
    r"|[vV]?\d+(\.\d+)+"                              # 0.62, v1.2
    r"|[-－]\s*第.{1,8}[章集话話卷部]+.*)\s*$"           # - 第一章第四集
)


def clean_release_title(title: str) -> str | None:
    """A usable Chinese name from a release title: Han characters, no Latin
    letters (mixed product names like "Fate/Extella 下載版"), edition suffixes off."""
    t = title.strip()
    for _ in range(3):
        t = _RELEASE_SUFFIX.sub("", t).strip()
    if not _HAN.search(t) or re.search("[A-Za-z]", t) or len(t) < 2:
        return None
    return t


def _chinese_release_titles(dump: Dump, ids: set[int]) -> tuple[dict[int, str], dict[int, str]]:
    """Chinese titles from releases of a single VN.

    Returns (title from an official, non-patch, human-translated Chinese release,
    all Chinese release titles joined) per VN. Of several official titles the one
    most others start with wins, so "Rewrite" beats "Rewrite 体验版"."""
    if not ids or not dump.has("releases_titles"):
        return {}, {}
    rv = dump.read("releases_vn", ["id", "vid"])
    rv["vid"] = strip_id(rv["vid"]).astype(int)
    rv = rv[rv["vid"].isin(ids)]
    per_release = rv.groupby("id")["vid"].nunique()
    rv = rv[rv["id"].isin(per_release[per_release == 1].index)]  # bundles have no single title
    rt = dump.read("releases_titles", dump.optional_columns("releases_titles", ["id", "lang", "mtl", "title"]))
    rt = rt[rt["lang"].isin(ZH_LANGS) & rt["title"].notna() & rt["id"].isin(set(rv["id"]))].copy()
    rt["title"] = rt["title"].map(unescape).str.strip()
    rt = rt.merge(rv, on="id")
    text = rt.groupby("vid")["title"].agg(" ".join).to_dict()
    rel = dump.read("releases", dump.optional_columns("releases", ["id", "official", "patch"]))
    ok = pd.Series(True, index=rel.index)
    if "official" in rel:
        ok &= rel["official"] == "t"
    if "patch" in rel:
        ok &= rel["patch"] != "t"
    good = rt[rt["id"].isin(set(rel.loc[ok, "id"])) & (rt["mtl"] != "t" if "mtl" in rt else True)]
    titles: dict[int, str] = {}
    for vid, g in good.groupby("vid"):
        ts = [c for x in g["title"] if (c := clean_release_title(x))]
        if ts:
            titles[int(vid)] = max(dict.fromkeys(ts), key=lambda c: (sum(x.startswith(c) for x in ts), -len(c)))
    return titles, text


def _attach_images(dump: Dump, vn: pd.DataFrame) -> None:
    vn["image_sexual"] = np.nan
    vn["image_violence"] = np.nan
    if "image" not in vn:
        vn["image"] = None
        return
    vn["image"] = strip_id(vn["image"])
    if dump.has("images") and "c_sexual_avg" in dump.header("images"):
        img = dump.read("images", dump.optional_columns("images", ["id", "c_sexual_avg", "c_violence_avg"]))
        img = img[img["id"].str.startswith("cv", na=False)]
        img["id"] = strip_id(img["id"])
        sexual = pd.to_numeric(img["c_sexual_avg"], errors="coerce")
        # 0 (safe) .. 2 (explicit), possibly stored x100.
        if sexual.max() > 2:
            sexual = sexual / 100
        vn["image_sexual"] = vn["image"].map(pd.Series(sexual.to_numpy(), index=img["id"].to_numpy()))
        if "c_violence_avg" in img:
            violence = pd.to_numeric(img["c_violence_avg"], errors="coerce")
            if violence.max() > 2:
                violence = violence / 100
            vn["image_violence"] = vn["image"].map(pd.Series(violence.to_numpy(), index=img["id"].to_numpy()))


def _attach_releases(dump: Dump, vn: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    """Earliest release date and the main developer of each VN."""
    rv = dump.read("releases_vn", dump.optional_columns("releases_vn", ["id", "vid", "rtype"]))
    rv["vid"] = strip_id(rv["vid"]).astype(int)
    rv = rv[rv["vid"].isin(ids)]
    rel = dump.read("releases", ["id", "released"])
    rel["released"] = pd.to_numeric(rel["released"], errors="coerce")
    rv = rv.merge(rel, on="id", how="left")
    valid = rv["released"].between(19000000, 99990000, inclusive="left")
    rv["date_key"] = rv["released"].where(valid, 99999999)
    rv["trial"] = (rv["rtype"] == "trial") if "rtype" in rv else False

    first = rv[valid & ~rv["trial"]].groupby("vid")["released"].min()
    any_first = rv[valid].groupby("vid")["released"].min()
    vn["released"] = vn["id"].map(first.combine_first(any_first)).astype("Int64")

    rp = dump.read("releases_producers", ["id", "pid", "developer"])
    rp = rp[rp["developer"] == "t"]
    cand = rv.merge(rp[["id", "pid"]], on="id").sort_values(["vid", "trial", "date_key", "id", "pid"])
    dev = cand.drop_duplicates("vid").set_index("vid")["pid"]
    vn["dev_id"] = strip_id(vn["id"].map(dev))

    pcols = dump.optional_columns("producers", ["id", "name", "latin"])
    prod = dump.read("producers", pcols)
    prod["id"] = strip_id(prod["id"])
    prod = prod[prod["id"].isin(set(vn["dev_id"].dropna()))].copy()
    prod["name"] = prod["name"].map(unescape)
    prod["latin"] = prod["latin"].map(unescape) if "latin" in prod else None
    prod["search"] = [normalize_search(n, l) for n, l in zip(prod["name"], prod["latin"])]
    return prod.sort_values("id").reset_index(drop=True)


def _load_relations(dump: Dump, ids: set[int]) -> dict[int, list[list]]:
    if not dump.has("vn_relations"):
        return {}
    r = dump.read("vn_relations", ["id", "vid", "relation", "official"])
    r["id"] = strip_id(r["id"]).astype(int)
    r["vid"] = strip_id(r["vid"]).astype(int)
    r = r[r["id"].isin(ids) & r["vid"].isin(ids) & (r["official"] == "t")]
    out: dict[int, list[list]] = {}
    for a, b, rel in zip(r["id"], r["vid"], r["relation"]):
        out.setdefault(int(a), []).append([int(b), rel])
    return out


@dataclass
class Votes:
    uid: np.ndarray   # int32, sorted by (uid, vidx)
    vidx: np.ndarray  # int32 index into Catalogue.vn
    vote: np.ndarray  # int16 on the 10-100 scale
    year: np.ndarray  # int16 year of the vote (0 = unknown)
    labels: np.ndarray | None = None  # uint8 bit mask of list labels 1..6 (bit k-1 = label k)

    def __len__(self) -> int:
        return len(self.uid)


LABELS = 6  # 1 playing, 2 finished, 3 stalled, 4 dropped, 5 wishlist, 6 blacklist


def load_users(dump: Dump) -> tuple[set[int], dict[int, str]]:
    """Ignored users (VNDB discards their votes) and usernames."""
    if not dump.has("users"):
        return set(), {}
    cols = dump.optional_columns("users", ["id", "ign_votes", "username"])
    u = dump.read("users", cols)
    u["id"] = strip_id(u["id"]).astype(int)
    ignored = set(u.loc[u["ign_votes"] == "t", "id"]) if "ign_votes" in u else set()
    names = dict(zip(u["id"], u["username"].map(unescape))) if "username" in u else {}
    return ignored, names


@dataclass
class Extras:
    """Side products of the ulist_vns pass."""
    user_total: pd.Series   # uid -> votes on any VN
    user_year: pd.Series    # uid -> votes cast in the dump's year
    notes: pd.DataFrame     # uid, vidx, date (YYYYMMDD), vote (or 0), labels (bit mask), text; ranked VNs only


NOTE_MIN_CHARS = 20
NOTE_MAX_CHARS = 3000


def load_votes(
    dump: Dump, catalogue: Catalogue, ignored: set[int] | None = None, year: int | None = None, with_notes: bool = True
) -> tuple[Votes, dict, np.ndarray, Extras]:
    """Single streaming pass over ulist_vns: collect votes on ranked VNs, per-VN
    list-label counts, per-user totals, notes and global vote statistics.
    Votes of users VNDB flags as ignored are dropped everywhere."""
    ignored = ignored or set()
    year_of_dump = year
    total_parts, year_count_parts, note_parts = [], [], []
    n_vn = len(catalogue.vn)
    id2idx = pd.Series(np.arange(n_vn, dtype=np.int32), index=catalogue.ids)
    uid_parts, vidx_parts, vote_parts, year_parts, label_parts = [], [], [], [], []
    vn_labels = np.zeros((n_vn, LABELS + 1), dtype=np.int64)

    hist = np.zeros(101, dtype=np.int64)  # exact votes 0..100
    year_n: dict[int, int] = {}
    year_sum: dict[int, float] = {}
    year_sq: dict[int, float] = {}
    labels = np.zeros(LABELS + 1, dtype=np.int64)
    rows = 0

    wanted = ["uid", "vid", "vote", "vote_date", "lastmod", "labels"] + (["notes"] if with_notes else [])
    cols = dump.optional_columns("ulist_vns", wanted)
    for chunk in dump.read_chunks("ulist_vns", cols):
        chunk["uid"] = strip_id(chunk["uid"])
        if ignored:
            chunk = chunk[~chunk["uid"].isin(ignored)]
        rows += len(chunk)
        chunk_vidx = strip_id(chunk["vid"]).map(id2idx)
        lmask = pd.Series(np.zeros(len(chunk), np.uint8), index=chunk.index)
        if "labels" in chunk:
            lab = chunk["labels"].fillna("")
            ranked = chunk_vidx.notna().to_numpy()
            ranked_idx = chunk_vidx.to_numpy()[ranked].astype(np.int64)
            for k in range(1, LABELS + 1):
                has = lab.str.contains(rf"[{{,]{k}[,}}]", regex=True).to_numpy()
                labels[k] += has.sum()
                np.add.at(vn_labels[:, k], ranked_idx[has[ranked]], 1)
                lmask |= (has.astype(np.uint8) << (k - 1)).astype(np.uint8)
        if "notes" in chunk:
            has_note = chunk["notes"].notna() & chunk_vidx.notna()
            if has_note.any():
                nt = chunk.loc[has_note, ["uid", "notes"]].copy()
                nt["text"] = nt["notes"].map(unescape).str.strip()
                nt = nt[nt["text"].str.len() >= NOTE_MIN_CHARS]
                if len(nt):
                    date_col = chunk.loc[nt.index, "lastmod"] if "lastmod" in chunk else pd.Series("", index=nt.index)
                    note_parts.append(pd.DataFrame({
                        "uid": nt["uid"].astype(np.int64).to_numpy(),
                        "vidx": chunk_vidx.loc[nt.index].astype(np.int64).to_numpy(),
                        "date": pd.to_numeric(date_col.str.replace("-", "").str[:8], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
                        "vote": pd.to_numeric(chunk.loc[nt.index, "vote"], errors="coerce").fillna(0).astype(np.int64).to_numpy(),
                        "labels": lmask.loc[nt.index].to_numpy(),
                        "text": nt["text"].str.slice(0, NOTE_MAX_CHARS).to_numpy(),
                    }))

        vote = pd.to_numeric(chunk["vote"], errors="coerce")
        ok = vote.notna()
        chunk, vote, chunk_vidx, lmask = chunk[ok], vote[ok].astype(np.int16), chunk_vidx[ok], lmask[ok]
        hist += np.bincount(vote.clip(0, 100), minlength=101)
        total_parts.append(chunk["uid"].value_counts())

        if "vote_date" in chunk:
            year = pd.to_numeric(chunk["vote_date"].str[:4], errors="coerce")
            if year_of_dump is not None:
                year_count_parts.append(chunk.loc[(year == year_of_dump).to_numpy(), "uid"].value_counts())
            g = (vote / 10).groupby(year).agg(["count", "sum"])
            g["sq"] = ((vote / 10) ** 2).groupby(year).sum()
            for y, c, s_, q in zip(g.index, g["count"], g["sum"], g["sq"]):
                if pd.isna(y):
                    continue
                y = int(y)
                year_n[y] = year_n.get(y, 0) + int(c)
                year_sum[y] = year_sum.get(y, 0.0) + float(s_)
                year_sq[y] = year_sq.get(y, 0.0) + float(q)
        else:
            year = pd.Series(np.nan, index=chunk.index)

        keep = chunk_vidx.notna().to_numpy()
        uid_parts.append(chunk["uid"].to_numpy()[keep].astype(np.int32))
        vidx_parts.append(chunk_vidx.to_numpy()[keep].astype(np.int32))
        vote_parts.append(vote.to_numpy()[keep])
        year_parts.append(year.fillna(0).to_numpy()[keep].astype(np.int16))
        label_parts.append(lmask.to_numpy()[keep].astype(np.uint8))

    cat = lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt)
    uid, vidx, vote, year = cat(uid_parts, np.int32), cat(vidx_parts, np.int32), cat(vote_parts, np.int16), cat(year_parts, np.int16)
    lab = cat(label_parts, np.uint8)
    order = np.lexsort((vidx, uid))
    votes = Votes(uid=uid[order], vidx=vidx[order], vote=vote[order], year=year[order], labels=lab[order])
    log.info("votes: %d on ranked VNs from %d users (%d ignored users dropped)", len(uid), len(np.unique(uid)), len(ignored))

    total = int(hist.sum())
    scale = np.arange(101) / 10
    mean = float((hist * scale).sum() / total) if total else 0.0
    std = float(np.sqrt((hist * (scale - mean) ** 2).sum() / total)) if total else 0.0
    years = []
    for y in sorted(year_n):
        n = year_n[y]
        m = year_sum[y] / n
        years.append({"year": y, "count": n, "mean": round(m, 3), "std": round(float(np.sqrt(max(year_sq[y] / n - m * m, 0))), 3)})

    stats = {
        "ulist_rows": rows,
        "votes": {"count": total, "mean": round(mean, 3), "std": round(std, 3), "histogram": vote_buckets(hist)},
        "labels": {str(k): int(labels[k]) for k in range(1, LABELS + 1)},
        "years": years,
        "ranked_votes": int(len(uid)),
        "ranked_users": int(len(np.unique(uid))),
    }
    merge = lambda parts: pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=np.int64)
    notes = pd.concat(note_parts, ignore_index=True) if note_parts else pd.DataFrame(columns=["uid", "vidx", "date", "vote", "labels", "text"])
    extras = Extras(user_total=merge(total_parts), user_year=merge(year_count_parts), notes=notes)
    log.info("notes: %d (>= %d chars) on ranked VNs", len(notes), NOTE_MIN_CHARS)
    return votes, stats, vn_labels, extras


def vote_buckets(hist101: np.ndarray) -> list[int]:
    """Exact 0..100 vote counts -> 10 buckets for 1..10 (7.5 counts towards 7)."""
    buckets = [0] * 10
    for v, c in enumerate(hist101):
        if c and v >= 10:
            buckets[min(v // 10, 10) - 1] += int(c)
    return buckets
