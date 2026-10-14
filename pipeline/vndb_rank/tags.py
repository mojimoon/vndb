"""VN tags from the dump (``tags`` + ``tags_vn``) and their TF-IDF weights.

A VN's tag rating is computed the way VNDB does it: the mean of the users'
votes (-3..3), ignoring votes a moderator marked as ignored, and the tag only
counts when that mean is above 0. The spoiler level is the mean of the spoiler
votes, bucketed like VNDB (> 1.3 major, > 0.4 minor, else none; the tag's
default when nobody voted on it); a tag marked as a lie counts as a major
spoiler. Only searchable (live) tags are kept.

Scores (computed in the browser from what is stored here):

* Title page: ``rating x idf`` with ``idf = ln(N / df)``, N = VNs with any
  tag, df = VNs carrying the tag (all of VNDB).
* User page: TF is the tag's rating summed over the user's voted titles and
  divided by their number of votes; IDF is weighted by votes instead of
  titles: ``ln(V / V_t)``, V = votes on ranked titles, V_t = votes on ranked
  titles carrying the tag, so tags of popular titles (that most lists contain)
  weigh less.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dump import Dump, strip_id, unescape

log = logging.getLogger(__name__)

SPOIL_MINOR = 0.4
SPOIL_MAJOR = 1.3
CATS = {"cont": 0, "ero": 1, "tech": 2}
RATING_SCALE = 50  # binary doc: rating (0..3] stored as round(rating * 50)


@dataclass
class Tags:
    meta: pd.DataFrame  # id, name, cat (0 content / 1 sexual / 2 technical), df, idf, idf_votes
    vn: pd.DataFrame    # vid, tag, rating (>0), spoiler (0/1/2), voters; sorted by vid, rating desc
    n_vns: int          # VNs with at least one tag


def load_tags(dump: Dump) -> Tags | None:
    if not (dump.has("tags") and dump.has("tags_vn")):
        log.info("tags: not in dump, skipped")
        return None
    t = dump.read("tags", ["id", "cat", "defaultspoil", "searchable", "name"])
    t = t[t["searchable"] == "t"].copy()
    t["id"] = strip_id(t["id"]).astype(int)
    t["defaultspoil"] = pd.to_numeric(t["defaultspoil"]).fillna(0).astype(int)

    cols = dump.optional_columns("tags_vn", ["tag", "vid", "vote", "spoiler", "ignore", "lie"])
    parts = []
    for c in dump.read_chunks("tags_vn", cols):
        if "ignore" in c:
            c = c[c["ignore"] != "t"]
        df = pd.DataFrame({
            "tag": strip_id(c["tag"]).astype(int),
            "vid": strip_id(c["vid"]).astype(int),
            "vote": pd.to_numeric(c["vote"]).astype(np.int8),
            "spoiler": pd.to_numeric(c["spoiler"]) if "spoiler" in c else np.nan,
            "lie": (c["lie"] == "t") if "lie" in c else False,
        })
        parts.append(df)
    tv = pd.concat(parts, ignore_index=True)
    tv = tv[tv["tag"].isin(t["id"])]
    g = tv.groupby(["vid", "tag"], sort=False)
    vn = g.agg(rating=("vote", "mean"), voters=("vote", "size"), spoil=("spoiler", "mean"), lie=("lie", "any")).reset_index()
    vn = vn[vn["rating"] > 0]
    default = t.set_index("id")["defaultspoil"]
    spoil = vn["spoil"].to_numpy()
    level = np.where(spoil > SPOIL_MAJOR, 2, np.where(spoil > SPOIL_MINOR, 1, 0))
    level = np.where(np.isnan(spoil), default.reindex(vn["tag"]).fillna(0).to_numpy(), level)
    vn["spoiler"] = np.where(vn["lie"], 2, level).astype(np.int8)
    vn = vn[["vid", "tag", "rating", "spoiler", "voters"]].sort_values(["vid", "rating", "tag"], ascending=[True, False, True], ignore_index=True)

    n_vns = int(vn["vid"].nunique())
    df = vn.groupby("tag").size()
    meta = t[t["id"].isin(df.index)][["id", "name", "cat"]].copy()
    meta["name"] = meta["name"].map(unescape)
    meta["cat"] = meta["cat"].map(CATS).fillna(0).astype(int)
    meta["df"] = df.reindex(meta["id"]).to_numpy()
    meta["idf"] = np.log(n_vns / meta["df"])
    meta = meta.sort_values("id", ignore_index=True)
    log.info("tags: %d tags on %d VNs (%d VN-tag pairs)", len(meta), n_vns, len(vn))
    return Tags(meta=meta, vn=vn, n_vns=n_vns)


def vote_idf(tags: Tags, ids: np.ndarray, votes_per_vn: np.ndarray) -> pd.Series:
    """User-page IDF: ln(V / V_t) over ranked VNs, weighted by their vote counts."""
    w = pd.Series(votes_per_vn.astype(np.float64), index=ids)
    sub = tags.vn[tags.vn["vid"].isin(w.index)]
    vt = sub.assign(w=w.reindex(sub["vid"]).to_numpy()).groupby("tag")["w"].sum()
    total = float(w.sum()) or 1.0
    return np.log(total / vt.reindex(tags.meta["id"]).to_numpy())


def vn_tag_json(tags: Tags | None, ids: np.ndarray) -> list[list[list]]:
    """Per ranked VN: [[tag, rating (2 dp), spoiler, voters], ...], best rated first."""
    out: dict[int, list[list]] = {}
    if tags is not None:
        sub = tags.vn[tags.vn["vid"].isin(ids)]
        for vid, tag, r, s, n in zip(sub["vid"], sub["tag"], sub["rating"], sub["spoiler"], sub["voters"]):
            out.setdefault(int(vid), []).append([int(tag), round(float(r), 2), int(s), int(n)])
    return [out.get(int(v), []) for v in ids]


def tags_doc(tags: Tags, idf_votes: np.ndarray) -> dict:
    """Tag dictionary for the frontend: [[id, name, cat, df, idf, idf_votes], ...]."""
    m = tags.meta
    rows = [
        [int(i), n, int(c), int(d), round(float(f), 4), None if not np.isfinite(v) else round(float(v), 4)]
        for i, n, c, d, f, v in zip(m["id"], m["name"], m["cat"], m["df"], m["idf"], idf_votes)
    ]
    return {"n_vns": tags.n_vns, "tags": rows}


def vn_tags_blob(tags: Tags, ids: np.ndarray) -> str:
    """Tags of every ranked VN for user-page analysis, base64 of:
    u32 n, u32 offsets[n + 1] (entry index per VN idx), then 4-byte entries
    (tag u16, round(rating * 50) u8, spoiler u8), little endian."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    sub = tags.vn[tags.vn["vid"].isin(ids)]
    idx = pos.reindex(sub["vid"]).to_numpy()
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    counts = np.bincount(idx, minlength=len(ids))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("<u4")
    ent = np.zeros(len(idx), dtype=[("tag", "<u2"), ("r", "u1"), ("s", "u1")])
    ent["tag"] = sub["tag"].to_numpy()[order]
    ent["r"] = np.clip(np.rint(sub["rating"].to_numpy()[order] * RATING_SCALE), 1, 255)
    ent["s"] = sub["spoiler"].to_numpy()[order]
    raw = np.array([len(ids)], dtype="<u4").tobytes() + offsets.tobytes() + ent.tobytes()
    return base64.b64encode(raw).decode("ascii")
