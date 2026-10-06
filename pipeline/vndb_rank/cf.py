"""Collaborative filtering on mean-centered votes.

* Similar users: cosine similarity of mean-centered vote vectors, shrunk by
  common / (common + shrink) so two users with three VNs in common don't look
  like soulmates.
* Recommendations (user-based CF): for each user, the k most similar users
  vote on VNs the user hasn't voted on; the prediction is the user's own mean
  plus the similarity-weighted average of the neighbors' deviations from
  their means. Only VNs rated by at least ``min_support`` neighbors qualify.
* Similar VNs ("people who liked this also liked"): the same centered cosine
  between VN columns.

Everything is computed in blocks so memory stays at a few GB for ~50k users
and ~8k VNs.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .extract import Votes

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CFConfig:
    min_user_votes: int = 5        # users with fewer ranked votes get no user page
    min_neighbor_votes: int = 15   # users eligible as someone's neighbor
    neighbors: int = 30            # k for predictions
    similar_users: int = 10        # shown on the user page
    recommendations: int = 20
    min_support: int = 3           # neighbors who must have voted on a recommended VN
    min_common: int = 3            # common VNs for two users to be compared
    user_shrink: float = 10.0
    item_shrink: float = 20.0
    item_min_common: int = 10
    similar_items: int = 10
    block: int = 2000


def encode_votes(idx: np.ndarray, vote: np.ndarray) -> str:
    """Pack (matrix index uint16, vote uint8) pairs as base64 (3 bytes per vote)."""
    rec = np.empty(len(idx), dtype=[("i", "<u2"), ("v", "u1")])
    rec["i"] = idx
    rec["v"] = vote
    return base64.b64encode(rec.tobytes()).decode("ascii")


def decode_votes(s: str) -> tuple[np.ndarray, np.ndarray]:
    rec = np.frombuffer(base64.b64decode(s), dtype=[("i", "<u2"), ("v", "u1")])
    return rec["i"].astype(np.int64), rec["v"].astype(np.int64)


def _top_k(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Column indices and values of the k largest entries per row, descending."""
    k = min(k, mat.shape[1])
    if k == 0:
        return np.zeros((mat.shape[0], 0), int), np.zeros((mat.shape[0], 0))
    part = np.argpartition(-mat, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(mat, part, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


@dataclass
class UserModel:
    uids: np.ndarray        # users with >= min_user_votes ranked votes
    records: dict[int, dict]


def build_users(votes: Votes, n_items: int, names: dict[int, str], cfg: CFConfig) -> UserModel:
    users, uinv, counts = np.unique(votes.uid, return_inverse=True, return_counts=True)
    keep_user = counts >= cfg.min_user_votes
    row_of = np.full(len(users), -1)
    row_of[keep_user] = np.arange(keep_user.sum())
    mask = keep_user[uinv]
    rows = row_of[uinv[mask]]
    cols = votes.vidx[mask]
    v = votes.vote[mask].astype(np.float64) / 10
    U = int(keep_user.sum())
    uids = users[keep_user]
    log.info("cf: %d users with >= %d ranked votes", U, cfg.min_user_votes)
    if U == 0:
        return UserModel(uids=uids, records={})

    mean = np.bincount(rows, weights=v, minlength=U) / np.bincount(rows, minlength=U)
    c = v - mean[rows]
    R = sp.csr_matrix((c.astype(np.float32), (rows, cols)), shape=(U, n_items))
    B = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(U, n_items))
    norm = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
    norm[norm == 0] = np.inf  # users who voted everything the same have no direction

    ucount = np.asarray(B.sum(axis=1)).ravel()
    cand = np.flatnonzero(ucount >= cfg.min_neighbor_votes)
    Rc = R[cand].toarray()  # (C, N) dense
    Bc = B[cand].toarray()
    RcT, BcT = np.ascontiguousarray(Rc.T), np.ascontiguousarray(Bc.T)
    cand_norm = norm[cand]
    log.info("cf: %d neighbor candidates", len(cand))

    records: dict[int, dict] = {}
    for start in range(0, U, cfg.block):
        stop = min(start + cfg.block, U)
        Rb, Bb = R[start:stop], B[start:stop]
        dot = np.asarray(Rb @ RcT)          # (b, C)
        common = np.asarray(Bb @ BcT)
        sim = dot / np.outer(norm[start:stop], cand_norm)
        sim *= common / (common + cfg.user_shrink)
        sim[common < cfg.min_common] = 0
        # no self-similarity
        self_rows, self_cols = np.nonzero(cand[None, :] == np.arange(start, stop)[:, None])
        sim[self_rows, self_cols] = 0
        sim = np.nan_to_num(sim, nan=0.0)

        nb_idx, nb_sim = _top_k(sim, cfg.neighbors)
        nb_sim = np.maximum(nb_sim, 0)
        b = stop - start
        W = sp.csr_matrix((nb_sim.ravel(), (np.repeat(np.arange(b), nb_idx.shape[1]), nb_idx.ravel())), shape=(b, len(cand)))
        num = np.asarray(W @ Rc)
        den = np.asarray(W @ Bc)
        support = np.asarray((W > 0).astype(np.float32) @ Bc)
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = mean[start:stop, None] + num / den
        pred[(support < cfg.min_support) | (den <= 0)] = -np.inf
        pred[Bb.toarray() > 0] = -np.inf  # already voted
        rec_idx, rec_val = _top_k(pred, cfg.recommendations)

        for r in range(b):
            u = start + r
            uid = int(uids[u])
            lo, hi = R.indptr[u], R.indptr[u + 1]
            item_idx = R.indices[lo:hi]
            order = np.argsort(item_idx)
            raw = np.rint((R.data[lo:hi][order] + mean[u]) * 10).astype(np.int64)
            sims = [
                [int(uids[cand[j]]), names.get(int(uids[cand[j]]), ""), round(float(s), 4), int(common[r, j])]
                for j, s in zip(nb_idx[r, : cfg.similar_users], nb_sim[r, : cfg.similar_users]) if s > 0
            ]
            recs = [
                [int(i), round(float(p), 2), int(support[r, i])]
                for i, p in zip(rec_idx[r], rec_val[r]) if np.isfinite(p)
            ]
            records[uid] = {
                "name": names.get(uid, ""),
                "votes": encode_votes(item_idx[order], np.clip(raw, 10, 100)),
                "similar": sims,
                "recs": recs,
            }
        log.info("cf: users %d-%d done", start, stop)
    return UserModel(uids=uids, records=records)


def similar_items(votes: Votes, n_items: int, cfg: CFConfig) -> list[list[list]]:
    """Top similar VNs per VN index: [[other_index, similarity, common_voters], ...]."""
    users, uinv = np.unique(votes.uid, return_inverse=True)
    v = votes.vote.astype(np.float64) / 10
    mean = np.bincount(uinv, weights=v) / np.bincount(uinv)
    c = (v - mean[uinv]).astype(np.float32)
    R = sp.csc_matrix((c, (uinv, votes.vidx)), shape=(len(users), n_items))
    B = sp.csc_matrix((np.ones(len(c), np.float32), (uinv, votes.vidx)), shape=(len(users), n_items))
    norm = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
    norm[norm == 0] = np.inf
    RT, BT = R.T.tocsr(), B.T.tocsr()
    out: list[list[list]] = []
    for start in range(0, n_items, cfg.block):
        stop = min(start + cfg.block, n_items)
        dot = (RT[start:stop] @ R).toarray()
        common = (BT[start:stop] @ B).toarray()
        sim = dot / np.outer(norm[start:stop], norm)
        sim *= common / (common + cfg.item_shrink)
        sim[common < cfg.item_min_common] = 0
        sim[np.arange(stop - start), np.arange(start, stop)] = 0
        idx, val = _top_k(np.nan_to_num(sim), cfg.similar_items)
        for r in range(stop - start):
            out.append([[int(j), round(float(s), 4), int(common[r, j])] for j, s in zip(idx[r], val[r]) if s > 0])
    return out
