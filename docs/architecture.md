# Architecture

## Why the Supabase version did not fit

The previous design uploaded per-user votes (`ulist`, one row per user × VN,
millions of rows) to Supabase and planned to compute or serve rankings from
there. Raw votes are only needed to *build* things; the site only ever needs
the results. Shipping inputs instead of outputs is what blew through a small
instance's storage.

## Data flow

```
dl.vndb.org dump (daily, ~08:00 UTC)
  └─ GitHub Actions refresh.yml (12:07 UTC)
       ├─ extract 10 tables from the 190 MB .tar.zst
       ├─ fetch yesterday's rank history from D1 (`SELECT id, history FROM vn`)
       ├─ pipeline/ (python -m vndb_rank, ~10 min)
       │    extract.py    catalogue (VNs with >= 30 votes), users, one streaming pass over ulist_vns
       │    analysis.py   per-VN rating analysis
       │    pairs.py      N x N pair counts + mean votes / sample percentiles
       │    methods.py    8 PONet, 40 rankit, 8 Borda methods + VNDB reference
       │    neighbors.py  per-VN head-to-head opponents
       │    cf.py         similar users, recommendations, similar VNs
       │    storage.py    binary pair blocks, user shards, history
       │    export.py     snapshot.sql (schema + data + table swap)
       └─ wrangler d1 execute --remote --file snapshot.sql
Cloudflare D1  <──  Worker (Hono, web/worker/index.ts)  <──  React SPA (static assets)
```

Votes of users VNDB flags `ign_votes` are dropped everywhere. On the
2026-10-06 dump that removed ~2.2M of 7.1M comparable pairs.

## Schema (defined in `pipeline/vndb_rank/export.py`)

Numbers from the 2026-10-06 dump:

| table | rows | what one row holds |
| --- | --- | --- |
| `vn` | 7,945 | one VN: display fields + JSON columns `ranks` (57 methods), `analysis`, `neighbors`, `similar`, `relations`, `history` |
| `producer` | 2,410 | one developer |
| `pair_block` | 7,948 | all pairs (a, b > a) for one `a` as `b,pv,nv,tv` little-endian uint16, sorted by b (split at 40 KB) |
| `user_block` | 2,048 | JSON `{uid: {name, votes, similar, recs}}` for users with `uid % 2048 == shard`; votes are base64 `(vn idx uint16, vote uint8)` |
| `user_name` | 64 | JSON `{lower(username): uid}` for `fnv1a(name) % 64 == shard` |
| `meta` | 4 | `snapshot`, `info`, `stats`, `kendall` |

Total: **20,419 rows** and a 147 MB SQL file; an estimated ~100 MB in D1 (pair blocks are 39 MB of it).

Design choices:

* **One row read per page.** Everything a VN page needs is on its `vn` row; a
  user page reads one shard; a pair lookup reads two `vn.idx` values and one
  pair block. Rows written per refresh stay fixed no matter how many users or
  pairs there are, because many small records share one row.
* **Binary where it pays.** 4.9M pairs as JSON would be ~150 MB; packed they
  are 39 MB. `vn.idx` (the VN's position in this snapshot) keeps pair entries
  and user votes at 2 bytes per VN reference.
* **No secondary indexes.** Every query is a primary-key lookup or a full scan
  of `vn` (catalogue / ranks). Indexes would only add writes.
* **The raw data stays offline.** Votes and the N² matrices live only in the
  pipeline's memory (a few GB on a 16 GB GitHub runner).

## Refresh: build, then swap

D1 rejects `BEGIN`/`COMMIT` in imported files. `snapshot.sql` therefore
creates every table as `<name>__next`, fills it, and only at the end runs
`DROP TABLE <name>; ALTER TABLE <name>__next RENAME TO <name>` for each table,
with `meta` (holding the snapshot id) last. Each row is written exactly once,
readers see either the old or the new data, and schema changes ship with the
data (there are no separate migrations).

The Worker includes the snapshot id in every edge-cache key, so a refresh
simply starts using new cache entries.

## Free-tier budget (Workers Free: 100k rows written, 5M rows read per day, 500 MB per DB)

* **Writes:** ~20k per daily refresh.
* **Reads per cache miss:**
  * `/api/catalogue`: ~2N (vn + producer join); `/api/ranks?m=`: N.
    Both are cached at the edge for a week per snapshot, and the browser
    joins them, so switching methods or adding comparison columns only
    fetches the small per-method rank list.
  * `/api/vn/:id`: 1 + up to ~60 (neighbors, relations, similar titles).
  * `/api/pair/:a/:b`: 2 + 1-2. `/api/user/:uid`: 1-2. `/api/user-lookup`: 1.
  * The snapshot id is memoized in each isolate for 60 s.
* **Storage:** an estimated ~100 MB.
* **Import size:** D1 accepts files up to 5 GB.

Edge caching via the Cache API only works on a custom domain, not on
`*.workers.dev`.

## Collaborative filtering (cf.py)

* Votes are mean-centered per user; similarity is cosine over centered
  vectors times `common / (common + 10)`, requiring >= 3 common VNs.
* Users with >= 5 ranked votes get a page (55,778); users with >= 15 are
  eligible as neighbors (31,483). Each user's 30 nearest neighbors predict
  unvoted VNs: `mean + Σ s·dev / Σ s`, damped by `support / (support + 3)`,
  capped at 10, and requiring >= 3 supporting neighbors.
* Similar VNs use the same centered cosine between VN columns
  (`common / (common + 20)`, >= 10 common voters).
* Computed in blocks of 2,000 users with dense candidate matrices; ~80 s on
  a GitHub runner.

## Adding a ranking method

1. Add a function returning one score per VN (higher = better, NaN = unranked)
   in `pipeline/vndb_rank/methods.py` and register it in `compute_all`.
2. Add its name and description in `web/src/lib/i18n.tsx` (`PO`, `MERGED`, ...).
3. Optionally list it in `FEATURED` (`pipeline/vndb_rank/__main__.py`).

No schema change is needed: ranks are stored in the `vn.ranks` JSON column.
