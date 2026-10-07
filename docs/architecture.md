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
       │    methods.py    7 PONet, 24 rankit, 8 Borda methods + VNDB reference
       │    leaderboards.py user leaderboards
       │    neighbors.py  per-VN head-to-head opponents
       │    cf.py         similar users, recommendations, similar VNs
       │    storage.py    voter blobs, user / note shards, text parts, history
       │    export.py     snapshot.sql (drop + recreate every table)
       └─ wrangler d1 execute --remote --file snapshot.sql
Cloudflare D1  <──  Worker (Hono, web/worker/index.ts)  <──  React SPA (static assets)
```

Votes of users VNDB flags `ign_votes` are dropped everywhere. On the
2026-10-06 dump that removed ~2.2M of 7.1M comparable pairs.

### Which ranking methods are kept

Every method was checked on the 2026-10-06 dump against the consensus (median
rank over all methods, Kendall tau) and against popularity (Spearman rho with
log vote count). Dropped:

| method(s) | tau vs consensus | why |
| --- | --- | --- |
| `po_rw` (random walk) | -0.04 | unrelated to every other method |
| `keener_*` | 0.39-0.56 | rho ~0.75 with vote count: a popularity contest |
| `markov_rv_*` | 0.42-0.62 | same (rho ~0.77) |
| `difference_prob/ari/geo` | 0.35-0.62 | top lists are simply the most-voted titles |
| `massey_prob` | 0.41 | raw preference counts let big pairs dominate |

`po_total` (tau 0.63) shows the same bias but is kept as one of the original
PONet scores, without being featured. `difference_sp_ari` and
`difference_sp_geo` were removed in v3 and added back in v5 as inputs of the
grand ranking. 41 methods remain. The grand ranking (`borda_grand`) is a Borda
count over 9 of them; see [ranking-experiments.md](ranking-experiments.md)
for the full history of methods and the experiments behind the grand ranking.

## Schema (defined in `pipeline/vndb_rank/export.py`)

Numbers from the 2026-10-06 dump:

| table | rows | what one row holds |
| --- | --- | --- |
| `vn` | 7,945 | one VN: display fields + JSON columns `ranks`, `analysis`, `neighbors`, `similar`, `relations`, `history` |
| `producer` | 2,410 | one developer |
| `doc` | ~190 | a prebuilt API response split into ≤ 80 KB parts: `catalogue` (columnar) and `ranks:<method>` |
| `vn_voters` | ~1.6k | voters of the VNs with `idx % 1024 == shard`: segments `(idx u16, n u32)` + n × 13-byte records sorted by uid: `uid u32, vote u8, sample percentile ×200 u8, list labels u8, year−1990 u8, the user's vote count u16, the user's mean vote u8, r vs VNDB ×100 i8, r vs SciRanking ×100 i8` (`storage.VOTER_DTYPE`) |
| `vn_notes` | ~700 | JSON `[[idx, uid, name, vote, date, text, has_page, labels, user_votes, sp%], ...]` for `idx % 512 == shard`, newest first |
| `user_notes` | ~1.3k | JSON `[[uid, idx, vote, date, text, labels, sp%], ...]` for `uid % 1024 == shard` |
| `user_block` | 2,048 | JSON `{uid: {name, votes, similar, recs}}` for `uid % 2048 == shard`; votes are base64 `(vn idx u16, vote u8)` |
| `user_name` | 64 | JSON `{lower(username): uid}` for `fnv1a(name) % 64 == shard` |
| `meta` | 5 | `snapshot`, `info`, `stats`, `kendall`, `leaderboards` |

Design choices:

* **Serve prebuilt payloads.** The catalogue (every ranked VN's display
  fields, ~2 MB of JSON) and each method's rank list are stored as finished
  JSON, so the worker concatenates a few rows and returns them without
  parsing: ~25 rows read for the catalogue instead of ~16k.
* **The catalogue is the shared lookup.** Every page loads it once (cached by
  the browser and the edge) and resolves titles and developers from it, so
  `/api/vn/:id` is a single-row read and developer pages need no API at all.
* **Voter lists instead of a pair table.** Head-to-head counts and the 10×10
  joint vote / percentile matrices of any two VNs are computed in the worker
  by intersecting two uid-sorted voter lists. That replaced 7.9k rows (39 MB)
  of precomputed pairs with ~11 MB and also covers pairs with < 5 common voters.
* **Shard many small records into one row.** Users, notes and voters are
  grouped by `id % shards`, so rows written per refresh stay fixed no matter
  how many users or notes there are, and a page reads one shard (1-10 rows).
* **No secondary indexes.** Every query is a primary-key lookup.
* **The raw data stays offline.** Votes and the N² matrices live only in the
  pipeline's memory (a few GB on a 16 GB GitHub runner).

### Notes ("reviews")

The dump carries the free-text notes of public user lists. On ranked VNs there
are 160k non-empty notes (16.9 MB); the 105k with at least 20 characters are
kept (capped at 3,000 characters each). They are stored twice, by VN and by
user, which costs ~33 MB and ~2k row writes per day, and are only fetched when
a reviews tab is opened. The Worker sorts (time, rating, sample percentile)
and filters (list labels, the author's vote count) a VN's reviews before
paging, so every query costs the same shard read and is cached on its own.
Pass `--skip-notes` to the pipeline to leave them out.

### Voter filters

The ratings tab can recompute a VN's analysis over a subset of its voters
(list status, the voter's vote count, the voter's correlation with VNDB
ratings or SciRanking). `/api/voters/:idx` returns the VN's voter records
without uids (9 bytes each, ~20 KB for a typical VN) from the same shard the
joint endpoint reads; the browser does the filtering. The user-level fields
are copied onto every vote (1.9M ranked votes × 13 bytes ≈ 24 MB) so the
request never has to look users up.

## Refresh: replace in one import

D1 runs an imported file atomically: the database does not serve queries
while the import runs, and any failure rolls it back to the previous snapshot.
`snapshot.sql` therefore simply drops every table and recreates it, with
`meta` (holding the snapshot id) last. D1 rejects `BEGIN`/`COMMIT` in imported
files, so the file has none. Schema changes ship with the data (there are no
separate migrations).

Dropping first matters for the 500 MB database limit. An earlier version built
`<name>__next` tables beside the live ones and swapped them at the end, so
every import briefly held two full copies (plus the pages freed by previous
imports, which SQLite keeps). Once the snapshot passed ~165 MB that no longer
fit and the import failed in its last step. Writing over the dropped tables
reuses their pages, so the database stays at about one snapshot.

The Worker includes the snapshot id in every edge-cache key, so a refresh
simply starts using new cache entries. When a response format changes, bump
`API_VERSION` in both `web/worker/index.ts` and `web/src/lib/api.ts`: it is
part of every request URL and every edge-cache key, so neither browsers nor
the edge can hand an old-format response to new code.

## Free-tier budget (Workers Free: 100k rows written, 5M rows read per day, 500 MB per DB)

* **Writes:** see the summary of the latest refresh (about 15k rows per day).
* **Reads per cache miss** (every response is cached at the edge for a week per
  snapshot, and in the browser for 5 minutes):
  * `/api/meta`: 5. `/api/catalogue`: ~25. `/api/ranks?m=`: 2-3.
  * `/api/vn/:id`: 1. `/api/user/:uid`: 1-2. `/api/user-lookup`: 1.
  * `/api/joint/:a/:b`: 1-10 (two voter shards). `/api/voters/:idx`: 1-5.
  * `/api/notes/vn/:idx`: 1-15. `/api/user/:uid/notes`: 1-2.
  * The snapshot id is memoized in each isolate for 60 s (2 rows).
  * The app requests `/api/meta` and `/api/catalogue` once at startup; every
    later page reuses them.
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
  (`common / (common + 20)`, >= 10 common voters), and carry their
  head-to-head counts from the pair table.
* Similar users carry a head-to-head over their common titles (who voted
  higher, equal) and the other user's ranked vote count.
* Computed in blocks of 2,000 users with dense candidate matrices; ~80 s on
  a GitHub runner.

## Adding a ranking method

1. Add a function returning one score per VN (higher = better, NaN = unranked)
   in `pipeline/vndb_rank/methods.py` and register it in `compute_all`.
2. Add its name and description in `web/src/lib/i18n.tsx` (`PO`, `MERGED`, ...).
3. Optionally list it in `FEATURED` (`pipeline/vndb_rank/__main__.py`).

No schema change is needed: ranks are stored in the `vn.ranks` JSON column.
