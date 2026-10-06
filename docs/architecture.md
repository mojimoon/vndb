# Architecture

## Why the Supabase version did not fit

The previous design uploaded per-user votes (`ulist`, one row per user × VN,
millions of rows) to Supabase and planned to compute or serve rankings from
there. Raw votes are only needed to *build* the partial order network; the site
only ever needs the results. Shipping inputs instead of outputs is what blew
through a small instance's storage.

## Data flow

```
dl.vndb.org dump (daily, ~08:00 UTC)
  └─ GitHub Actions refresh.yml (12:07 UTC)
       ├─ extract 10 tables (~0.8 GB) from the 190 MB .tar.zst
       ├─ pipeline/ (python -m vndb_rank)
       │    extract.py    catalogue (VNs with >= 30 votes) + one streaming pass over ulist_vns
       │    pairs.py      N x N pair counts (pv, nv, tv) + mean votes / sample percentiles
       │    methods.py    8 PONet methods, 40 rankit methods, 8 Borda merges, + VNDB reference
       │    neighbors.py  per-VN head-to-head opponents
       │    export.py     snapshot.sql
       └─ wrangler d1 execute --remote --file snapshot.sql
Cloudflare D1  <──  Worker (Hono, worker/index.ts)  <──  React SPA (static assets)
```

## Schema (`web/migrations/0001_init.sql`)

| table | rows (2026-10 dump) | written per refresh |
| --- | --- | --- |
| `vn` | ~7.9k (VNs with >= 30 votes) | all |
| `producer` | developers referenced by `vn` | all |
| `meta` | 4 (`snapshot`, `info`, `stats`, `kendall`) | all |

Design choices:

* **Denormalize what is read together.** A VN page needs that VN's rank under
  all 57 methods and its head-to-head list. A normalized `rank(method, vid)`
  table would be ~450k rows and a pair table several million, and both are
  rewritten on every refresh. As JSON columns on `vn` they cost one row write
  per VN and one row read per page view.
* **No secondary indexes.** Every query is a primary-key lookup or a full scan
  of `vn` (the ranking list). Indexes would multiply rows written on refresh
  and save nothing.
* **The raw data stays offline.** Votes and the N² matrices live only in the
  pipeline's memory (~3 GB for 8k VNs on a 16 GB GitHub runner).

## Refresh without a transaction

D1 rejects `BEGIN`/`COMMIT` in imported files, so `snapshot.sql`:

1. upserts every `vn` / `producer` row stamped with the new snapshot id,
2. deletes rows whose snapshot differs (VNs that dropped out),
3. writes `meta.snapshot` last.

The Worker includes the snapshot id in every edge-cache key, so the switch to
new data happens atomically from a reader's point of view once step 3 lands.
During the few seconds of the import, an uncached request may see a mix of old
and new rows, which is acceptable for this site.

## Free-tier budget (Workers Free: 100k rows written, 5M rows read per day, 500 MB per DB)

* **Writes:** a refresh writes ~N + producers + 4 rows (upserts may count
  double), roughly 20-25k, well below 100k/day. Daily refreshes are fine.
* **Reads:**
  * `/api/ranking?m=` scans `vn` and joins `producer`, ~2N rows per cache miss.
    The response is cached at the edge per (snapshot, method) for a week, and
    in the browser for 5 minutes.
  * `/api/vn/:id` reads 1 + up to ~60 rows (neighbors and relations).
  * The snapshot id is memoized in each isolate for 60 s.
* **Storage:** ~4 KB per VN row (ranks + neighbors JSON) → ~35-40 MB.

Edge caching via the Cache API only works on a custom domain, not on
`*.workers.dev`.

## Pipeline performance

* `pairs.py` loops over users (one vectorized `triu_indices` block each) and
  accumulates into dense N x N arrays: 3 x int32 counts + 8 x float32 sums.
* The rankit methods dominate the runtime (each builds a `Table` by iterating
  rows in Python). `--skip-rankit` computes only the PONet methods for quick
  iterations.
* Users' items are processed in ascending matrix-index order so every pair lands
  in the upper triangle (the legacy script lost about half the comparisons here).

## Adding a ranking method

1. Add a function returning one score per VN (higher = better, NaN = unranked)
   in `pipeline/vndb_rank/methods.py` and register it in `compute_all`.
2. Add its name and description in `web/src/lib/i18n.tsx` (`PO`, `MERGED`, ...).
3. Optionally list it in `FEATURED` (`pipeline/vndb_rank/__main__.py`).

No schema change is needed: ranks are stored in the `vn.ranks` JSON column.
