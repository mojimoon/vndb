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
       ├─ extract 14 tables from the 190 MB .tar.zst
       ├─ fetch the stored history from D1 (`history_day`, .github/scripts/fetch-history.sh; any error stops the run)
       ├─ pipeline/ (python -m vndb_rank, ~10 min)
       │    extract.py    catalogue (VNs with >= 30 votes), users, one streaming pass over ulist_vns
       │    analysis.py   per-VN rating analysis
       │    pairs.py      N x N pair counts + mean votes / sample percentiles
       │    methods.py    7 PONet, 24 rankit, 8 Borda methods + VNDB reference
       │    leaderboards.py user leaderboards
       │    tags.py       VNDB-style tag ratings, document frequencies, IDF
       │    neighbors.py  per-VN head-to-head opponents
       │    cf.py         similar users, recommendations, similar VNs
       │    history.py    daily rank / VNDB rating / votes snapshots, thinning
       │    storage.py    voter blobs, user / note shards, text parts
       │    export.py     snapshot.sql (drop + recreate every table except history_day)
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
| `history_day` | ~3 per day | every VN ranked that day, by rank: 14-byte records `vid u32, default-method rank u16, VNDB rank u16, VNDB rating ×100 u16, votes u32` (`history.HIST_DTYPE`), base64; kept across snapshots |
| `producer` | 2,410 | one developer |
| `doc` | ~190 | a prebuilt API response split into ≤ 80 KB parts: `catalogue` (columnar), `ranks:<method>`, `tags` (tag dictionary) and `vntags` (base64 tags of every ranked VN) |
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

### Chinese titles

VNDB has a Chinese title for only about a quarter of the ranked titles. For
the rest `title_zh` is filled in this order (`extract._attach_titles`):

1. VNDB's own Chinese titles (`vn_titles`, official ones first; zh-Hans, then
   zh-Hant, then zh).
2. The title of an official, non-patch, human-translated Chinese release of
   that title alone (`releases_titles`), with edition, version and episode
   suffixes stripped and titles mixing in Latin letters skipped. Of several,
   the one most others start with wins ("花冠之淚", "機械學報告").
3. A Chinese alias (Han characters, no kana). If the original title is in kana
   or hangul any such alias beats nothing, so the first is used. If it is in
   Latin script the original is readable and often what Chinese players use,
   so an alias is used only when it is the only Chinese one (Fate/stay night →
   命运之夜) or a Chinese release title contains it. Rewrite's two aliases
   (罚抄, a joke, and 改写, a literal translation) are therefore ignored.
4. Otherwise none: the UI shows the Japanese or original title.

There is no perfect rule: a Latin-script title with several Chinese aliases
(FORTUNE ARTERIAL: 命运动脉 / 赤之约定) keeps its original title.

### Cover images

`image_sexual` / `image_violence` are the means of VNDB users' flags (0 safe /
tame, 1 suggestive / violent, 2 explicit / brutal); the level is the rounded
mean. Covers at level 1 (and unflagged ones) are blurred until clicked; covers
at level 2 are never loaded, a placeholder links to VNDB instead.

### Tags

`tags.py` reads `tags` and `tags_vn` and rates each VN's tags as VNDB does:
the mean of the non-ignored votes (-3..3), kept when above 0; the spoiler
level is the mean spoiler vote bucketed at > 0.4 (minor) and > 1.3 (major), the
tag's default when nobody voted on it, and major for tags marked as a lie.
Only searchable tags are kept. Names are shown as on VNDB (never translated).

* Title page (`vn.tags` column): score = significance × ln(N / df), N = VNs
  with any tag, df = VNs carrying the tag, both over all of VNDB.
* User page: computed in the browser from the user's votes and `/api/vntags`
  (one cached ~2 MB binary of every ranked VN's tags: u32 n, u32 offsets,
  then tag u16 / rating × 50 u8 / spoiler u8). TF = the tag's significance
  summed over the user's voted titles ÷ their number of votes; IDF is
  vote-weighted, ln(V / V_t), V = votes on ranked titles and V_t = votes on
  those carrying the tag, so tags that nearly every list contains weigh
  little. Storing per-user tag profiles instead would have cost ~90 MB for
  56k users.

The cloud shows the top 35 (20 on phones) content tags, coloured by
significance (grey 0 -> teal -> blue -> violet 3). Size is the geometric mean
of the score's position between the lowest and highest shown score and a
Zipf-like rank curve (1/sqrt(rank)), so both clouds have a few big tags: a
title's top scores are packed closely (significance 1.5-3 x similar IDFs),
while a user's scores have a long tail. The table defaults to VNDB's view
(content + technical, no spoilers) and marks spoilers like VNDB, a small
yellow "s" (minor) or a red "S" (major); on title pages the cloud follows the
table's spoiler setting. User pages have no spoiler levels: a tag summed over
many titles does not spoil any one of them.

### History

Each refresh records, for every ranked VN, its rank under the default method
(the grand ranking), its VNDB rank, VNDB rating and vote count, as one
`history_day` row group (~150 KB per day for 8k titles). Kept: every day for
90 days, then the first recorded day of each week (and of each month) for a
year, then the first recorded day of each month (`history.keep_days`).

`history_day` is the only table a snapshot does not rebuild. The refresh
fetches it a few days per query, the pipeline adds today and thins it, and the
snapshot deletes the thinned days and inserts the new one in place. Everything
derived from it is rebuilt each time: `vn.history` (`[[day, rank, vndb_rank,
rating, votes], ...]`, the charts on a title's Ranks tab), `vn.trend` (30-day
change, Movers) and `info.history_days` (the dates the Ranking → History tab
offers; `/api/history/:day` returns one day as binary).

Safeguards:

* The fetch fails the run on any error except a missing table, and checks that
  it received every part of every day. Wrong or partial history is never
  built on.
* The snapshot only deletes days it fetched, and inserts new days with plain
  `INSERT`s, so a day it did not know about makes the import fail (and roll
  back) instead of being overwritten.
* Titles that leave the ranking stay in the stored days (shown as "no longer
  ranked" in the History tab) and get their chart back if they return.
* Each run uploads the fetched history as the `history-backup` artifact
  (kept 30 days). To restore, download it and run the pipeline with
  `--history history.json` against an empty `history_day`.
* `--skip-rankit` runs (different default method) do not record a day.

Before `history_day` existed the history was a per-VN column holding only the
two ranks. The first refresh with this version reads that column instead
(`--legacy-history`) and turns it into days with no rating or votes.

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
`meta` (holding the snapshot id) last. The exception is `history_day`, which is
updated in place (see History). D1 rejects `BEGIN`/`COMMIT` in imported
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
  * `/api/history/:day`: 2-3.
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
