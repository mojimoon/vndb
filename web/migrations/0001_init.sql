-- VNDB Ranking+ schema for Cloudflare D1.
--
-- Sized for the D1 free tier (100k rows written / 5M rows read per day,
-- 500 MB per database): a full weekly refresh writes ~N VNs + their developers
-- (roughly 10-15k rows) instead of millions of raw votes or N^2 pairs.
-- Raw votes and the full pair matrix stay in the offline pipeline.

-- One row per ranked VN (>= min_vote votes).
CREATE TABLE vn (
  id           INTEGER PRIMARY KEY,   -- VNDB id without the "v" prefix
  title        TEXT    NOT NULL,      -- title in the original language
  latin        TEXT,                  -- romanization of `title`
  title_ja     TEXT,
  title_zh     TEXT,                  -- zh-Hans > zh-Hant > Chinese alias
  title_en     TEXT,
  olang        TEXT,
  released     INTEGER,               -- YYYYMMDD of the first full release (MM/DD may be 99)
  dev_id       INTEGER,               -- producer.id of the main developer
  image        INTEGER,               -- cover id ("cv" + id on t.vndb.org)
  image_sexual REAL,                  -- 0 safe .. 2 explicit (VNDB image flagging)
  length       INTEGER,
  votes        INTEGER NOT NULL,      -- VNDB vote count
  rating       REAL,                  -- VNDB Bayesian rating, 1-10
  average      REAL,                  -- VNDB raw average, 1-10
  search       TEXT    NOT NULL DEFAULT '',  -- normalized titles + aliases
  ranks        TEXT    NOT NULL,      -- JSON {method: [rank, score]}
  neighbors    TEXT    NOT NULL DEFAULT '[]', -- JSON [[vid, wins, losses, common], ...]
  relations    TEXT    NOT NULL DEFAULT '[]', -- JSON [[vid, relation], ...]
  snapshot     TEXT    NOT NULL
);

CREATE TABLE producer (
  id       INTEGER PRIMARY KEY,
  name     TEXT NOT NULL,
  latin    TEXT,
  search   TEXT NOT NULL DEFAULT '',
  snapshot TEXT NOT NULL
);

-- Small JSON documents: snapshot id, info (methods, config), stats, kendall.
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;
