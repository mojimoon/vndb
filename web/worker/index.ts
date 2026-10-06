import { Hono, type Context } from "hono";

// Storage formats mirror pipeline/vndb_rank/storage.py and export.py.
const USER_SHARDS = 2048;
const NAME_SHARDS = 64;
const VOTER_SHARDS = 1024;
const VN_NOTE_SHARDS = 512;
const USER_NOTE_SHARDS = 1024;
const NOTES_PAGE = 30;

interface Info {
  methods: string[];
  featured: string[];
  default_method: string;
  [k: string]: unknown;
}
interface Meta {
  at: number;
  snapshot: string;
  info: Info;
}

type AppEnv = { Bindings: Env; Variables: { meta: Meta } };
type AppContext = Context<AppEnv>;

const app = new Hono<AppEnv>();

// ---------------------------------------------------------------------------
// Snapshot id + info are needed by every request, so keep them in isolate
// memory for a minute (2 D1 rows read per isolate per minute).
let metaMemo: Meta | null = null;
const META_TTL_MS = 60_000;

async function currentMeta(db: D1Database): Promise<Meta | null> {
  if (metaMemo && Date.now() - metaMemo.at < META_TTL_MS) return metaMemo;
  let results: { key: string; value: string }[];
  try {
    ({ results } = await db
      .prepare("SELECT key, value FROM meta WHERE key IN ('snapshot', 'info')")
      .all<{ key: string; value: string }>());
  } catch (e) {
    // Before the first import there are no tables at all.
    if (String(e).includes("no such table")) return null;
    throw e;
  }
  const m = Object.fromEntries(results.map((r) => [r.key, JSON.parse(r.value)]));
  if (!m.snapshot || !m.info) return null;
  metaMemo = { at: Date.now(), snapshot: m.snapshot, info: m.info };
  return metaMemo;
}

// ---------------------------------------------------------------------------
// Edge cache: responses are immutable per snapshot, so the snapshot id is part
// of the cache key and a refresh simply starts using new keys. `build` returns
// a JSON string (passed through as is), an object, or null for 404.
async function cached(c: AppContext, build: () => Promise<string | object | null>) {
  const snapshot = c.get("meta").snapshot;
  const url = new URL(c.req.url);
  url.searchParams.set("__snapshot", snapshot);
  const key = new Request(url.toString(), { method: "GET" });
  const cache = caches.default;
  const hit = await cache.match(key);
  if (hit) return hit;

  const body = await build();
  if (body === null) return c.json({ error: "not found" }, 404);
  const res = new Response(typeof body === "string" ? body : JSON.stringify(body), {
    headers: {
      "content-type": "application/json; charset=utf-8",
      // Browsers revalidate after 5 minutes; the edge keeps it for a week.
      "cache-control": "public, max-age=300, s-maxage=604800",
      "x-snapshot": snapshot,
    },
  });
  c.executionCtx.waitUntil(cache.put(key, res.clone()));
  return res;
}

const toBytes = (v: unknown): Uint8Array =>
  v instanceof Uint8Array ? v : v instanceof ArrayBuffer ? new Uint8Array(v) : Uint8Array.from(v as number[]);

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(s)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function decodeVotes(b64: string): [number, number][] {
  const bin = atob(b64);
  const out: [number, number][] = [];
  for (let i = 0; i + 2 < bin.length; i += 3) {
    out.push([bin.charCodeAt(i) | (bin.charCodeAt(i + 1) << 8), bin.charCodeAt(i + 2)]);
  }
  return out;
}

/** All parts of one row group, in order (1 row read per part). */
async function parts<T = string>(db: D1Database, table: string, keyCol: string, key: string | number): Promise<T[]> {
  const { results } = await db.prepare(`SELECT data FROM ${table} WHERE ${keyCol} = ? ORDER BY part`).bind(key).all<{ data: T }>();
  return results.map((r) => r.data);
}

const doc = async (db: D1Database, key: string) => {
  const p = await parts(db, "doc", "key", key);
  return p.length ? p.join("") : null;
};

interface Voter {
  uid: Uint32Array;
  vote: Uint8Array;
  spd: Uint8Array;
}

/** Voters of one VN from its shard (format: storage.voter_shards). */
function readVoters(blobs: unknown[], idx: number): Voter {
  const uid: number[] = [];
  const vote: number[] = [];
  const spd: number[] = [];
  for (const blob of blobs) {
    const b = toBytes(blob);
    const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
    let pos = 0;
    while (pos + 6 <= b.byteLength) {
      const i = view.getUint16(pos, true);
      const n = view.getUint32(pos + 2, true);
      pos += 6;
      if (i === idx) {
        for (let k = 0; k < n; k++) {
          const o = pos + k * 6;
          uid.push(view.getUint32(o, true));
          vote.push(b[o + 4]);
          spd.push(b[o + 5]);
        }
      }
      pos += n * 6;
    }
  }
  return { uid: Uint32Array.from(uid), vote: Uint8Array.from(vote), spd: Uint8Array.from(spd) };
}

const isMethod = (m: string, info: Info) => info.methods.includes(m);

app.onError((err, c) => {
  console.error(err);
  return c.json({ error: "internal error" }, 500);
});

app.use("/api/*", async (c, next) => {
  const meta = await currentMeta(c.env.DB);
  if (!meta) return c.json({ error: "no data imported yet" }, 503);
  c.set("meta", meta);
  await next();
});

// GET /api/meta -> snapshot, method list, statistics, agreement matrix, leaderboards
app.get("/api/meta", (c) =>
  cached(c, async () => {
    const { results } = await c.env.DB.prepare("SELECT key, value FROM meta").all<{ key: string; value: string }>();
    return `{${results.map((r) => `${JSON.stringify(r.key)}:${r.value}`).join(",")}}`;
  }),
);

// GET /api/catalogue -> {columns, rows}: display fields of every ranked VN (prebuilt, ~25 rows read).
app.get("/api/catalogue", (c) => cached(c, () => doc(c.env.DB, "catalogue")));

// GET /api/ranks?m=<method> -> {method, ranks: [[id, rank, score], ...]} (prebuilt).
app.get("/api/ranks", async (c) => {
  const m = c.get("meta");
  const method = c.req.query("m") ?? m.info.default_method;
  if (!isMethod(method, m.info)) return c.json({ error: "unknown method" }, 400);
  return cached(c, () => doc(c.env.DB, `ranks:${method}`));
});

// GET /api/vn/:id -> one VN row (titles of related VNs come from the catalogue). 1 row read.
app.get("/api/vn/:id{[0-9]+}", (c) =>
  cached(c, async () => {
    const vn = await c.env.DB.prepare(
      `SELECT id, idx, title, latin, title_ja, title_zh, title_en, olang, released, dev_id, image, image_sexual,
              length, votes, rating, average, trend, ranks, neighbors, relations, analysis, similar, history
         FROM vn WHERE id = ?`,
    )
      .bind(Number(c.req.param("id")))
      .first<Record<string, unknown>>();
    if (!vn) return null;
    const raw = new Set(["ranks", "neighbors", "relations", "analysis", "similar", "history"]);
    // JSON columns are spliced in verbatim instead of parsed and re-serialized.
    return `{${Object.entries(vn)
      .map(([k, v]) => `${JSON.stringify(k)}:${raw.has(k) ? v : JSON.stringify(v)}`)
      .join(",")}}`;
  }),
);

// GET /api/joint/:a/:b (VN idx) -> exact head-to-head and 10x10 joint distributions of
// raw votes and sample-percentile deciles among the users who voted on both.
app.get("/api/joint/:a{[0-9]+}/:b{[0-9]+}", (c) =>
  cached(c, async () => {
    const a = Number(c.req.param("a"));
    const b = Number(c.req.param("b"));
    if (a === b) return null;
    const sa = a % VOTER_SHARDS;
    const sb = b % VOTER_SHARDS;
    const blobsA = await parts<unknown>(c.env.DB, "vn_voters", "shard", sa);
    const blobsB = sa === sb ? blobsA : await parts<unknown>(c.env.DB, "vn_voters", "shard", sb);
    const va = readVoters(blobsA, a);
    const vb = readVoters(blobsB, b);
    if (!va.uid.length || !vb.uid.length) return null;
    const raw = Array.from({ length: 10 }, () => Array(10).fill(0));
    const sp = Array.from({ length: 10 }, () => Array(10).fill(0));
    let wins = 0;
    let losses = 0;
    let common = 0;
    // Both lists are sorted by uid: merge-intersect.
    for (let i = 0, j = 0; i < va.uid.length && j < vb.uid.length; ) {
      if (va.uid[i] < vb.uid[j]) i++;
      else if (va.uid[i] > vb.uid[j]) j++;
      else {
        const x = va.vote[i];
        const y = vb.vote[j];
        raw[Math.min(9, Math.max(0, Math.floor(x / 10) - 1))][Math.min(9, Math.max(0, Math.floor(y / 10) - 1))]++;
        sp[va.spd[i]][vb.spd[j]]++;
        if (x > y) wins++;
        else if (x < y) losses++;
        common++;
        i++;
        j++;
      }
    }
    return { a, b, wins, losses, common, raw, sp };
  }),
);

// GET /api/notes/vn/:idx?page=N -> notes on one VN, newest first, NOTES_PAGE per page.
app.get("/api/notes/vn/:idx{[0-9]+}", (c) =>
  cached(c, async () => {
    const idx = Number(c.req.param("idx"));
    const page = Math.max(0, Number(c.req.query("page") ?? 0) || 0);
    const rows: unknown[][] = [];
    for (const p of await parts(c.env.DB, "vn_notes", "shard", idx % VN_NOTE_SHARDS)) {
      for (const r of JSON.parse(p) as unknown[][]) if (r[0] === idx) rows.push(r);
    }
    const slice = rows.slice(page * NOTES_PAGE, (page + 1) * NOTES_PAGE);
    return {
      total: rows.length,
      page,
      pageSize: NOTES_PAGE,
      notes: slice.map(([, uid, name, vote, date, text, hasPage]) => ({ uid, name, vote, date, text, hasPage })),
    };
  }),
);

// GET /api/user/:uid -> votes (as [vn idx, vote]), similar users, recommendations.
app.get("/api/user/:uid{[0-9]+}", (c) =>
  cached(c, async () => {
    const uid = Number(c.req.param("uid"));
    for (const p of await parts(c.env.DB, "user_block", "shard", uid % USER_SHARDS)) {
      const u = (JSON.parse(p) as Record<string, { name: string; votes: string; similar: unknown[]; recs: unknown[] }>)[String(uid)];
      if (u) {
        return {
          uid,
          name: u.name,
          votes: decodeVotes(u.votes),
          similar: (u.similar as [number, string, number, number][]).map(([id, name, sim, common]) => ({ uid: id, name, sim, common })),
          recs: (u.recs as [number, number, number][]).map(([idx, pred, support]) => ({ idx, pred, support })),
        };
      }
    }
    return null;
  }),
);

// GET /api/user/:uid/notes -> all of the user's notes on ranked VNs.
app.get("/api/user/:uid{[0-9]+}/notes", (c) =>
  cached(c, async () => {
    const uid = Number(c.req.param("uid"));
    const notes: { idx: unknown; vote: unknown; date: unknown; text: unknown }[] = [];
    for (const p of await parts(c.env.DB, "user_notes", "shard", uid % USER_NOTE_SHARDS)) {
      for (const [u, idx, vote, date, text] of JSON.parse(p) as unknown[][]) if (u === uid) notes.push({ idx, vote, date, text });
    }
    return { uid, notes };
  }),
);

// GET /api/user-lookup?name=<username> -> { uid }
app.get("/api/user-lookup", (c) =>
  cached(c, async () => {
    const name = (c.req.query("name") ?? "").trim().toLowerCase();
    if (!name) return null;
    for (const p of await parts(c.env.DB, "user_name", "shard", fnv1a(name) % NAME_SHARDS)) {
      const uid = (JSON.parse(p) as Record<string, number>)[name];
      if (uid) return { uid };
    }
    return null;
  }),
);

app.all("/api/*", (c) => c.json({ error: "not found" }, 404));

export default app;
