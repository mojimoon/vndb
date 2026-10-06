import { Hono, type Context } from "hono";

// Storage formats mirror pipeline/vndb_rank/storage.py and export.py.
const USER_SHARDS = 2048;
const NAME_SHARDS = 64;

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
// Snapshot id + info are read on almost every request, so keep them in isolate
// memory for a minute (1-2 D1 rows read per isolate per minute).
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
// of the cache key and a refresh simply starts using new keys.
async function cachedJson(c: AppContext, build: () => Promise<unknown | null>) {
  const snapshot = c.get("meta").snapshot;
  const url = new URL(c.req.url);
  url.searchParams.set("__snapshot", snapshot);
  const key = new Request(url.toString(), { method: "GET" });
  const cache = caches.default;
  const hit = await cache.match(key);
  if (hit) return hit;

  const body = await build();
  if (body === null) return c.json({ error: "not found" }, 404);
  const res = new Response(JSON.stringify(body), {
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

// GET /api/meta -> snapshot, method list, statistics, method agreement matrix
app.get("/api/meta", (c) =>
  cachedJson(c, async () => {
    const { results } = await c.env.DB.prepare("SELECT key, value FROM meta").all<{ key: string; value: string }>();
    return Object.fromEntries(results.map((r) => [r.key, JSON.parse(r.value)]));
  }),
);

// GET /api/catalogue -> every ranked VN's display fields (~N rows read per cache miss).
app.get("/api/catalogue", (c) =>
  cachedJson(c, async () => {
    const { results } = await c.env.DB.prepare(
      `SELECT v.id, v.idx, v.title, v.latin, v.title_ja, v.title_zh, v.title_en, v.olang, v.released,
              v.dev_id, p.name AS dev, p.latin AS dev_latin, v.votes, v.rating, v.length, v.trend,
              json_extract(v.ranks, '$.vndb[0]') AS vndb_rank, v.search
         FROM vn v LEFT JOIN producer p ON p.id = v.dev_id
        ORDER BY v.idx`,
    ).all();
    return { items: results };
  }),
);

// GET /api/ranks?m=<method> -> [[id, rank, score], ...] sorted by rank.
app.get("/api/ranks", async (c) => {
  const m = c.get("meta");
  const method = c.req.query("m") ?? m.info.default_method;
  if (!isMethod(method, m.info)) return c.json({ error: "unknown method" }, 400);
  return cachedJson(c, async () => {
    const rows = await c.env.DB.prepare(
      `SELECT id, json_extract(ranks, ?1 || '[0]') AS r, json_extract(ranks, ?1 || '[1]') AS s
         FROM vn ORDER BY r, id`,
    )
      .bind(`$.${method}`)
      .raw<[number, number, number | null]>();
    return { method, ranks: rows };
  });
});

// GET /api/vn/:id -> one VN with ranks, analysis, history, head-to-head, similar VNs.
app.get("/api/vn/:id{[0-9]+}", (c) =>
  cachedJson(c, async () => {
    const id = Number(c.req.param("id"));
    const vn = await c.env.DB.prepare(
      `SELECT v.*, p.name AS dev, p.latin AS dev_latin
         FROM vn v LEFT JOIN producer p ON p.id = v.dev_id WHERE v.id = ?`,
    )
      .bind(id)
      .first<Record<string, unknown>>();
    if (!vn) return null;

    const json = (k: string) => JSON.parse(vn[k] as string);
    const neighbors = json("neighbors") as [number, number, number, number][];
    const relations = json("relations") as [number, string][];
    const similar = json("similar") as [number, number, number][];
    const ids = [...new Set([...neighbors.map((n) => n[0]), ...relations.map((r) => r[0]), ...similar.map((s) => s[0])])];
    const { results: others } = await c.env.DB.prepare(
      `SELECT id, title, latin, title_ja, title_zh, title_en, released, votes, rating,
              json_extract(ranks, ?2 || '[0]') AS rank
         FROM vn WHERE id IN (SELECT value FROM json_each(?1))`,
    )
      .bind(JSON.stringify(ids), `$.${c.get("meta").info.default_method}`)
      .all();

    const { search: _s, ...rest } = vn;
    return {
      ...rest,
      ranks: json("ranks"),
      analysis: json("analysis"),
      history: json("history"),
      neighbors: neighbors.map(([vid, wins, losses, common]) => ({ id: vid, wins, losses, common })),
      relations: relations.map(([vid, relation]) => ({ id: vid, relation })),
      similar: similar.map(([vid, sim, common]) => ({ id: vid, sim, common })),
      others,
    };
  }),
);

// GET /api/pair/:a/:b -> head-to-head of any two ranked VNs (from the pair blocks).
app.get("/api/pair/:a{[0-9]+}/:b{[0-9]+}", (c) =>
  cachedJson(c, async () => {
    const a = Number(c.req.param("a"));
    const b = Number(c.req.param("b"));
    if (a === b) return null;
    const { results } = await c.env.DB.prepare("SELECT id, idx FROM vn WHERE id IN (?, ?)").bind(a, b).all<{ id: number; idx: number }>();
    const idx = new Map(results.map((r) => [r.id, r.idx]));
    if (!idx.has(a) || !idx.has(b)) return null;
    const ia = idx.get(a)!;
    const ib = idx.get(b)!;
    const [lo, hi] = ia < ib ? [ia, ib] : [ib, ia];
    const { results: parts } = await c.env.DB.prepare("SELECT data FROM pair_block WHERE a = ? ORDER BY part").bind(lo).all<{ data: unknown }>();
    for (const part of parts) {
      const bytes = toBytes(part.data);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      let l = 0;
      let r = bytes.byteLength / 8 - 1;
      if (r < 0 || view.getUint16(r * 8, true) < hi) continue;
      while (l <= r) {
        const mid = (l + r) >> 1;
        const key = view.getUint16(mid * 8, true);
        if (key === hi) {
          const pv = view.getUint16(mid * 8 + 2, true);
          const nv = view.getUint16(mid * 8 + 4, true);
          const tv = view.getUint16(mid * 8 + 6, true);
          const [wins, losses] = ia < ib ? [pv, nv] : [nv, pv];
          return { a, b, wins, losses, common: tv };
        }
        if (key < hi) l = mid + 1;
        else r = mid - 1;
      }
      break;
    }
    // Fewer than min_common_vote users voted on both.
    return { a, b, wins: 0, losses: 0, common: 0, below_threshold: true };
  }),
);

// GET /api/user/:uid -> votes (as [vn idx, vote]), similar users, recommendations.
app.get("/api/user/:uid{[0-9]+}", (c) =>
  cachedJson(c, async () => {
    const uid = Number(c.req.param("uid"));
    const { results } = await c.env.DB.prepare("SELECT data FROM user_block WHERE shard = ? ORDER BY part")
      .bind(uid % USER_SHARDS)
      .all<{ data: string }>();
    for (const r of results) {
      const shard = JSON.parse(r.data) as Record<string, { name: string; votes: string; similar: unknown[]; recs: unknown[] }>;
      const u = shard[String(uid)];
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

// GET /api/user-lookup?name=<username> -> { uid }
app.get("/api/user-lookup", (c) =>
  cachedJson(c, async () => {
    const name = (c.req.query("name") ?? "").trim().toLowerCase();
    if (!name) return null;
    const { results } = await c.env.DB.prepare("SELECT data FROM user_name WHERE shard = ? ORDER BY part")
      .bind(fnv1a(name) % NAME_SHARDS)
      .all<{ data: string }>();
    for (const r of results) {
      const uid = (JSON.parse(r.data) as Record<string, number>)[name];
      if (uid) return { uid };
    }
    return null;
  }),
);

app.all("/api/*", (c) => c.json({ error: "not found" }, 404));

export default app;
