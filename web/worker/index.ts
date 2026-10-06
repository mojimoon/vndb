import { Hono, type Context } from "hono";

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
  const { results } = await db
    .prepare("SELECT key, value FROM meta WHERE key IN ('snapshot', 'info')")
    .all<{ key: string; value: string }>();
  const m = Object.fromEntries(results.map((r) => [r.key, JSON.parse(r.value)]));
  if (!m.snapshot || !m.info) return null;
  metaMemo = { at: Date.now(), snapshot: m.snapshot, info: m.info };
  return metaMemo;
}

// ---------------------------------------------------------------------------
// Edge cache: responses are immutable per snapshot, so the snapshot id is part
// of the cache key and a refresh simply starts using new keys.
async function cachedJson(c: AppContext, snapshot: string, build: () => Promise<unknown | null>) {
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

const meta = (c: AppContext) => c.get("meta");

// GET /api/meta -> snapshot, method list, statistics, method agreement matrix
app.get("/api/meta", async (c) => {
  const m = meta(c);
  return cachedJson(c, m.snapshot, async () => {
    const { results } = await c.env.DB.prepare("SELECT key, value FROM meta").all<{ key: string; value: string }>();
    return Object.fromEntries(results.map((r) => [r.key, JSON.parse(r.value)]));
  });
});

// GET /api/ranking?m=<method> -> every ranked VN with its rank under <method>.
// ~N rows read per cache miss; the client filters/sorts/paginates locally.
app.get("/api/ranking", async (c) => {
  const m = meta(c);
  const method = c.req.query("m") ?? m.info.default_method;
  if (!m.info.methods.includes(method) && method !== "vndb") return c.json({ error: "unknown method" }, 400);
  return cachedJson(c, m.snapshot, async () => {
    const path = `$.${method}`;
    const { results } = await c.env.DB.prepare(
      `SELECT v.id, v.title, v.latin, v.title_ja, v.title_zh, v.title_en, v.olang, v.released,
              v.dev_id, p.name AS dev, p.latin AS dev_latin, v.votes, v.rating,
              json_extract(v.ranks, ?1 || '[0]') AS rank,
              json_extract(v.ranks, ?1 || '[1]') AS score,
              json_extract(v.ranks, '$.vndb[0]') AS vndb_rank,
              v.search
         FROM vn v LEFT JOIN producer p ON p.id = v.dev_id
        ORDER BY rank, v.id`,
    )
      .bind(path)
      .all();
    return { snapshot: m.snapshot, method, items: results };
  });
});

// GET /api/vn/:id -> one VN with all ranks, head-to-head list and relations.
app.get("/api/vn/:id{[0-9]+}", async (c) => {
  const m = meta(c);
  const id = Number(c.req.param("id"));
  return cachedJson(c, m.snapshot, async () => {
    const vn = await c.env.DB.prepare(
      `SELECT v.*, p.name AS dev, p.latin AS dev_latin
         FROM vn v LEFT JOIN producer p ON p.id = v.dev_id WHERE v.id = ?`,
    )
      .bind(id)
      .first<Record<string, unknown> & { ranks: string; neighbors: string; relations: string }>();
    if (!vn) return null;

    const neighbors = JSON.parse(vn.neighbors) as [number, number, number, number][];
    const relations = JSON.parse(vn.relations) as [number, string][];
    const ids = [...new Set([...neighbors.map((n) => n[0]), ...relations.map((r) => r[0])])];
    const { results: others } = await c.env.DB.prepare(
      `SELECT id, title, latin, title_ja, title_zh, title_en, released, votes, rating,
              json_extract(ranks, ?2 || '[0]') AS rank
         FROM vn WHERE id IN (SELECT value FROM json_each(?1))`,
    )
      .bind(JSON.stringify(ids), `$.${m.info.default_method}`)
      .all();

    const { search: _s, snapshot: _snap, ...rest } = vn;
    return {
      ...rest,
      ranks: JSON.parse(vn.ranks),
      neighbors: neighbors.map(([vid, wins, losses, common]) => ({ id: vid, wins, losses, common })),
      relations: relations.map(([vid, relation]) => ({ id: vid, relation })),
      others,
    };
  });
});

app.all("/api/*", (c) => c.json({ error: "not found" }, 404));

export default app;
