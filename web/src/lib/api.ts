import { useEffect, useMemo, useState } from "react";

// ---------------------------------------------------------------------------
// Response types (see web/worker/index.ts)

export interface TitleFields {
  id: number;
  title: string;
  latin: string | null;
  title_ja: string | null;
  title_zh: string | null;
  title_en: string | null;
}

export interface CatalogueItem extends TitleFields {
  idx: number;
  olang: string | null;
  released: number | null;
  dev_id: number | null;
  dev: string | null;
  dev_latin: string | null;
  votes: number;
  rating: number | null;
  length: number | null;
  trend: number | null;
  vndb_rank: number;
  /** Rank under the default ranking ("SciRanking"). */
  sci_rank: number;
  search: string;
}

export interface RanksResponse {
  method: string;
  ranks: [id: number, rank: number, score: number | null][];
}

export interface OtherVn extends TitleFields {
  released: number | null;
  votes: number;
  rating: number | null;
  rank: number | null;
}

export interface Analysis {
  n: number;
  hist: number[];
  mean: number | null;
  std: number | null;
  years: [year: number, count: number, mean: number][];
  labels: Record<string, number>;
  sp: { mean: number; hist: number[] } | null;
  bias: number | null;
  notes?: number;
}

export interface VnDetail extends TitleFields {
  idx: number;
  olang: string | null;
  released: number | null;
  dev_id: number | null;
  image: number | null;
  image_sexual: number | null;
  length: number | null;
  votes: number;
  rating: number | null;
  average: number | null;
  trend: number | null;
  ranks: Record<string, [number, number | null]>;
  analysis: Analysis;
  history: [day: number, rank: number, vndbRank: number][];
  neighbors: { id: number; wins: number; losses: number; common: number }[];
  relations: { id: number; relation: string }[];
  similar: { id: number; sim: number; common: number; wins: number; losses: number }[];
}

/** Exact head-to-head and 10x10 joint distributions of two VNs (by idx). */
export interface Joint {
  a: number;
  b: number;
  wins: number;
  losses: number;
  common: number;
  raw: number[][]; // [A's vote bucket 1..10][B's]
  sp: number[][]; // [A's sample-percentile decile][B's]
}

export interface Note {
  uid: number;
  name: string;
  vote: number;
  date: number;
  text: string;
  hasPage: boolean;
  labels: number; // list labels bit mask (bit k-1 = label k)
  nvotes: number; // the author's votes on any VN
  sp: number | null; // 0-100: where the vote sits within the author's list
}

export interface NotesPage {
  total: number;
  matched: number;
  page: number;
  pageSize: number;
  notes: Note[];
}

export interface UserNote {
  idx: number;
  vote: number;
  date: number;
  text: string;
  labels: number;
  sp: number | null;
}

/** [uid, name, value, ranked votes, has user page] */
export type LeaderEntry = [number, string, number, number, boolean];

export interface UserData {
  uid: number;
  name: string;
  votes: [idx: number, vote: number][];
  /** higher / equal / lower: common titles this user voted higher than, equal to, lower than the other. */
  similar: { uid: number; name: string; sim: number; common: number; higher: number; equal: number; lower: number; votes: number }[];
  recs: { idx: number; pred: number; support: number }[];
}

export interface Stats {
  ulist_rows: number;
  votes: { count: number; mean: number; std: number; histogram: number[] };
  labels: Record<string, number>;
  years: { year: number; count: number; mean: number; std: number }[];
  ranked_votes: number;
  ranked_users: number;
  ranked_vns: number;
  pairs: number;
  user_pages?: number;
  notes?: number;
  tables?: Record<string, number>;
}

export interface Meta {
  snapshot: string;
  info: {
    dump_date: string;
    day: number;
    generated_at: string;
    pipeline_version: string;
    config: { min_vote: number; min_common_vote: number; neighbors_per_category: number; skip_rankit: boolean; min_user_votes?: number; skip_notes?: boolean };
    methods: string[];
    featured: string[];
    default_method: string;
    history_methods?: string[];
  };
  stats: Stats;
  kendall: { methods: string[]; matrix: number[][] };
  leaderboards?: {
    min_votes: number;
    most_votes: LeaderEntry[];
    most_votes_year: LeaderEntry[];
    highest_mean: LeaderEntry[];
    lowest_mean: LeaderEntry[];
    most_mainstream: LeaderEntry[];
    most_contrarian: LeaderEntry[];
  };
}

// ---------------------------------------------------------------------------
// Fetching with an in-memory cache so navigating back and forth doesn't refetch.

const cache = new Map<string, Promise<unknown>>();

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Bump when a response format changes: it is part of every API URL, so
 *  browser and edge caches never hand old-format responses to new code. */
export const API_VERSION = 4;

const versioned = (path: string) => `${path}${path.includes("?") ? "&" : "?"}v=${API_VERSION}`;

export function fetchJson<T>(path: string): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined;
  if (!p) {
    p = fetch(versioned(path)).then(async (r) => {
      if (!r.ok) {
        const body = (await r.json().catch(() => ({}))) as { error?: string };
        throw new ApiError(r.status, body.error ?? r.statusText);
      }
      return r.json() as Promise<T>;
    });
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return p;
}

/** Binary responses (no in-memory JSON parsing); cached like fetchJson. */
const binCache = new Map<string, Promise<ArrayBuffer>>();
export function fetchBinary(path: string): Promise<ArrayBuffer> {
  let p = binCache.get(path);
  if (!p) {
    p = fetch(versioned(path)).then((r) => {
      if (!r.ok) throw new ApiError(r.status, r.statusText);
      return r.arrayBuffer();
    });
    p.catch(() => binCache.delete(path));
    binCache.set(path, p);
  }
  return p;
}

export type Loadable<T> =
  | { state: "loading"; data?: undefined; error?: undefined }
  | { state: "ok"; data: T; error?: undefined }
  | { state: "error"; data?: undefined; error: ApiError | Error };

export function useApi<T>(path: string | null): Loadable<T> {
  const [res, setRes] = useState<{ path: string | null; value: Loadable<T> }>({ path, value: { state: "loading" } });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    fetchJson<T>(path).then(
      (data) => alive && setRes({ path, value: { state: "ok", data } }),
      (error: Error) => alive && setRes({ path, value: { state: "error", error } }),
    );
    return () => {
      alive = false;
    };
  }, [path]);
  return res.path === path ? res.value : { state: "loading" };
}

/** Combine several loadables: ok only when all are ok. */
export function all<T extends unknown[]>(...xs: { [K in keyof T]: Loadable<T[K]> }): Loadable<T> {
  const err = xs.find((x) => x.state === "error");
  if (err) return err as Loadable<T>;
  if (xs.some((x) => x.state === "loading")) return { state: "loading" };
  return { state: "ok", data: xs.map((x) => x.data) as T };
}

// ---------------------------------------------------------------------------
// Shared hooks

export const useMeta = () => useApi<Meta>("/api/meta");

export interface Catalogue {
  items: CatalogueItem[];
  byId: Map<number, CatalogueItem>;
  byIdx: CatalogueItem[];
}

let catalogueMemo: { src: unknown; value: Catalogue } | null = null;

export function useCatalogue(): Loadable<Catalogue> {
  const raw = useApi<{ columns?: string[]; rows?: unknown[][]; items?: CatalogueItem[] }>("/api/catalogue");
  return useMemo<Loadable<Catalogue>>(() => {
    if (raw.state !== "ok") return raw;
    // Shared across components: the catalogue is ~8k rows, build the objects once.
    if (catalogueMemo?.src === raw.data) return { state: "ok", data: catalogueMemo.value };
    const { columns, rows, items: legacy } = raw.data;
    if (!legacy && !(columns && rows)) return { state: "error", error: new ApiError(500, "unexpected catalogue format") };
    // `items` is the pre-v3 format, still possible from an old cache entry.
    const items = legacy ?? rows!.map((r) => Object.fromEntries(columns!.map((c, i) => [c, r[i]])) as unknown as CatalogueItem);
    const byIdx: CatalogueItem[] = [];
    for (const it of items) byIdx[it.idx] = it;
    const value = { items, byId: new Map(items.map((i) => [i.id, i])), byIdx };
    catalogueMemo = { src: raw.data, value };
    return { state: "ok", data: value };
  }, [raw]);
}

export type RankMap = Map<number, { rank: number; score: number | null }>;

export function useRanks(method: string | null): Loadable<RankMap> {
  const raw = useApi<RanksResponse>(method ? `/api/ranks?m=${encodeURIComponent(method)}` : null);
  return useMemo<Loadable<RankMap>>(() => {
    if (raw.state !== "ok") return raw;
    return { state: "ok", data: new Map((raw.data.ranks ?? []).map(([id, rank, score]) => [id, { rank, score }])) };
  }, [raw]);
}

/** Ranks for several methods at once (each cached separately). */
export function useManyRanks(methods: string[]): Loadable<Record<string, RankMap>> {
  const key = methods.join(",");
  const [state, setState] = useState<{ key: string; value: Loadable<Record<string, RankMap>> }>({ key, value: { state: "loading" } });
  useEffect(() => {
    let alive = true;
    Promise.all(methods.map((m) => fetchJson<RanksResponse>(`/api/ranks?m=${encodeURIComponent(m)}`))).then(
      (rs) => {
        if (!alive) return;
        const out: Record<string, RankMap> = {};
        rs.forEach((r) => (out[r.method] = new Map((r.ranks ?? []).map(([id, rank, score]) => [id, { rank, score }]))));
        setState({ key, value: { state: "ok", data: out } });
      },
      (error: Error) => alive && setState({ key, value: { state: "error", error } }),
    );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state.key === key ? state.value : { state: "loading" };
}

type RawVn = Omit<VnDetail, "neighbors" | "relations" | "similar"> & {
  neighbors: [number, number, number, number][];
  relations: [number, string][];
  similar: [number, number, number][];
};

/** The worker passes the JSON columns through verbatim (compact arrays); expand them here. */
export function useVn(id: number): Loadable<VnDetail> {
  const raw = useApi<RawVn>(`/api/vn/${id}`);
  return useMemo<Loadable<VnDetail>>(() => {
    if (raw.state !== "ok") return raw;
    const v = raw.data;
    // Arrays from the v3 worker; objects (pre-v3) are passed through.
    const expand = <T,>(list: unknown, f: (row: never[]) => T): T[] =>
      Array.isArray(list) ? list.map((row) => (Array.isArray(row) ? f(row as never[]) : (row as T))) : [];
    return {
      state: "ok",
      data: {
        ...v,
        analysis: { ...v.analysis, years: v.analysis?.years ?? [], labels: v.analysis?.labels ?? {} },
        history: v.history ?? [],
        neighbors: expand(v.neighbors, ([nid, wins, losses, common]) => ({ id: nid, wins, losses, common })),
        relations: expand(v.relations, ([rid, relation]) => ({ id: rid, relation })),
        similar: expand(v.similar, ([sid, sim, common, wins, losses]) => ({ id: sid, sim, common, wins: wins ?? 0, losses: losses ?? 0 })),
      },
    };
  }, [raw]);
}
export const useUser = (uid: number) => useApi<UserData>(`/api/user/${uid}`);

/** One VN's voters with the attributes the ratings filters need (see /api/voters). */
export interface Voters {
  n: number;
  vote: Uint8Array; // 10..100
  sp: Uint8Array; // sample percentile x 200
  labels: Uint8Array;
  year: Uint16Array; // 0 = unknown
  nvotes: Uint16Array;
  umean: Uint8Array; // 10..100
  cv: Float32Array; // r vs VNDB rating, NaN = n/a
  cs: Float32Array; // r vs default ranking
}

export function useVoters(idx: number | null): Loadable<Voters> {
  const [res, setRes] = useState<{ idx: number | null; value: Loadable<Voters> }>({ idx, value: { state: "loading" } });
  useEffect(() => {
    if (idx === null) return;
    let alive = true;
    fetchBinary(`/api/voters/${idx}`).then(
      (buf) => alive && setRes({ idx, value: { state: "ok", data: parseVoters(buf) } }),
      (error: Error) => alive && setRes({ idx, value: { state: "error", error } }),
    );
    return () => {
      alive = false;
    };
  }, [idx]);
  return res.idx === idx ? res.value : { state: "loading" };
}

const VOTER_BYTES = 9;
function parseVoters(buf: ArrayBuffer): Voters {
  const b = new Uint8Array(buf);
  const dv = new DataView(buf);
  const n = Math.floor(b.length / VOTER_BYTES);
  const v: Voters = {
    n,
    vote: new Uint8Array(n),
    sp: new Uint8Array(n),
    labels: new Uint8Array(n),
    year: new Uint16Array(n),
    nvotes: new Uint16Array(n),
    umean: new Uint8Array(n),
    cv: new Float32Array(n),
    cs: new Float32Array(n),
  };
  const r = (x: number) => (x === -128 ? NaN : x / 100);
  for (let i = 0, o = 0; i < n; i++, o += VOTER_BYTES) {
    v.vote[i] = b[o];
    v.sp[i] = b[o + 1];
    v.labels[i] = b[o + 2];
    v.year[i] = b[o + 3] ? 1990 + b[o + 3] : 0;
    v.nvotes[i] = dv.getUint16(o + 4, true);
    v.umean[i] = b[o + 6];
    v.cv[i] = r(dv.getInt8(o + 7));
    v.cs[i] = r(dv.getInt8(o + 8));
  }
  return v;
}

export interface NotesQuery {
  sort?: "date" | "vote" | "sp";
  dir?: "desc" | "asc";
  st?: number; // label mask: any of
  minv?: number | null;
  maxv?: number | null;
}

export function notesPath(idx: number, page: number, q: NotesQuery = {}): string {
  const p = new URLSearchParams({ page: String(page) });
  if (q.sort && q.sort !== "date") p.set("sort", q.sort);
  if (q.dir === "asc") p.set("dir", "asc");
  if (q.st) p.set("st", String(q.st));
  if (q.minv) p.set("minv", String(q.minv));
  if (q.maxv) p.set("maxv", String(q.maxv));
  return `/api/notes/vn/${idx}?${p}`;
}
/** a, b are catalogue idx values (not VNDB ids). */
export const useJoint = (a: number | null | undefined, b: number | null | undefined) =>
  useApi<Joint>(a != null && b != null && a !== b ? `/api/joint/${a}/${b}` : null);
export const useVnNotes = (idx: number | null | undefined, page: number, q?: NotesQuery) => useApi<NotesPage>(idx != null ? notesPath(idx, page, q) : null);
export const useUserNotes = (uid: number) => useApi<{ uid: number; notes: UserNote[] }>(`/api/user/${uid}/notes`);

/** Start the requests every page needs as soon as the app loads. */
export function prefetchCore() {
  void fetchJson("/api/meta").catch(() => {});
  void fetchJson("/api/catalogue").catch(() => {});
}
