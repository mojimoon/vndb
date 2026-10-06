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
}

export interface VnDetail extends TitleFields {
  idx: number;
  olang: string | null;
  released: number | null;
  dev_id: number | null;
  dev: string | null;
  dev_latin: string | null;
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
  similar: { id: number; sim: number; common: number }[];
  others: OtherVn[];
}

export interface PairResult {
  a: number;
  b: number;
  wins: number;
  losses: number;
  common: number;
  below_threshold?: boolean;
}

export interface UserData {
  uid: number;
  name: string;
  votes: [idx: number, vote: number][];
  similar: { uid: number; name: string; sim: number; common: number }[];
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
  tables?: Record<string, number>;
}

export interface Meta {
  snapshot: string;
  info: {
    dump_date: string;
    day: number;
    generated_at: string;
    pipeline_version: string;
    config: { min_vote: number; min_common_vote: number; neighbors_per_category: number; skip_rankit: boolean; min_user_votes?: number };
    methods: string[];
    featured: string[];
    default_method: string;
    history_methods?: string[];
  };
  stats: Stats;
  kendall: { methods: string[]; matrix: number[][] };
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

export function fetchJson<T>(path: string): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined;
  if (!p) {
    p = fetch(path).then(async (r) => {
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

export function useCatalogue(): Loadable<Catalogue> {
  const raw = useApi<{ items: CatalogueItem[] }>("/api/catalogue");
  return useMemo<Loadable<Catalogue>>(() => {
    if (raw.state !== "ok") return raw;
    const items = raw.data.items;
    const byIdx: CatalogueItem[] = [];
    for (const it of items) byIdx[it.idx] = it;
    return { state: "ok", data: { items, byId: new Map(items.map((i) => [i.id, i])), byIdx } };
  }, [raw]);
}

export type RankMap = Map<number, { rank: number; score: number | null }>;

export function useRanks(method: string | null): Loadable<RankMap> {
  const raw = useApi<RanksResponse>(method ? `/api/ranks?m=${encodeURIComponent(method)}` : null);
  return useMemo<Loadable<RankMap>>(() => {
    if (raw.state !== "ok") return raw;
    return { state: "ok", data: new Map(raw.data.ranks.map(([id, rank, score]) => [id, { rank, score }])) };
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
        rs.forEach((r) => (out[r.method] = new Map(r.ranks.map(([id, rank, score]) => [id, { rank, score }]))));
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

export const useVn = (id: number) => useApi<VnDetail>(`/api/vn/${id}`);
export const useUser = (uid: number) => useApi<UserData>(`/api/user/${uid}`);
export const usePair = (a: number | null, b: number | null) => useApi<PairResult>(a && b && a !== b ? `/api/pair/${a}/${b}` : null);
