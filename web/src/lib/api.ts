import { useEffect, useState } from "react";

export interface RankItem {
  id: number;
  title: string;
  latin: string | null;
  title_ja: string | null;
  title_zh: string | null;
  title_en: string | null;
  olang: string | null;
  released: number | null;
  dev_id: number | null;
  dev: string | null;
  dev_latin: string | null;
  votes: number;
  rating: number | null;
  rank: number;
  score: number | null;
  vndb_rank: number;
  search: string;
}

export interface RankingResponse {
  snapshot: string;
  method: string;
  items: RankItem[];
}

export type TitleFields = Pick<RankItem, "id" | "title" | "latin" | "title_ja" | "title_zh" | "title_en">;

export interface OtherVn extends TitleFields {
  released: number | null;
  votes: number;
  rating: number | null;
  rank: number | null;
}

export interface VnDetail extends TitleFields {
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
  ranks: Record<string, [number, number | null]>;
  neighbors: { id: number; wins: number; losses: number; common: number }[];
  relations: { id: number; relation: string }[];
  others: OtherVn[];
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
  tables?: Record<string, number>;
}

export interface Meta {
  snapshot: string;
  info: {
    dump_date: string;
    generated_at: string;
    pipeline_version: string;
    config: { min_vote: number; min_common_vote: number; neighbors_per_category: number; skip_rankit: boolean };
    methods: string[];
    featured: string[];
    default_method: string;
  };
  stats: Stats;
  kendall: { methods: string[]; matrix: number[][] };
}

// Tiny in-memory cache so navigating back and forth doesn't refetch.
const cache = new Map<string, Promise<unknown>>();

export class ApiError extends Error {
  constructor(public status: number, message: string) {
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

export const useMeta = () => useApi<Meta>("/api/meta");
