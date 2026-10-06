import { useMemo } from "react";
import { useCatalogue, useManyRanks, type CatalogueItem, type Loadable } from "../../lib/api";
import { normalizeQuery, year } from "../../lib/format";

export interface Row extends CatalogueItem {
  rank: number;
  score: number | null;
  extra: Record<string, { rank: number; score: number | null } | undefined>;
}

/** URL keys of all filters (cleared together). */
export const FILTER_KEYS = ["q", "lang", "from", "to", "minv", "maxv", "rmin", "rmax", "dev", "len"];
export const ADVANCED_KEYS = ["lang", "from", "to", "minv", "maxv", "rmin", "rmax", "len"];

export interface Filters {
  q: string;
  olang: string;
  from: number | null;
  to: number | null;
  minVotes: number;
  maxVotes: number | null;
  minRating: number | null;
  maxRating: number | null;
  dev: number | null;
  length: number | null;
}

export function readFilters(params: URLSearchParams): Filters {
  return {
    q: params.get("q") ?? "",
    olang: params.get("lang") ?? "",
    from: Number(params.get("from")) || null,
    to: Number(params.get("to")) || null,
    minVotes: Number(params.get("minv")) || 0,
    maxVotes: Number(params.get("maxv")) || null,
    minRating: Number(params.get("rmin")) || null,
    maxRating: Number(params.get("rmax")) || null,
    dev: Number(params.get("dev")) || null,
    length: Number(params.get("len")) || null,
  };
}

/** Catalogue joined with the primary method's ranks and any extra methods' ranks. */
export function useRows(method: string | null, extras: string[]): Loadable<Row[]> {
  const cat = useCatalogue();
  const methods = method ? [method, ...extras.filter((x) => x !== method)] : [];
  const ranks = useManyRanks(methods);
  const extrasKey = extras.join(",");
  return useMemo<Loadable<Row[]>>(() => {
    if (cat.state === "error") return cat;
    if (ranks.state === "error") return ranks;
    if (cat.state !== "ok" || ranks.state !== "ok" || !method || !ranks.data[method]) return { state: "loading" };
    const primary = ranks.data[method];
    const rows: Row[] = cat.data.items.map((it) => {
      const r = primary.get(it.id);
      return {
        ...it,
        rank: r?.rank ?? Number.MAX_SAFE_INTEGER,
        score: r?.score ?? null,
        extra: Object.fromEntries(extras.map((m) => [m, ranks.data[m]?.get(it.id)])),
      };
    });
    rows.sort((a, b) => a.rank - b.rank || a.id - b.id);
    return { state: "ok", data: rows };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cat, ranks, method, extrasKey]);
}

export function applyFilters(rows: Row[], f: Filters): Row[] {
  const nq = normalizeQuery(f.q);
  const idq = /^v?(\d+)$/i.exec(f.q.trim());
  return rows.filter((it) => {
    if (idq && it.id === Number(idq[1])) return true;
    if (nq && !it.search.includes(nq)) return false;
    if (f.olang && it.olang !== f.olang) return false;
    const y = year(it.released);
    if (f.from && (!y || y < f.from)) return false;
    if (f.to && (!y || y > f.to)) return false;
    if (f.minVotes && it.votes < f.minVotes) return false;
    if (f.maxVotes && it.votes > f.maxVotes) return false;
    if (f.minRating && (it.rating ?? 0) < f.minRating) return false;
    if (f.maxRating && (it.rating ?? 0) > f.maxRating) return false;
    if (f.dev && it.dev_id !== f.dev) return false;
    if (f.length && it.length !== f.length) return false;
    return true;
  });
}
