import { useMemo } from "react";
import { useCatalogue, useMeta, useRanks, type CatalogueItem, type Loadable, type RankMap } from "../../lib/api";
import { year } from "../../lib/format";

export interface DevStats {
  id: number;
  name: string | null;
  latin: string | null;
  vns: (CatalogueItem & { rank: number })[]; // sorted by rank
  count: number;
  best: number;
  median: number;
  meanRating: number | null;
  votes: number;
  first: number | null;
  last: number | null;
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function buildDevStats(items: CatalogueItem[], ranks: RankMap): Map<number, DevStats> {
  const by = new Map<number, (CatalogueItem & { rank: number })[]>();
  for (const it of items) {
    if (!it.dev_id) continue;
    const list = by.get(it.dev_id) ?? [];
    list.push({ ...it, rank: ranks.get(it.id)?.rank ?? Number.MAX_SAFE_INTEGER });
    by.set(it.dev_id, list);
  }
  const out = new Map<number, DevStats>();
  for (const [id, vns] of by) {
    vns.sort((a, b) => a.rank - b.rank);
    const rated = vns.filter((v) => v.rating !== null);
    const years = vns.map((v) => year(v.released)).filter((y): y is number => y !== null);
    out.set(id, {
      id,
      name: vns[0].dev,
      latin: vns[0].dev_latin,
      vns,
      count: vns.length,
      best: vns[0].rank,
      median: median(vns.map((v) => v.rank)),
      meanRating: rated.length ? rated.reduce((s, v) => s + v.rating!, 0) / rated.length : null,
      votes: vns.reduce((s, v) => s + v.votes, 0),
      first: years.length ? Math.min(...years) : null,
      last: years.length ? Math.max(...years) : null,
    });
  }
  return out;
}

/** Developer aggregates under the default ranking method. */
export function useDevStats(): Loadable<{ devs: Map<number, DevStats>; method: string; total: number }> {
  const meta = useMeta();
  const cat = useCatalogue();
  const method = meta.state === "ok" ? meta.data.info.default_method : null;
  const ranks = useRanks(method);
  return useMemo(() => {
    if (meta.state === "error") return meta;
    if (cat.state === "error") return cat;
    if (ranks.state === "error") return ranks;
    if (cat.state !== "ok" || ranks.state !== "ok" || !method) return { state: "loading" };
    return { state: "ok", data: { devs: buildDevStats(cat.data.items, ranks.data), method, total: cat.data.items.length } };
  }, [meta, cat, ranks, method]);
}

export type DevKey = "count" | "best" | "median" | "meanRating" | "votes";
/** Sort direction where a smaller value is better. */
export const LOWER_IS_BETTER: Record<DevKey, boolean> = { count: false, best: true, median: true, meanRating: false, votes: false };

/** Share of developers (with >= minCount VNs) that this one beats on a metric. */
export function percentileAmong(devs: DevStats[], d: DevStats, key: DevKey): number {
  const v = d[key];
  if (v === null) return 0;
  const others = devs.filter((o) => o[key] !== null);
  const beaten = others.filter((o) => (LOWER_IS_BETTER[key] ? (o[key] as number) > v : (o[key] as number) < v)).length;
  return beaten / Math.max(1, others.length - 1);
}
