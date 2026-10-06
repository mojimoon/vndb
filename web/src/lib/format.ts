import type { CatalogueItem, TitleFields } from "./api";
import type { Lang } from "./i18n";

/** Main and secondary title for the current UI language. */
export function titles(v: TitleFields, lang: Lang): { main: string; sub: string | null } {
  const main =
    lang === "zh"
      ? v.title_zh ?? v.title_ja ?? v.title
      : v.title_en ?? v.latin ?? v.title;
  const sub = v.title !== main ? v.title : v.latin && v.latin !== main ? v.latin : null;
  return { main, sub };
}

/** YYYYMMDD with VNDB's 99 placeholders -> "2019-04-26" / "2019-04" / "2019". */
export function formatDate(d: number | null | undefined, tba = "TBA"): string {
  if (!d) return "—";
  if (d >= 99999999) return tba;
  const y = Math.floor(d / 10000);
  const m = Math.floor(d / 100) % 100;
  const day = d % 100;
  if (m < 1 || m > 12) return String(y);
  if (day < 1 || day > 31) return `${y}-${String(m).padStart(2, "0")}`;
  return `${y}-${String(m).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

export const year = (d: number | null | undefined) => (d && d < 99999999 ? Math.floor(d / 10000) : null);

export function formatScore(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  if (Number.isInteger(x)) return x.toLocaleString();
  const a = Math.abs(x);
  return a >= 100 ? x.toFixed(1) : a >= 1 ? x.toFixed(3) : x.toPrecision(3);
}

export const formatInt = (x: number | null | undefined) => (x === null || x === undefined ? "—" : x.toLocaleString());

/** Same normalization as the pipeline's search key (vndb_rank.extract.normalize_search). */
export function normalizeQuery(q: string): string {
  return q.toLowerCase().replace(/[^0-9a-z぀-ヺー-ヿ㐀-鿿가-힯]+/g, "");
}

export function coverUrl(id: number): string {
  return `https://t.vndb.org/cv/${String(id % 100).padStart(2, "0")}/${id}.jpg`;
}

/** Pipeline day number (days since 2000-01-01) -> "YYYY-MM-DD". */
export function dayToDate(day: number): string {
  return new Date(Date.UTC(2000, 0, 1) + day * 86_400_000).toISOString().slice(0, 10);
}

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export function pearson(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 3) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
}

/** Developer display name for the UI language. */
export function devName(it: Pick<CatalogueItem, "dev" | "dev_latin">, lang: Lang): string | null {
  return lang === "zh" ? it.dev ?? it.dev_latin : it.dev_latin ?? it.dev;
}

/** Sample percentile of each vote within one user's list (same formula as the pipeline). */
export function samplePercentiles(votes: number[]): number[] {
  const order = votes.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(votes.length);
  for (let s = 0; s < order.length; ) {
    let e = s;
    while (e + 1 < order.length && order[e + 1][0] === order[s][0]) e++;
    const avgRank = (s + e) / 2 + 1; // 1-based average rank of the tie group
    for (let k = s; k <= e; k++) out[order[k][1]] = avgRank / (votes.length + 1);
    s = e + 1;
  }
  return out;
}

export const decileLabels = Array.from({ length: 10 }, (_, i) => `${i * 10}%`);
export const scoreLabels = Array.from({ length: 10 }, (_, i) => String(i + 1));
