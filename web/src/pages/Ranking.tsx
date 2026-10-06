import { useDeferredValue, useMemo } from "react";
import { Link, useSearchParams } from "react-router";
import { useApi, useMeta, type RankItem, type RankingResponse } from "../lib/api";
import { formatInt, formatScore, normalizeQuery, titles, year } from "../lib/format";
import { langName, methodDesc, useI18n } from "../lib/i18n";
import { MethodSelect } from "../components/MethodSelect";
import { Status } from "../components/Status";

const PAGE_SIZE = 50;
type SortKey = "rank" | "vndb_rank" | "votes" | "released";

export default function Ranking() {
  const meta = useMeta();
  const [params, setParams] = useSearchParams();
  const method = params.get("m") ?? (meta.state === "ok" ? meta.data.info.default_method : null);
  const ranking = useApi<RankingResponse>(method ? `/api/ranking?m=${encodeURIComponent(method)}` : null);

  if (meta.state !== "ok") return <Status value={meta} />;
  const info = meta.data.info;

  const update = (patch: Record<string, string | null>, resetPage = true) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    }
    if (resetPage) next.delete("page");
    setParams(next, { replace: true });
  };

  return (
    <div className="space-y-5">
      <Header method={method!} methods={info.methods} featured={info.featured} onMethod={(m) => update({ m: m === info.default_method ? null : m })} />
      {ranking.state === "ok" ? (
        <RankTable items={ranking.data.items} params={params} update={update} />
      ) : (
        <Status value={ranking} />
      )}
    </div>
  );
}

function Header({ method, methods, featured, onMethod }: { method: string; methods: string[]; featured: string[]; onMethod: (m: string) => void }) {
  const { t, lang } = useI18n();
  return (
    <section className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div className="max-w-xl">
        <h1 className="text-2xl font-semibold tracking-tight">{t("nav.ranking")}</h1>
        <p className="mt-1 text-sm text-ink-2">{methodDesc(method, lang) ?? t("site.tagline")}</p>
      </div>
      <label className="block w-full md:w-80">
        <span className="mb-1 block text-xs font-medium text-ink-3">{t("rank.method")}</span>
        <MethodSelect methods={methods} featured={featured} value={method} onChange={onMethod} />
      </label>
    </section>
  );
}

function RankTable({
  items,
  params,
  update,
}: {
  items: RankItem[];
  params: URLSearchParams;
  update: (patch: Record<string, string | null>, resetPage?: boolean) => void;
}) {
  const { t, lang } = useI18n();
  const q = params.get("q") ?? "";
  const deferredQ = useDeferredValue(q);
  const olang = params.get("lang") ?? "";
  const from = Number(params.get("from")) || null;
  const to = Number(params.get("to")) || null;
  const minVotes = Number(params.get("minv")) || 0;
  const dev = Number(params.get("dev")) || null;
  const sort = (params.get("sort") as SortKey) || "rank";
  const page = Math.max(1, Number(params.get("page")) || 1);

  const languages = useMemo(() => {
    const counts = new Map<string, number>();
    for (const it of items) if (it.olang) counts.set(it.olang, (counts.get(it.olang) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([l]) => l);
  }, [items]);

  const devName = useMemo(() => {
    const it = dev ? items.find((i) => i.dev_id === dev) : null;
    return it ? (lang === "zh" ? it.dev : it.dev_latin ?? it.dev) : null;
  }, [items, dev, lang]);

  const filtered = useMemo(() => {
    const nq = normalizeQuery(deferredQ);
    const idq = /^v?(\d+)$/i.exec(deferredQ.trim());
    let out = items.filter((it) => {
      if (idq && it.id === Number(idq[1])) return true;
      if (nq && !it.search.includes(nq)) return false;
      if (olang && it.olang !== olang) return false;
      const y = year(it.released);
      if (from && (!y || y < from)) return false;
      if (to && (!y || y > to)) return false;
      if (minVotes && it.votes < minVotes) return false;
      if (dev && it.dev_id !== dev) return false;
      return true;
    });
    if (sort !== "rank") {
      const key: Record<Exclude<SortKey, "rank">, (i: RankItem) => number> = {
        vndb_rank: (i) => i.vndb_rank,
        votes: (i) => -i.votes,
        released: (i) => -(i.released && i.released < 99999999 ? i.released : 0),
      };
      out = [...out].sort((a, b) => key[sort](a) - key[sort](b) || a.rank - b.rank);
    }
    return out;
  }, [items, deferredQ, olang, from, to, minVotes, dev, sort]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const cur = Math.min(page, pages);
  const rows = filtered.slice((cur - 1) * PAGE_SIZE, cur * PAGE_SIZE);
  const hasFilters = q || olang || from || to || minVotes || dev;

  const input = "rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3";

  const SortHeader = ({ k, children, className = "" }: { k: SortKey; children: React.ReactNode; className?: string }) => (
    <th scope="col" className={`px-3 py-2 font-medium ${className}`} aria-sort={sort === k ? "ascending" : undefined}>
      <button type="button" onClick={() => update({ sort: k === "rank" ? null : k })} className={`inline-flex items-center gap-1 hover:text-ink ${sort === k ? "text-ink" : ""}`}>
        {children}
        {sort === k && <span aria-hidden>↓</span>}
      </button>
    </th>
  );

  return (
    <>
      <section className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <input
          type="search"
          value={q}
          onChange={(e) => update({ q: e.target.value })}
          placeholder={t("rank.search")}
          aria-label={t("rank.search")}
          className={`${input} col-span-2 sm:col-span-3 lg:col-span-2`}
        />
        <select value={olang} onChange={(e) => update({ lang: e.target.value })} aria-label={t("rank.olang")} className={input}>
          <option value="">
            {t("rank.olang")}: {t("rank.all")}
          </option>
          {languages.map((l) => (
            <option key={l} value={l}>
              {langName(l, lang)}
            </option>
          ))}
        </select>
        <input type="number" inputMode="numeric" value={from ?? ""} onChange={(e) => update({ from: e.target.value })} placeholder={t("rank.yearFrom")} aria-label={t("rank.yearFrom")} className={input} />
        <input type="number" inputMode="numeric" value={to ?? ""} onChange={(e) => update({ to: e.target.value })} placeholder={t("rank.yearTo")} aria-label={t("rank.yearTo")} className={input} />
        <input type="number" inputMode="numeric" min={0} step={10} value={minVotes || ""} onChange={(e) => update({ minv: e.target.value })} placeholder={t("rank.minVotes")} aria-label={t("rank.minVotes")} className={input} />
      </section>

      <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span className="tabular">{t("rank.results", { n: filtered.length.toLocaleString() })}</span>
        {dev && (
          <span className="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2.5 py-0.5 text-xs text-ink">
            {t("rank.developer")}: {devName ?? `p${dev}`}
            <button type="button" onClick={() => update({ dev: null })} aria-label={t("rank.clear")} className="ml-0.5 text-ink-2 hover:text-ink">
              ×
            </button>
          </span>
        )}
        {hasFilters && (
          <button type="button" onClick={() => update({ q: null, lang: null, from: null, to: null, minv: null, dev: null })} className="text-accent-ink underline-offset-2 hover:underline">
            {t("rank.clear")}
          </button>
        )}
      </div>

      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-3">
            <tr>
              <SortHeader k="rank" className="w-12 text-right sm:w-16">{t("rank.col.rank")}</SortHeader>
              <th scope="col" className="w-16 px-2 py-2 text-right font-medium sm:w-20 sm:px-3" title={t("rank.deltaHint")}>
                {t("rank.col.delta")}
              </th>
              <th scope="col" className="px-3 py-2 font-medium">{t("rank.col.title")}</th>
              <SortHeader k="released" className="hidden w-16 sm:table-cell">{t("rank.col.year")}</SortHeader>
              <SortHeader k="votes" className="hidden w-20 text-right sm:table-cell">{t("rank.col.votes")}</SortHeader>
              <SortHeader k="vndb_rank" className="w-20 text-right">{t("rank.col.rating")}</SortHeader>
              <th scope="col" className="hidden w-24 px-3 py-2 text-right font-medium sm:table-cell">{t("rank.col.score")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((it) => (
              <Row key={it.id} it={it} onDev={(d) => update({ dev: String(d) })} />
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className="p-8 text-center text-sm text-ink-3">{t("rank.empty")}</p>}
      </div>

      {pages > 1 && (
        <nav className="flex items-center justify-center gap-3 text-sm" aria-label="pagination">
          <button type="button" disabled={cur <= 1} onClick={() => update({ page: String(cur - 1) }, false)} className="rounded-md border border-line bg-surface px-3 py-1.5 disabled:opacity-40">
            {t("rank.prev")}
          </button>
          <span className="tabular text-ink-2">{t("rank.page", { p: cur, n: pages })}</span>
          <button type="button" disabled={cur >= pages} onClick={() => update({ page: String(cur + 1) }, false)} className="rounded-md border border-line bg-surface px-3 py-1.5 disabled:opacity-40">
            {t("rank.next")}
          </button>
        </nav>
      )}
    </>
  );
}

function Row({ it, onDev }: { it: RankItem; onDev: (dev: number) => void }) {
  const { lang } = useI18n();
  const { main, sub } = titles(it, lang);
  const delta = it.vndb_rank - it.rank;
  const dev = lang === "zh" ? it.dev : it.dev_latin ?? it.dev;
  return (
    <tr className="border-b border-line last:border-0 hover:bg-surface-2/60">
      <td className="tabular px-3 py-2.5 text-right font-semibold">{it.rank}</td>
      <td className={`tabular px-3 py-2.5 text-right text-xs ${delta > 0 ? "text-up" : delta < 0 ? "text-down" : "text-ink-3"}`}>
        {delta > 0 ? `▲ ${delta}` : delta < 0 ? `▼ ${-delta}` : "–"}
      </td>
      <td className="px-3 py-2.5">
        <Link to={`/vn/${it.id}`} className="font-medium text-ink hover:text-accent-ink">
          {main}
        </Link>
        <div className="mt-0.5 flex flex-wrap gap-x-2 text-xs text-ink-3">
          {sub && <span>{sub}</span>}
          {dev && it.dev_id && (
            <button type="button" onClick={() => onDev(it.dev_id!)} className="hover:text-ink-2 hover:underline">
              {dev}
            </button>
          )}
        </div>
      </td>
      <td className="tabular hidden px-3 py-2.5 text-ink-2 sm:table-cell">{year(it.released) ?? "—"}</td>
      <td className="tabular hidden px-3 py-2.5 text-right text-ink-2 sm:table-cell">{formatInt(it.votes)}</td>
      <td className="tabular px-3 py-2.5 text-right text-ink-2">{it.rating?.toFixed(2) ?? "—"}</td>
      <td className="tabular hidden px-3 py-2.5 text-right text-ink-3 sm:table-cell">{formatScore(it.score)}</td>
    </tr>
  );
}
