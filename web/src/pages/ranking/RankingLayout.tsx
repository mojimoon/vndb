import { useDeferredValue, useMemo } from "react";
import { Outlet, useOutletContext, useSearchParams } from "react-router";
import { useMeta, type Meta } from "../../lib/api";
import { langName, methodDesc, useI18n } from "../../lib/i18n";
import { MethodSelect } from "../../components/MethodSelect";
import { Status } from "../../components/Status";
import { Tabs } from "../../components/Tabs";
import { applyFilters, readFilters, useRows, type Filters, type Row } from "./data";

export interface RankingContext {
  meta: Meta;
  method: string;
  extras: string[];
  rows: Row[];
  filtered: Row[];
  filters: Filters;
  params: URLSearchParams;
  update: (patch: Record<string, string | null>, resetPage?: boolean) => void;
}

export const useRanking = () => useOutletContext<RankingContext>();

export default function RankingLayout() {
  const { t, lang } = useI18n();
  const meta = useMeta();
  const [params, setParams] = useSearchParams();
  const info = meta.state === "ok" ? meta.data.info : null;
  const method = params.get("m") ?? info?.default_method ?? null;
  const extras = useMemo(() => (params.get("x") ?? "").split(",").filter((m) => m && info?.methods.includes(m) && m !== method).slice(0, 3), [params, info, method]);
  const rows = useRows(info ? method : null, extras);
  const filters = readFilters(params);
  const deferredQ = useDeferredValue(filters.q);
  const filtered = useMemo(
    () => (rows.state === "ok" ? applyFilters(rows.data, { ...filters, q: deferredQ }) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, deferredQ, filters.olang, filters.from, filters.to, filters.minVotes, filters.dev, filters.length],
  );
  const languages = useMemo(() => {
    if (rows.state !== "ok") return [];
    const counts = new Map<string, number>();
    for (const it of rows.data) if (it.olang) counts.set(it.olang, (counts.get(it.olang) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([l]) => l);
  }, [rows]);

  if (meta.state !== "ok") return <Status value={meta} />;

  const update = (patch: Record<string, string | null>, resetPage = true) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    }
    if (resetPage) next.delete("page");
    setParams(next, { replace: true });
  };

  const devName = (() => {
    if (!filters.dev || rows.state !== "ok") return null;
    const it = rows.data.find((i) => i.dev_id === filters.dev);
    return it ? (lang === "zh" ? it.dev : it.dev_latin ?? it.dev) : `p${filters.dev}`;
  })();
  const input = "rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3";
  const hasFilters = filters.q || filters.olang || filters.from || filters.to || filters.minVotes || filters.dev || filters.length;

  return (
    <div className="space-y-5">
      <section className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div className="max-w-xl">
          <h1 className="text-2xl font-semibold tracking-tight">{t("nav.ranking")}</h1>
          <p className="mt-1 text-sm text-ink-2">{methodDesc(method!, lang) ?? t("site.tagline")}</p>
        </div>
        <label className="block w-full md:w-80">
          <span className="mb-1 block text-xs font-medium text-ink-3">{t("rank.method")}</span>
          <MethodSelect
            methods={meta.data.info.methods}
            featured={meta.data.info.featured}
            value={method!}
            onChange={(m) => update({ m: m === meta.data.info.default_method ? null : m }, false)}
          />
        </label>
      </section>

      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        <input type="search" value={filters.q} onChange={(e) => update({ q: e.target.value })} placeholder={t("rank.search")} aria-label={t("rank.search")} className={`${input} col-span-2`} />
        <select value={filters.olang} onChange={(e) => update({ lang: e.target.value })} aria-label={t("rank.olang")} className={input}>
          <option value="">
            {t("rank.olang")}: {t("rank.all")}
          </option>
          {languages.map((l) => (
            <option key={l} value={l}>
              {langName(l, lang)}
            </option>
          ))}
        </select>
        <select value={filters.length ?? ""} onChange={(e) => update({ len: e.target.value })} aria-label={t("rank.col.length")} className={input}>
          <option value="">
            {t("rank.col.length")}: {t("rank.all")}
          </option>
          {[1, 2, 3, 4, 5].map((l) => (
            <option key={l} value={l}>
              {t(`length.${l}` as never)}
            </option>
          ))}
        </select>
        <input type="number" inputMode="numeric" value={filters.from ?? ""} onChange={(e) => update({ from: e.target.value })} placeholder={t("rank.yearFrom")} aria-label={t("rank.yearFrom")} className={input} />
        <input type="number" inputMode="numeric" value={filters.to ?? ""} onChange={(e) => update({ to: e.target.value })} placeholder={t("rank.yearTo")} aria-label={t("rank.yearTo")} className={input} />
        <input type="number" inputMode="numeric" min={0} step={10} value={filters.minVotes || ""} onChange={(e) => update({ minv: e.target.value })} placeholder={t("rank.minVotes")} aria-label={t("rank.minVotes")} className={input} />
      </section>

      <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span className="tabular">{t("rank.results", { n: filtered.length.toLocaleString() })}</span>
        {filters.dev && (
          <span className="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2.5 py-0.5 text-xs text-ink">
            {t("rank.developer")}: {devName}
            <button type="button" onClick={() => update({ dev: null })} aria-label={t("rank.clear")} className="ml-0.5 text-ink-2 hover:text-ink">
              ×
            </button>
          </span>
        )}
        {hasFilters && (
          <button type="button" onClick={() => update({ q: null, lang: null, from: null, to: null, minv: null, dev: null, len: null })} className="text-accent-ink underline-offset-2 hover:underline">
            {t("rank.clear")}
          </button>
        )}
      </div>

      <Tabs
        base=""
        tabs={[
          { path: "", label: t("rank.tab.table") },
          { path: "years", label: t("rank.tab.years") },
          { path: "disputes", label: t("rank.tab.disputes") },
          { path: "movers", label: t("rank.tab.movers") },
        ]}
      />

      {rows.state === "ok" ? (
        <Outlet context={{ meta: meta.data, method: method!, extras, rows: rows.data, filtered, filters, params, update } satisfies RankingContext} />
      ) : (
        <Status value={rows} />
      )}
    </div>
  );
}
