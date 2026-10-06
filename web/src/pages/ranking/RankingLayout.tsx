import { useDeferredValue, useMemo, useState } from "react";
import { Outlet, useOutletContext, useSearchParams } from "react-router";
import { useMeta, type Meta } from "../../lib/api";
import { langName, methodDesc, useI18n } from "../../lib/i18n";
import { MethodSelect } from "../../components/MethodSelect";
import { Status } from "../../components/Status";
import { Tabs } from "../../components/Tabs";
import { normalizeQuery, devName } from "../../lib/format";
import { ADVANCED_KEYS, FILTER_KEYS, applyFilters, readFilters, useRows, type Filters, type Row } from "./data";

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
    [rows, deferredQ, params],
  );
  const [advanced, setAdvanced] = useState(() => ADVANCED_KEYS.some((k) => params.get(k)));
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

  const input = "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3";
  const hasFilters = FILTER_KEYS.some((k) => params.get(k));
  const advancedCount = ADVANCED_KEYS.filter((k) => params.get(k)).length;

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

      <section className="space-y-2">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-[2fr_1.4fr_1.3fr_auto]">
          <input type="search" value={filters.q} onChange={(e) => update({ q: e.target.value })} placeholder={t("rank.search")} aria-label={t("rank.search")} className={input} />
          {rows.state === "ok" ? <DevFilter rows={rows.data} value={filters.dev} onChange={(d) => update({ dev: d ? String(d) : null })} /> : <div />}
          <YearRange from={filters.from} to={filters.to} update={update} input={input} />
          <button
            type="button"
            onClick={() => setAdvanced(!advanced)}
            aria-expanded={advanced}
            className={`rounded-md border px-3 py-2 text-sm ${advancedCount ? "border-accent text-ink" : "border-line text-ink-2"} bg-surface hover:bg-surface-2`}
          >
            {t("rank.advanced")}
            {advancedCount ? ` (${advancedCount})` : ""} {advanced ? "▴" : "▾"}
          </button>
        </div>
        {advanced && (
          <div className="grid grid-cols-2 gap-2 rounded-lg border border-line bg-surface p-3 sm:grid-cols-4">
            <Field label={t("rank.olang")}>
              <select value={filters.olang} onChange={(e) => update({ lang: e.target.value })} className={input}>
                <option value="">{t("rank.all")}</option>
                {languages.map((l) => (
                  <option key={l} value={l}>
                    {langName(l, lang)}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t("rank.col.length")}>
              <select value={filters.length ?? ""} onChange={(e) => update({ len: e.target.value })} className={input}>
                <option value="">{t("rank.all")}</option>
                {[1, 2, 3, 4, 5].map((l) => (
                  <option key={l} value={l}>
                    {t(`length.${l}` as never)}
                  </option>
                ))}
              </select>
            </Field>
            <Range label={t("rank.col.votes")} lo={["minv", filters.minVotes || null]} hi={["maxv", filters.maxVotes]} update={update} input={input} step={10} />
            <Range label={t("rank.col.rating")} lo={["rmin", filters.minRating]} hi={["rmax", filters.maxRating]} update={update} input={input} step={0.1} />
          </div>
        )}
      </section>

      <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span className="tabular">{t("rank.results", { n: filtered.length.toLocaleString() })}</span>
        {hasFilters && (
          <button type="button" onClick={() => update(Object.fromEntries(FILTER_KEYS.map((k) => [k, null])))} className="text-accent-ink underline-offset-2 hover:underline">
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

/** Release-year range without a field label (it sits in the basic filter row). */
function YearRange({ from, to, update, input }: { from: number | null; to: number | null; update: (patch: Record<string, string | null>) => void; input: string }) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-1" role="group" aria-label={t("rank.col.year")}>
      <input type="number" inputMode="numeric" step={1} value={from ?? ""} onChange={(e) => update({ from: e.target.value })} placeholder={t("rank.yearFrom")} aria-label={t("rank.yearFrom")} className={input} />
      <span className="text-ink-3">–</span>
      <input type="number" inputMode="numeric" step={1} value={to ?? ""} onChange={(e) => update({ to: e.target.value })} placeholder={t("rank.yearTo")} aria-label={t("rank.yearTo")} className={input} />
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function Range({
  label,
  lo,
  hi,
  update,
  input,
  step,
}: {
  label: string;
  lo: [string, number | null];
  hi: [string, number | null];
  update: (patch: Record<string, string | null>) => void;
  input: string;
  step: number;
}) {
  return (
    <Field label={label}>
      <div className="flex items-center gap-1">
        <input type="number" inputMode="decimal" step={step} value={lo[1] ?? ""} onChange={(e) => update({ [lo[0]]: e.target.value })} placeholder="min" aria-label={`${label} min`} className={input} />
        <span className="text-ink-3">–</span>
        <input type="number" inputMode="decimal" step={step} value={hi[1] ?? ""} onChange={(e) => update({ [hi[0]]: e.target.value })} placeholder="max" aria-label={`${label} max`} className={input} />
      </div>
    </Field>
  );
}

/** Developer filter: type to search developers that have ranked titles. */
function DevFilter({ rows, value, onChange }: { rows: Row[]; value: number | null; onChange: (dev: number | null) => void }) {
  const { t, lang } = useI18n();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const devs = useMemo(() => {
    const m = new Map<number, { id: number; name: string; key: string; n: number }>();
    for (const r of rows) {
      if (!r.dev_id) continue;
      const e = m.get(r.dev_id) ?? { id: r.dev_id, name: devName(r, lang) ?? `p${r.dev_id}`, key: normalizeQuery(`${r.dev ?? ""}${r.dev_latin ?? ""}`), n: 0 };
      e.n++;
      m.set(r.dev_id, e);
    }
    return [...m.values()].sort((a, b) => b.n - a.n);
  }, [rows, lang]);
  const current = value ? devs.find((d) => d.id === value) : null;
  const nq = normalizeQuery(q);
  const hits = open ? devs.filter((d) => !nq || d.key.includes(nq)).slice(0, 10) : [];
  if (current)
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-accent bg-surface px-3 py-2 text-sm">
        <span className="truncate">
          <span className="text-xs text-ink-3">{t("rank.developer")}: </span>
          {current.name}
        </span>
        <button type="button" onClick={() => onChange(null)} aria-label={t("rank.clear")} className="text-ink-3 hover:text-ink">
          ×
        </button>
      </div>
    );
  return (
    <div className="relative">
      <input
        type="search"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder={t("rank.devSearch")}
        aria-label={t("rank.devSearch")}
        className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3"
      />
      {hits.length > 0 && (
        <ul className="absolute z-30 mt-1 max-h-80 w-full overflow-y-auto rounded-md border border-line bg-surface shadow-lg">
          {hits.map((d) => (
            <li key={d.id}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(d.id);
                  setQ("");
                  setOpen(false);
                }}
                className="flex w-full justify-between gap-2 px-3 py-2 text-left text-sm hover:bg-surface-2"
              >
                <span className="truncate">{d.name}</span>
                <span className="tabular text-xs text-ink-3">{d.n}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
