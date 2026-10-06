import { useMemo, useState } from "react";
import { Link } from "react-router";
import { devName, formatInt, formatScore, titles, year } from "../../lib/format";
import { langName, methodName, methodShort, useI18n, type StringKey } from "../../lib/i18n";
import { MethodSelect } from "../../components/MethodSelect";
import { useRanking } from "./RankingLayout";
import type { Row } from "./data";

const OPTIONAL = ["delta", "trend", "dev", "year", "votes", "rating", "lang", "length", "score"] as const;
type Col = (typeof OPTIONAL)[number];
const DEFAULT_COLS: Col[] = ["delta", "dev", "year", "votes", "rating", "score"];
const PAGE_SIZES = [25, 50, 100, 200];

function loadCols(): Col[] {
  try {
    const v = JSON.parse(localStorage.getItem("rank.cols.v2") ?? "null");
    if (Array.isArray(v)) return v.filter((c) => (OPTIONAL as readonly string[]).includes(c));
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_COLS;
}

const COL_LABEL: Record<Col, StringKey> = {
  delta: "rank.col.delta",
  trend: "rank.col.trend",
  dev: "rank.col.dev",
  year: "rank.col.year",
  votes: "rank.col.votes",
  rating: "rank.col.rating",
  lang: "rank.col.lang",
  length: "rank.col.length",
  score: "rank.col.score",
};

// Sort keys: "rank", "x:<method>", or an optional column. Ranks ascend, the rest descend.
function sortValue(r: Row, key: string): number {
  if (key === "rank") return r.rank;
  if (key.startsWith("x:")) return r.extra[key.slice(2)]?.rank ?? Number.MAX_SAFE_INTEGER;
  switch (key) {
    case "delta":
      return -(r.vndb_rank - r.rank);
    case "trend":
      return -(r.trend ?? -1e9);
    case "year":
      return -(r.released && r.released < 99999999 ? r.released : 0);
    case "votes":
      return -r.votes;
    case "rating":
      return -(r.rating ?? 0);
    case "length":
      return -(r.length ?? 0);
    case "score":
      return r.rank;
    default:
      return r.rank;
  }
}

export default function RankTable() {
  const { t, lang } = useI18n();
  const { meta, method, extras, filtered, params, update } = useRanking();
  const [cols, setColsState] = useState<Col[]>(loadCols);
  const [showCols, setShowCols] = useState(false);
  const setCols = (c: Col[]) => {
    setColsState(c);
    try {
      localStorage.setItem("rank.cols.v2", JSON.stringify(c));
    } catch {
      /* storage unavailable */
    }
  };
  const sort = params.get("sort") ?? "rank";
  const pageSize = PAGE_SIZES.includes(Number(params.get("ps"))) ? Number(params.get("ps")) : 50;
  const page = Math.max(1, Number(params.get("page")) || 1);

  const sorted = useMemo(
    () => (sort === "rank" ? filtered : [...filtered].sort((a, b) => sortValue(a, sort) - sortValue(b, sort) || a.rank - b.rank)),
    [filtered, sort],
  );
  const pages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const cur = Math.min(page, pages);
  const rows = sorted.slice((cur - 1) * pageSize, cur * pageSize);
  const has = (c: Col) => cols.includes(c);

  const setExtras = (xs: string[]) => update({ x: xs.length ? xs.join(",") : null }, false);

  const exportCsv = () => {
    const head = ["id", "title", methodName(method, "en"), ...extras.map((m) => methodName(m, "en")), "VNDB rank", "year", "votes", "VNDB rating", "score"];
    const esc = (v: unknown) => {
      const s = v === null || v === undefined ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [head, ...sorted.map((r) => [`v${r.id}`, titles(r, lang).main, r.rank, ...extras.map((m) => r.extra[m]?.rank), r.vndb_rank, year(r.released), r.votes, r.rating, r.score])];
    const blob = new Blob(["﻿" + lines.map((l) => l.map(esc).join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `vndb-ranking-${method}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const Th = ({ k, children, className = "", title }: { k: string; children: React.ReactNode; className?: string; title?: string }) => (
    <th scope="col" className={`px-3 py-2 font-medium ${className}`} aria-sort={sort === k ? "ascending" : undefined} title={title}>
      <button type="button" onClick={() => update({ sort: k === "rank" ? null : k })} className={`inline-flex items-center gap-1 hover:text-ink ${sort === k ? "text-ink" : ""}`}>
        {children}
        {sort === k && <span aria-hidden>↓</span>}
      </button>
    </th>
  );

  return (
    <div className="space-y-3">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-xs text-ink-3">{t("rank.compareWith")}:</span>
        {extras.map((m) => (
          <span key={m} className="inline-flex items-center gap-1 rounded-full border border-line bg-surface px-2.5 py-0.5 text-xs">
            {methodName(m, lang)}
            <button type="button" aria-label="remove" onClick={() => setExtras(extras.filter((x) => x !== m))} className="text-ink-3 hover:text-ink">
              ×
            </button>
          </span>
        ))}
        {extras.length < 3 && (
          <div className="w-52">
            <MethodSelect
              methods={meta.info.methods.filter((m) => m !== method && !extras.includes(m))}
              featured={meta.info.featured.filter((m) => m !== method && !extras.includes(m))}
              value=""
              placeholder={t("rank.addMethod")}
              onChange={(m) => m && setExtras([...extras, m])}
              compact
            />
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <div className="relative">
            <button type="button" onClick={() => setShowCols(!showCols)} aria-expanded={showCols} className="rounded-md border border-line bg-surface px-3 py-1.5 text-xs hover:bg-surface-2">
              {t("rank.columns")} ▾
            </button>
            {showCols && (
              <div className="absolute right-0 z-20 mt-1 w-44 rounded-md border border-line bg-surface p-2 shadow-lg">
                {OPTIONAL.map((c) => (
                  <label key={c} className="flex items-center gap-2 rounded px-2 py-1 text-sm hover:bg-surface-2">
                    <input type="checkbox" checked={has(c)} onChange={() => setCols(has(c) ? cols.filter((x) => x !== c) : OPTIONAL.filter((x) => x === c || has(x)))} />
                    {t(COL_LABEL[c])}
                  </label>
                ))}
              </div>
            )}
          </div>
          <label className="flex items-center gap-1 text-xs text-ink-3">
            {t("rank.pageSize")}
            <select value={pageSize} onChange={(e) => update({ ps: e.target.value === "50" ? null : e.target.value })} className="rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink">
              {PAGE_SIZES.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <button type="button" onClick={exportCsv} className="rounded-md border border-line bg-surface px-3 py-1.5 text-xs hover:bg-surface-2">
            {t("rank.export")}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-3">
            <tr>
              <Th k="rank" className="w-12 text-right sm:w-16" title={methodName(method, lang)}>
                {t("rank.col.rank")}
              </Th>
              {extras.map((m) => (
                <Th key={m} k={`x:${m}`} className="w-16 text-right" title={methodName(m, lang)}>
                  <span className="max-w-24 truncate">{methodShort(m, lang)}</span>
                </Th>
              ))}
              {has("delta") && (
                <Th k="delta" className="w-16 text-right" title={t("rank.deltaHint")}>
                  {t("rank.col.delta")}
                </Th>
              )}
              {has("trend") && (
                <Th k="trend" className="hidden w-14 text-right sm:table-cell">
                  {t("rank.col.trend")}
                </Th>
              )}
              <th scope="col" className="px-3 py-2 font-medium">
                {t("rank.col.title")}
              </th>
              {has("dev") && <th className="hidden px-3 py-2 font-medium md:table-cell">{t("rank.col.dev")}</th>}
              {has("year") && <Th k="year" className="hidden w-16 sm:table-cell">{t("rank.col.year")}</Th>}
              {has("lang") && <th className="hidden px-3 py-2 font-medium md:table-cell">{t("rank.col.lang")}</th>}
              {has("length") && <Th k="length" className="hidden md:table-cell">{t("rank.col.length")}</Th>}
              {has("votes") && <Th k="votes" className="hidden w-20 text-right sm:table-cell">{t("rank.col.votes")}</Th>}
              {has("rating") && <Th k="rating" className="w-16 text-right">{t("rank.col.rating")}</Th>}
              {has("score") && <th className="hidden w-24 px-3 py-2 text-right font-medium lg:table-cell">{t("rank.col.score")}</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const { main, sub } = titles(r, lang);
              const delta = r.vndb_rank - r.rank;
              const dev = devName(r, lang);
              return (
                <tr key={r.id} className="border-b border-line last:border-0 hover:bg-surface-2/60">
                  <td className="tabular px-3 py-2.5 text-right font-semibold">{r.rank}</td>
                  {extras.map((m) => {
                    const x = r.extra[m];
                    const diff = x ? r.rank - x.rank : 0; // > 0: this method ranks it higher than the primary one
                    return (
                      <td key={m} className="px-3 py-1.5 text-right" title={x ? `${methodName(m, lang)}: #${x.rank} · ${formatScore(x.score)}` : undefined}>
                        <div className={`tabular font-medium ${diff > 0 ? "text-up" : diff < 0 ? "text-down" : "text-ink-2"}`}>{x?.rank ?? "—"}</div>
                        <div className="tabular text-[11px] text-ink-3">{x ? formatScore(x.score) : ""}</div>
                      </td>
                    );
                  })}
                  {has("delta") && <td className="px-3 py-2.5 text-right text-xs"><Delta d={delta} /></td>}
                  {has("trend") && <td className="hidden px-3 py-2.5 text-right text-xs sm:table-cell">{r.trend === null ? <span className="text-ink-3">–</span> : <Delta d={r.trend} />}</td>}
                  <td className="px-3 py-2.5">
                    <Link to={`/vn/${r.id}`} className="font-medium text-ink hover:text-accent-ink">
                      {main}
                    </Link>
                    {sub && <div className="mt-0.5 truncate text-xs text-ink-3">{sub}</div>}
                  </td>
                  {has("dev") && (
                    <td className="hidden max-w-48 px-3 py-2.5 text-ink-2 md:table-cell">
                      {r.dev_id ? (
                        <span className="flex items-baseline gap-1.5">
                          <Link to={`/dev/${r.dev_id}`} className="truncate hover:text-accent-ink hover:underline">
                            {dev}
                          </Link>
                          <button type="button" onClick={() => update({ dev: String(r.dev_id) })} className="shrink-0 text-xs text-ink-3 hover:text-ink" title={t("rank.filterDev")} aria-label={t("rank.filterDev")}>
                            ⧩
                          </button>
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  )}
                  {has("year") && <td className="tabular hidden px-3 py-2.5 text-ink-2 sm:table-cell">{year(r.released) ?? "—"}</td>}
                  {has("lang") && <td className="hidden px-3 py-2.5 text-ink-2 md:table-cell">{r.olang ? langName(r.olang, lang) : "—"}</td>}
                  {has("length") && <td className="hidden px-3 py-2.5 text-ink-2 md:table-cell">{r.length ? t(`length.${r.length}` as StringKey) : "—"}</td>}
                  {has("votes") && <td className="tabular hidden px-3 py-2.5 text-right text-ink-2 sm:table-cell">{formatInt(r.votes)}</td>}
                  {has("rating") && <td className="tabular px-3 py-2.5 text-right text-ink-2">{r.rating?.toFixed(2) ?? "—"}</td>}
                  {has("score") && <td className="tabular hidden px-3 py-2.5 text-right text-ink-3 lg:table-cell">{formatScore(r.score)}</td>}
                </tr>
              );
            })}
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
    </div>
  );
}

export function Delta({ d }: { d: number }) {
  return <span className={`tabular ${d > 0 ? "text-up" : d < 0 ? "text-down" : "text-ink-3"}`}>{d > 0 ? `▲ ${d}` : d < 0 ? `▼ ${-d}` : "–"}</span>;
}
