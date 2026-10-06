import { useMemo, useState } from "react";
import { Link } from "react-router";
import { normalizeQuery } from "../../lib/format";
import { methodName, useI18n } from "../../lib/i18n";
import { useShowMore } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { LOWER_IS_BETTER, useDevStats, type DevKey, type DevStats } from "./data";

export default function DevList() {
  const { t, lang } = useI18n();
  const stats = useDevStats();
  const [minCount, setMinCount] = useState(3);
  const [sort, setSort] = useState<DevKey>("median");
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    if (stats.state !== "ok") return [];
    const nq = normalizeQuery(q);
    const list = [...stats.data.devs.values()].filter(
      (d) => d.count >= minCount && (!nq || normalizeQuery(`${d.name ?? ""}${d.latin ?? ""}`).includes(nq)),
    );
    const dir = LOWER_IS_BETTER[sort] ? 1 : -1;
    return list.sort((a, b) => dir * (((a[sort] ?? Infinity) as number) - ((b[sort] ?? Infinity) as number)) || b.count - a.count);
  }, [stats, minCount, sort, q]);
  // `table` is defined below the early return; the modal only calls it after this render.
  const [visible, more] = useShowMore(rows, 50, t("dev.title"), (items) => table(items));
  if (stats.state !== "ok") return <Status value={stats} />;

  const Th = ({ k, children }: { k: DevKey; children: React.ReactNode }) => (
    <th className="px-3 py-2 text-right font-medium" aria-sort={sort === k ? "ascending" : undefined}>
      <button type="button" onClick={() => setSort(k)} className={`hover:text-ink ${sort === k ? "text-ink" : ""}`}>
        {children}
        {sort === k && " ↓"}
      </button>
    </th>
  );

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">{t("dev.title")}</h1>
        <p className="mt-1 text-sm text-ink-2">{t("dev.hint", { m: methodName(stats.data.method, lang) })}</p>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("dev.search")} aria-label={t("dev.search")} className="w-full max-w-xs rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3" />
        <label className="flex items-center gap-2 text-sm text-ink-2">
          {t("dev.minCount")}
          <select value={minCount} onChange={(e) => setMinCount(Number(e.target.value))} className="rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink">
            {[1, 2, 3, 5, 10].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <span className="tabular text-sm text-ink-3">{t("dev.count", { n: rows.length })}</span>
      </div>
      {table(visible)}
      {more}
    </div>
  );

  function table(items: DevStats[]) {
    return (
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-2">
            <tr>
              <th className="w-10 px-3 py-2 text-right font-medium">#</th>
              <th className="px-3 py-2 font-medium">{t("rank.col.dev")}</th>
              <Th k="count">{t("dev.vns")}</Th>
              <Th k="best">{t("dev.best")}</Th>
              <Th k="median">{t("dev.median")}</Th>
              <Th k="meanRating">{t("dev.meanRating")}</Th>
              <Th k="votes">{t("rank.col.votes")}</Th>
            </tr>
          </thead>
          <tbody>
            {items.map((d: DevStats, i) => (
              <tr key={d.id} className="border-t border-line hover:bg-surface-2/60">
                <td className="tabular px-3 py-2 text-right text-ink-2">{i + 1}</td>
                <td className="px-3 py-2">
                  <Link to={`/dev/${d.id}`} className="font-medium hover:text-accent-ink">
                    {(lang === "zh" ? d.name ?? d.latin : d.latin ?? d.name) ?? `p${d.id}`}
                  </Link>
                  {d.first && <span className="ml-2 text-xs text-ink-3 tabular">{d.first === d.last ? d.first : `${d.first}–${d.last}`}</span>}
                </td>
                <td className="tabular px-3 py-2 text-right">{d.count}</td>
                <td className="tabular px-3 py-2 text-right">{d.best}</td>
                <td className="tabular px-3 py-2 text-right">{Math.round(d.median)}</td>
                <td className="tabular px-3 py-2 text-right">{d.meanRating?.toFixed(2) ?? "—"}</td>
                <td className="tabular px-3 py-2 text-right text-ink-2">{d.votes.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
}
