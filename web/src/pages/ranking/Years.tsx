import { useMemo, useState } from "react";
import { year } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useRanking } from "./RankingLayout";
import type { Row } from "./data";

export default function Years() {
  const { t } = useI18n();
  const { filtered } = useRanking();
  const [PER_YEAR, setPerYear] = useState(5);
  const groups = useMemo(() => {
    const by = new Map<number, Row[]>();
    for (const r of filtered) {
      const y = year(r.released);
      if (!y) continue;
      const list = by.get(y) ?? [];
      if (list.length < PER_YEAR) list.push(r);
      by.set(y, list);
    }
    return [...by.entries()].sort((a, b) => b[0] - a[0]);
  }, [filtered, PER_YEAR]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-ink-2">{t("rank.years.hint", { n: PER_YEAR })}</p>
        <label className="flex items-center gap-1 text-xs text-ink-3">
          {t("rank.pageSize")}
          <select value={PER_YEAR} onChange={(e) => setPerYear(Number(e.target.value))} className="rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink">
            {[5, 10, 20].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {groups.map(([y, list]) => (
          <section key={y} className="rounded-lg border border-line bg-surface p-3">
            <h2 className="tabular mb-2 text-lg font-semibold">{y}</h2>
            <ol className="space-y-1.5 text-sm">
              {list.map((r) => (
                <li key={r.id} className="flex items-baseline gap-2">
                  <span className="tabular w-10 shrink-0 text-right text-xs text-ink-3">#{r.rank}</span>
                  <VnLink vn={r} className="truncate" />
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  );
}
