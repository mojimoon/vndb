import { useState } from "react";
import { useMeta } from "../lib/api";
import { formatInt } from "../lib/format";
import { useI18n } from "../lib/i18n";
import { BarChart, LineChart } from "../components/Charts";
import { Status } from "../components/Status";

export default function Stats() {
  const meta = useMeta();
  const { t } = useI18n();
  const [table, setTable] = useState(false);
  if (meta.state !== "ok") return <Status value={meta} />;
  const s = meta.data.stats;

  const tiles: [string, string][] = [
    [t("stats.ranked"), formatInt(s.ranked_vns)],
    [t("stats.pairs"), formatInt(s.pairs)],
    [t("stats.votes"), formatInt(s.votes.count)],
    [t("stats.users"), formatInt(s.ranked_users)],
    [t("stats.mean"), `${s.votes.mean.toFixed(2)} ± ${s.votes.std.toFixed(2)}`],
  ];
  const hist = s.votes.histogram.map((y, i) => ({ x: String(i + 1), y }));
  const years = s.years.filter((y) => y.count > 0);

  return (
    <div className="space-y-8">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("stats.title")}</h1>
          <p className="mt-1 text-sm text-ink-2">{meta.data.info.dump_date}</p>
        </div>
        <div className="flex rounded-md border border-line bg-surface p-0.5 text-sm">
          {[false, true].map((v) => (
            <button key={String(v)} type="button" onClick={() => setTable(v)} aria-pressed={table === v} className={`rounded px-3 py-1 ${table === v ? "bg-surface-2 text-ink" : "text-ink-2"}`}>
              {v ? t("stats.table") : t("stats.chart")}
            </button>
          ))}
        </div>
      </header>

      <section className="grid grid-cols-2 gap-3 md:grid-cols-5">
        {tiles.map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-surface px-4 py-3">
            <div className="text-xs text-ink-3">{k}</div>
            <div className="tabular mt-1 text-xl font-semibold">{v}</div>
          </div>
        ))}
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <Card title={t("stats.dist")}>
          {table ? (
            <SimpleTable head={[t("stats.score"), t("stats.count")]} rows={hist.map((d) => [d.x, formatInt(d.y)])} />
          ) : (
            <BarChart data={hist} label={t("stats.dist")} />
          )}
        </Card>
        <Card title={t("stats.byYear")}>
          {table ? (
            <SimpleTable head={[t("stats.year"), t("stats.count"), t("stats.mean")]} rows={years.map((y) => [String(y.year), formatInt(y.count), y.mean.toFixed(2)])} />
          ) : (
            <BarChart data={years.map((y) => ({ x: String(y.year), y: y.count }))} label={t("stats.byYear")} />
          )}
        </Card>
        {!table && (
          <Card title={t("stats.meanByYear")}>
            <LineChart data={years.map((y) => ({ x: String(y.year), y: y.mean }))} format={(v) => v.toFixed(2)} label={t("stats.meanByYear")} />
          </Card>
        )}
      </section>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <h2 className="mb-2 text-sm font-semibold">{title}</h2>
      {children}
    </div>
  );
}

function SimpleTable({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="max-h-80 overflow-y-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-surface text-left text-xs text-ink-3">
          <tr>
            {head.map((h, i) => (
              <th key={h} className={`py-1.5 font-medium ${i ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]} className="border-t border-line">
              {r.map((c, i) => (
                <td key={i} className={`tabular py-1.5 ${i ? "text-right" : ""}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
