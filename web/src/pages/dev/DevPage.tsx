import { useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { formatInt, normalizeQuery, pct, year } from "../../lib/format";
import { langName, methodName, useI18n } from "../../lib/i18n";
import { BarChart } from "../../components/Charts";
import { useShowMore } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { CompareTable } from "../compare/CompareVn";
import { percentileAmong, useDevStats, type DevKey, type DevStats } from "./data";

const PEER_MIN = 3; // compare against developers with at least this many ranked VNs

export default function DevPage() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const { t, lang } = useI18n();
  const stats = useDevStats();
  if (stats.state !== "ok") return <Status value={stats} />;
  const { devs, method, total } = stats.data;
  const d = devs.get(Number(id));
  if (!d) return <p className="py-16 text-center text-sm text-ink-3">{t("common.notfound")}</p>;
  const vs = devs.get(Number(params.get("vs")));
  const peers = [...devs.values()].filter((x) => x.count >= PEER_MIN);
  const name = (x: DevStats) => (lang === "zh" ? x.name ?? x.latin : x.latin ?? x.name) ?? `p${x.id}`;
  const metrics: [DevKey, string, (v: number) => string][] = [
    ["count", t("dev.vns"), (v) => String(v)],
    ["best", t("dev.best"), (v) => `#${v}`],
    ["median", t("dev.median"), (v) => `#${Math.round(v)}`],
    ["meanRating", t("dev.meanRating"), (v) => v.toFixed(2)],
    ["votes", t("rank.col.votes"), (v) => formatInt(v)],
  ];

  return (
    <article className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs text-ink-3">
            <Link to="/dev" className="hover:text-accent-ink">
              {t("dev.title")}
            </Link>{" "}
            /
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">{name(d)}</h1>
          {d.name && d.latin && <p className="text-ink-2">{lang === "zh" ? d.latin : d.name}</p>}
        </div>
        <div className="flex gap-4 text-sm">
          <a href={`https://vndb.org/p${d.id}`} target="_blank" rel="noreferrer" className="text-accent-ink hover:underline">
            {t("vn.onVndb")} ↗
          </a>
          <Link to={`/?dev=${d.id}`} className="text-accent-ink hover:underline">
            {t("dev.inRanking")} →
          </Link>
        </div>
      </header>

      <section className="grid grid-cols-2 gap-2 md:grid-cols-5">
        {metrics.map(([k, label, fmt]) => (
          <Stat
            key={k}
            label={label}
            value={d[k] === null ? "—" : fmt(d[k] as number)}
            hint={d.count >= PEER_MIN ? t("dev.beats", { p: pct(percentileAmong(peers, d, k)), n: peers.length }) : undefined}
          />
        ))}
      </section>
      <p className="text-xs text-ink-3">{t("dev.statsHint", { m: methodName(method, lang), n: total.toLocaleString() })}</p>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("dev.byYear")}>
          <YearChart d={d} />
        </Card>
        <Card title={t("dev.compareWith")}>
          <DevPicker devs={peers.concat(d.count < PEER_MIN ? [d] : [])} exclude={d.id} onPick={(o) => setParams({ vs: String(o.id) }, { replace: true })} name={name} />
          {vs && (
            <div className="mt-3">
              <CompareTable
                heads={[name(d), name(vs)]}
                rows={metrics.map(([k, label, fmt]) => {
                  const a = d[k];
                  const b = vs[k];
                  const lower = k === "best" || k === "median";
                  const better = a === null || b === null || a === b ? 0 : (a < b) === lower ? -1 : 1;
                  return [label, a === null ? "—" : fmt(a as number), b === null ? "—" : fmt(b as number), better] as const;
                })}
              />
              <Link to={`/dev/${vs.id}?vs=${d.id}`} className="mt-2 inline-block text-xs text-accent-ink hover:underline">
                {name(vs)} →
              </Link>
            </div>
          )}
        </Card>
      </div>

      <VnTable d={d} />
    </article>
  );
}

function YearChart({ d }: { d: DevStats }) {
  const { t } = useI18n();
  const data = useMemo(() => {
    const by = new Map<number, { n: number; rating: number; rated: number }>();
    for (const v of d.vns) {
      const y = year(v.released);
      if (!y) continue;
      const e = by.get(y) ?? { n: 0, rating: 0, rated: 0 };
      e.n++;
      if (v.rating !== null) {
        e.rating += v.rating;
        e.rated++;
      }
      by.set(y, e);
    }
    const ys = [...by.keys()].sort((a, b) => a - b);
    if (!ys.length) return [];
    const out = [];
    for (let y = ys[0]; y <= ys[ys.length - 1]; y++) {
      const e = by.get(y);
      out.push({ year: y, n: e?.n ?? 0, rating: e && e.rated ? e.rating / e.rated : null });
    }
    return out;
  }, [d]);
  if (!data.length) return <p className="text-sm text-ink-3">—</p>;
  return (
    <BarChart
      data={data.map((e) => ({ x: String(e.year), y: e.n }))}
      label={t("dev.byYear")}
      barName={t("dev.vns")}
      line={{ name: t("dev.meanRating"), values: data.map((e) => e.rating), format: (v) => v.toFixed(1) }}
      height={170}
    />
  );
}

function DevPicker({ devs, exclude, onPick, name }: { devs: DevStats[]; exclude: number; onPick: (d: DevStats) => void; name: (d: DevStats) => string }) {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const nq = normalizeQuery(q);
  const hits = nq ? devs.filter((d) => d.id !== exclude && normalizeQuery(`${d.name ?? ""}${d.latin ?? ""}`).includes(nq)).sort((a, b) => b.count - a.count).slice(0, 8) : [];
  return (
    <div className="relative">
      <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("dev.search")} aria-label={t("dev.search")} className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3" />
      {hits.length > 0 && (
        <ul className="absolute z-30 mt-1 w-full rounded-md border border-line bg-surface shadow-lg">
          {hits.map((d) => (
            <li key={d.id}>
              <button
                type="button"
                onClick={() => {
                  onPick(d);
                  setQ("");
                }}
                className="flex w-full justify-between px-3 py-2 text-left text-sm hover:bg-surface-2"
              >
                <span>{name(d)}</span>
                <span className="text-xs text-ink-3 tabular">{d.count}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function VnTable({ d }: { d: DevStats }) {
  const { t, lang } = useI18n();
  const table = (items: DevStats["vns"]) => (
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-3">
            <tr>
              <th className="w-16 px-3 py-2 text-right font-medium">{t("rank.col.rank")}</th>
              <th className="px-3 py-2 font-medium">{t("rank.col.title")}</th>
              <th className="hidden px-3 py-2 font-medium sm:table-cell">{t("rank.col.lang")}</th>
              <th className="w-20 px-3 py-2 text-right font-medium">{t("rank.col.votes")}</th>
              <th className="w-20 px-3 py-2 text-right font-medium">{t("rank.col.rating")}</th>
            </tr>
          </thead>
          <tbody>
            {items.map((v) => (
              <tr key={v.id} className="border-t border-line">
                <td className="tabular px-3 py-2 text-right font-semibold">{v.rank}</td>
                <td className="max-w-0 px-3 py-2">
                  <VnLink vn={v} released={v.released} truncate />
                </td>
                <td className="hidden px-3 py-2 text-ink-2 sm:table-cell">{v.olang ? langName(v.olang, lang) : "—"}</td>
                <td className="tabular px-3 py-2 text-right text-ink-2">{formatInt(v.votes)}</td>
                <td className="tabular px-3 py-2 text-right text-ink-2">{v.rating?.toFixed(2) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
  );
  const [visible, more] = useShowMore(d.vns, 25, t("dev.titles", { n: d.count }), table);
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">{t("dev.titles", { n: d.count })}</h2>
      {table(visible)}
      {more}
    </section>
  );
}
