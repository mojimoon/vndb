import { Link } from "react-router";
import { langName, methodName, relationName, useI18n } from "../../lib/i18n";
import { devName, formatInt, scoreLabels, year } from "../../lib/format";
import type { CatalogueItem } from "../../lib/api";
import { BarChart, cumulativeLine } from "../../components/Charts";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

export default function Overview() {
  const { t, lang } = useI18n();
  const { vn, meta, cat } = useVnContext();
  const grand = meta.info.default_method;
  const me = cat.byId.get(vn.id);
  // The grand ranking overall and within the title's release year, developer and original language.
  const scoped = (pred: (o: CatalogueItem) => boolean) => {
    if (me?.sci_rank == null) return null;
    let rank = 1;
    let of = 0;
    for (const o of cat.items) {
      if (o.sci_rank == null || !pred(o)) continue;
      of++;
      if (o.sci_rank < me.sci_rank) rank++;
    }
    return { rank, of };
  };
  const y = year(me?.released ?? null);
  const tiles = [
    { label: methodName(grand, lang), to: "/", r: vn.ranks[grand] ? { rank: vn.ranks[grand][0], of: cat.items.length } : null },
    y && { label: String(y), to: `/?from=${y}&to=${y}`, r: scoped((o) => year(o.released) === y) },
    me?.dev_id && { label: devName(me, lang) ?? "", to: `/?dev=${me.dev_id}`, r: scoped((o) => o.dev_id === me.dev_id) },
    me?.olang && { label: langName(me.olang, lang), to: `/?lang=${me.olang}`, r: scoped((o) => o.olang === me.olang) },
  ].filter((x): x is { label: string; to: string; r: { rank: number; of: number } } => !!x && !!x.r);
  const a = vn.analysis;
  const signed = (x: number) => (x > 0 ? "+" : "") + x.toFixed(2);
  return (
    <div className="space-y-6">
      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {tiles.map(({ label, to, r }, i) => (
          <Link key={i} to={to} className="rounded-lg border border-line bg-surface px-3 py-2.5 hover:border-accent" title={i ? `${methodName(grand, lang)} · ${label}` : label}>
            <div className="truncate text-xs text-ink-2">{label}</div>
            <div className="mt-0.5 flex items-baseline gap-1.5">
              <span className="tabular text-xl font-semibold">#{r.rank}</span>
              <span className="tabular text-xs text-ink-3">/ {formatInt(r.of)}</span>
            </div>
          </Link>
        ))}
      </section>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card title={t("vn.ratings.dist")} aside={<Link to="ratings" className="text-xs text-accent-ink hover:underline">{t("vn.tab.ratings")} →</Link>}>
          <BarChart
            data={a.hist.map((y, i) => ({ x: scoreLabels[i], y }))}
            label={t("vn.ratings.dist")}
            barName={t("vn.votes")}
            line={cumulativeLine(a.hist, t("chart.cumulative"))}
            height={160}
          />
        </Card>
        <div className="grid grid-cols-3 gap-2 lg:grid-cols-1 lg:grid-rows-3">
          <Stat label={t("vn.meanStd")} value={a.mean !== null && a.std !== null ? `${a.mean.toFixed(2)} ± ${a.std.toFixed(2)}` : "—"} />
          <Stat label={t("vn.ratings.sp")} value={a.sp ? `${(a.sp.mean * 100).toFixed(1)}%` : "—"} />
          <Stat label={t("vn.ratings.bias")} value={a.bias !== null ? signed(a.bias) : "—"} />
        </div>
      </div>

      {vn.similar.length > 0 && (
        <Card title={t("vn.tab.similar")} aside={<Link to="similar" className="text-xs text-accent-ink hover:underline">{t("common.more")} →</Link>}>
          <ul className="grid gap-x-6 gap-y-1.5 text-sm sm:grid-cols-2">
            {vn.similar.slice(0, 6).map((s) => {
              const o = cat.byId.get(s.id);
              return <li key={s.id}>{o ? <VnLink vn={o} released={o.released} /> : `v${s.id}`}</li>;
            })}
          </ul>
        </Card>
      )}

      {vn.relations.length > 0 && (
        <Card title={t("vn.relations")}>
          <ul className="grid gap-2 sm:grid-cols-2">
            {vn.relations.map((r) => {
              const o = cat.byId.get(r.id);
              return (
                <li key={r.id} className="text-sm">
                  <span className="mr-2 text-xs text-ink-3">{relationName(r.relation, lang)}</span>
                  {o ? <VnLink vn={o} released={o.released} /> : <span>v{r.id}</span>}
                </li>
              );
            })}
          </ul>
        </Card>
      )}
    </div>
  );
}
