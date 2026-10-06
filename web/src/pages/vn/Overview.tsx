import { Link } from "react-router";
import { methodName, relationName, useI18n } from "../../lib/i18n";
import { scoreLabels } from "../../lib/format";
import { BarChart, cumulativeLine } from "../../components/Charts";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

export default function Overview() {
  const { t, lang } = useI18n();
  const { vn, meta, cat } = useVnContext();
  const featured = meta.info.featured.filter((m) => vn.ranks[m]).slice(0, 8);
  const a = vn.analysis;
  const signed = (x: number) => (x > 0 ? "+" : "") + x.toFixed(2);
  return (
    <div className="space-y-6">
      <section className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {featured.map((m) => (
          <Link key={m} to={`/?m=${m}`} className="rounded-lg border border-line bg-surface px-3 py-2.5 hover:border-accent">
            <div className="truncate text-xs text-ink-3" title={methodName(m, lang)}>
              {methodName(m, lang)}
            </div>
            <div className="tabular mt-0.5 text-xl font-semibold">#{vn.ranks[m][0]}</div>
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
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-1">
          <Stat label={t("user.mean")} value={a.mean?.toFixed(2) ?? "—"} />
          <Stat label={t("vn.ratings.std")} value={a.std?.toFixed(2) ?? "—"} />
          <Stat label={t("vn.ratings.sp")} value={a.sp ? `${(a.sp.mean * 100).toFixed(0)}%` : "—"} />
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
