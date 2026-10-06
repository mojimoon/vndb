import { pct } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { BarChart, HBars, LineChart } from "../../components/Charts";
import { Card, Stat } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

export default function Ratings() {
  const { t } = useI18n();
  const { vn } = useVnContext();
  const a = vn.analysis;
  const labels = [1, 2, 3, 4, 5, 6].map((k) => ({ label: t(`label.${k}` as never), value: a.labels[String(k)] ?? 0 })).filter((d) => d.value > 0);
  const years = a.years.map(([y, n, m]) => ({ year: y, n, m }));
  return (
    <div className="space-y-4">
      <section className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label={t("vn.ratings.count")} value={a.n.toLocaleString()} />
        <Stat label={t("user.mean")} value={a.mean?.toFixed(2) ?? "—"} />
        <Stat label={t("vn.ratings.std")} value={a.std?.toFixed(2) ?? "—"} />
        <Stat label={t("vn.rating")} value={vn.rating?.toFixed(2) ?? "—"} />
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("vn.ratings.dist")}>
          <BarChart data={a.hist.map((y, i) => ({ x: String(i + 1), y }))} label={t("vn.ratings.dist")} />
        </Card>
        <Card title={t("vn.ratings.labels")}>
          <HBars data={labels} />
        </Card>
        {a.sp && (
          <Card title={t("vn.ratings.sp")}>
            <p className="mb-2 text-xs text-ink-2">{t("vn.ratings.spHint", { p: pct(a.sp.mean) })}</p>
            <BarChart data={a.sp.hist.map((y, i) => ({ x: `${i * 20}–${i * 20 + 20}%`, y }))} label={t("vn.ratings.sp")} height={150} />
          </Card>
        )}
        {a.bias !== null && (
          <Card title={t("vn.ratings.bias")}>
            <p className="text-sm text-ink-2">{t("vn.ratings.biasHint", { b: (a.bias > 0 ? "+" : "") + a.bias.toFixed(2) })}</p>
            <BiasGauge value={a.bias} />
          </Card>
        )}
        {years.length > 1 && (
          <>
            <Card title={t("vn.ratings.byYear")}>
              <BarChart data={years.map((y) => ({ x: String(y.year), y: y.n }))} label={t("vn.ratings.byYear")} height={150} />
            </Card>
            <Card title={t("vn.ratings.meanByYear")}>
              <LineChart data={years.map((y) => ({ x: String(y.year), y: y.m }))} format={(v) => v.toFixed(2)} label={t("vn.ratings.meanByYear")} height={150} />
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

/** -3 .. +3 scale with the value marked; diverging from a neutral zero. */
function BiasGauge({ value }: { value: number }) {
  const clamp = Math.max(-3, Math.min(3, value));
  const pos = ((clamp + 3) / 6) * 100;
  return (
    <div className="mt-4">
      <div className="relative h-2 rounded-full" style={{ background: "linear-gradient(to right, var(--loss), var(--tie) 50%, var(--win))" }}>
        <span className="absolute -top-1.5 h-5 w-1 -translate-x-1/2 rounded bg-ink" style={{ left: `${pos}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-ink-3 tabular">
        <span>−3</span>
        <span>0</span>
        <span>+3</span>
      </div>
    </div>
  );
}
