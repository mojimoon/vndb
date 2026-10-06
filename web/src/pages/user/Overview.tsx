import { useMemo } from "react";
import { decileLabels, samplePercentiles, scoreLabels } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { BarChart, Matrix, cumulativeLine } from "../../components/Charts";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useUserContext, type JoinedVote } from "./UserLayout";

const signed = (x: number | null, d = 2) => (x === null ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(d)}`);
const bucket = (x: number) => Math.min(9, Math.max(0, Math.floor(x)));
const grid = () => Array.from({ length: 10 }, () => Array(10).fill(0) as number[]);

export default function UserOverview() {
  const { t } = useI18n();
  const { votes, summary: s } = useUserContext();
  const withDiff = votes.filter((v) => v.diff !== null);
  const loved = [...withDiff].sort((a, b) => b.diff! - a.diff!).filter((v) => v.diff! > 0);
  const hated = [...withDiff].sort((a, b) => a.diff! - b.diff!).filter((v) => v.diff! < 0);
  // rows = y, columns = x
  const heat = useMemo(() => {
    const vndb = grid();
    const sci = grid();
    const sp = samplePercentiles(votes.map((v) => v.vote));
    votes.forEach((v, i) => {
      if (v.vn.rating !== null) vndb[bucket(v.vn.rating - 1)][bucket(v.vote - 1)]++;
      sci[bucket(v.sci * 10)][bucket(sp[i] * 10)]++;
    });
    return { vndb, sci };
  }, [votes]);
  return (
    <div className="space-y-4">
      <section className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <Stat label={t("user.votes")} value={s.n.toLocaleString()} />
        <Stat label={t("vn.meanStd")} value={`${s.mean.toFixed(2)} ± ${s.std.toFixed(2)}`} />
        <Stat label={t("user.corr")} value={s.corr === null ? "—" : s.corr.toFixed(2)} />
        <Stat label={t("user.corrSci")} value={s.corrSci === null ? "—" : s.corrSci.toFixed(2)} />
        <Stat label={t("user.generosity")} value={signed(s.generosity)} />
      </section>
      <Card title={t("user.dist")}>
        <BarChart
          data={s.hist.map((y, i) => ({ x: String(i + 1), y }))}
          label={t("user.dist")}
          barName={t("user.votes")}
          line={cumulativeLine(s.hist, t("chart.cumulative"))}
          height={170}
        />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("user.heatVndb")}>
          <p className="mb-2 text-xs text-ink-2">{t("user.heatVndbHint")}</p>
          <Matrix data={heat.vndb} labels={scoreLabels} xName={t("user.axisVote")} yName={t("user.axisVndb")} label={t("user.heatVndb")} />
        </Card>
        <Card title={t("user.heatSci")}>
          <p className="mb-2 text-xs text-ink-2">{t("user.heatSciHint")}</p>
          <Matrix data={heat.sci} labels={decileLabels} xName={t("user.axisSp")} yName={t("user.axisSci")} label={t("user.heatSci")} />
        </Card>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <DiffList title={t("user.loved")} rows={loved} />
        <DiffList title={t("user.hated")} rows={hated} />
      </div>
    </div>
  );
}

function DiffList({ title, rows }: { title: string; rows: JoinedVote[] }) {
  const { t } = useI18n();
  const table = (list: JoinedVote[]) => (
    <table className="w-full table-fixed text-sm">
      <thead className="text-left text-xs text-ink-2">
        <tr>
          <th className="py-1 font-medium">{t("rank.col.title")}</th>
          <th className="w-14 py-1 text-right font-medium">{t("user.yourVote")}</th>
          <th className="w-14 py-1 text-right font-medium">VNDB</th>
          <th className="w-14 py-1 text-right font-medium">Δ</th>
        </tr>
      </thead>
      <tbody>
        {list.map((v) => (
          <tr key={v.vn.id} className="border-t border-line">
            <td className="py-1.5 pr-2">
              <VnLink vn={v.vn} released={v.vn.released} truncate />
            </td>
            <td className="tabular py-1.5 text-right">{v.vote}</td>
            <td className="tabular py-1.5 text-right text-ink-2">{v.vn.rating?.toFixed(2)}</td>
            <td className={`tabular py-1.5 text-right ${v.diff! > 0 ? "text-up" : "text-down"}`}>{signed(v.diff, 1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  const [visible, more] = useShowMore(rows, 8, title, table);
  return (
    <Card title={title}>
      {table(visible)}
      {more}
    </Card>
  );
}
