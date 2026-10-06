import { useI18n } from "../../lib/i18n";
import { BarChart } from "../../components/Charts";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useUserContext, type JoinedVote } from "./UserLayout";

const signed = (x: number | null, d = 2) => (x === null ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(d)}`);

export default function UserOverview() {
  const { t } = useI18n();
  const { votes, summary: s } = useUserContext();
  const withDiff = votes.filter((v) => v.diff !== null);
  const loved = [...withDiff].sort((a, b) => b.diff! - a.diff!).filter((v) => v.diff! > 0);
  const hated = [...withDiff].sort((a, b) => a.diff! - b.diff!).filter((v) => v.diff! < 0);
  return (
    <div className="space-y-4">
      <section className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label={t("user.votes")} value={s.n.toLocaleString()} />
        <Stat label={t("user.mean")} value={`${s.mean.toFixed(2)} ± ${s.std.toFixed(2)}`} />
        <Stat label={t("user.corr")} value={s.corr === null ? "—" : s.corr.toFixed(2)} />
        <Stat label={t("user.generosity")} value={signed(s.generosity)} />
      </section>
      <Card title={t("user.dist")}>
        <BarChart data={s.hist.map((y, i) => ({ x: String(i + 1), y }))} label={t("user.dist")} height={150} />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <DiffList title={t("user.loved")} rows={loved} />
        <DiffList title={t("user.hated")} rows={hated} />
      </div>
    </div>
  );
}

function DiffList({ title, rows }: { title: string; rows: JoinedVote[] }) {
  const { t } = useI18n();
  const [visible, more] = useShowMore(rows, 8, 20);
  return (
    <Card title={title}>
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1 font-medium">{t("rank.col.title")}</th>
            <th className="w-14 py-1 text-right font-medium">{t("user.yourVote")}</th>
            <th className="w-14 py-1 text-right font-medium">VNDB</th>
            <th className="w-14 py-1 text-right font-medium">Δ</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((v) => (
            <tr key={v.vn.id} className="border-t border-line">
              <td className="max-w-0 truncate py-1.5 pr-2">
                <VnLink vn={v.vn} released={v.vn.released} />
              </td>
              <td className="tabular py-1.5 text-right">{v.vote}</td>
              <td className="tabular py-1.5 text-right text-ink-2">{v.vn.rating?.toFixed(2)}</td>
              <td className={`tabular py-1.5 text-right ${v.diff! > 0 ? "text-up" : "text-down"}`}>{signed(v.diff, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {more}
    </Card>
  );
}
