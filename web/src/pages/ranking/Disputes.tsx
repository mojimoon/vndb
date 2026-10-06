import { useMemo } from "react";
import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useRanking } from "./RankingLayout";
import type { Row } from "./data";

const TOP = 500;

export default function Disputes() {
  const { t } = useI18n();
  const { filtered } = useRanking();
  const [under, over] = useMemo(() => {
    const u = filtered.filter((r) => r.rank <= TOP).sort((a, b) => b.vndb_rank - b.rank - (a.vndb_rank - a.rank));
    const o = filtered.filter((r) => r.vndb_rank <= TOP).sort((a, b) => b.rank - b.vndb_rank - (a.rank - a.vndb_rank));
    return [u.filter((r) => r.vndb_rank > r.rank), o.filter((r) => r.rank > r.vndb_rank)];
  }, [filtered]);
  return (
    <div className="space-y-3">
      <p className="text-sm text-ink-2">{t("rank.disputes.hint", { n: TOP })}</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <DisputeList title={t("rank.disputes.under")} rows={under} />
        <DisputeList title={t("rank.disputes.over")} rows={over} />
      </div>
    </div>
  );
}

function DisputeList({ title, rows }: { title: string; rows: Row[] }) {
  const { t } = useI18n();
  const [visible, more] = useShowMore(rows, 20, 30);
  return (
    <section className="rounded-lg border border-line bg-surface">
      <h2 className="border-b border-line px-4 py-2.5 text-sm font-semibold">{title}</h2>
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="w-14 px-3 py-1.5 text-right font-medium">{t("rank.col.rank")}</th>
            <th className="w-14 px-3 py-1.5 text-right font-medium">VNDB</th>
            <th className="px-3 py-1.5 font-medium">{t("rank.col.title")}</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r) => (
            <tr key={r.id} className="border-t border-line">
              <td className="tabular px-3 py-1.5 text-right font-semibold">{r.rank}</td>
              <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.vndb_rank}</td>
              <td className="px-3 py-1.5">
                <VnLink vn={r} released={r.released} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pb-2">{more}</div>
    </section>
  );
}
