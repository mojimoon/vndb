import { useMemo } from "react";
import { methodName, useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useRanking } from "./RankingLayout";
import { Delta } from "./RankTable";
import type { Row } from "./data";

export default function Movers() {
  const { t, lang } = useI18n();
  const { filtered, meta } = useRanking();
  const [up, down] = useMemo(() => {
    const withTrend = filtered.filter((r) => r.trend !== null && r.trend !== 0);
    return [
      [...withTrend].filter((r) => r.trend! > 0).sort((a, b) => b.trend! - a.trend!),
      [...withTrend].filter((r) => r.trend! < 0).sort((a, b) => a.trend! - b.trend!),
    ];
  }, [filtered]);
  if (!up.length && !down.length) return <p className="rounded-lg border border-line bg-surface p-6 text-sm text-ink-2">{t("rank.movers.none")}</p>;
  return (
    <div className="space-y-3">
      <p className="text-sm text-ink-2">{methodName(meta.info.default_method, lang)} · 7d</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <MoverList title={t("rank.movers.up")} rows={up} />
        <MoverList title={t("rank.movers.down")} rows={down} />
      </div>
    </div>
  );
}

function MoverList({ title, rows }: { title: string; rows: Row[] }) {
  const [visible, more] = useShowMore(rows, 20, 30);
  return (
    <section className="rounded-lg border border-line bg-surface">
      <h2 className="border-b border-line px-4 py-2.5 text-sm font-semibold">{title}</h2>
      <ul>
        {visible.map((r) => (
          <li key={r.id} className="flex items-baseline gap-3 border-t border-line px-3 py-1.5 text-sm first:border-0">
            <span className="w-14 shrink-0 text-right text-xs">
              <Delta d={r.trend!} />
            </span>
            <VnLink vn={r} released={r.released} className="truncate" />
          </li>
        ))}
      </ul>
      <div className="pb-2">{more}</div>
    </section>
  );
}
