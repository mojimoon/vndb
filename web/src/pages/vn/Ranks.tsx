import { Link } from "react-router";
import { dayToDate, formatScore } from "../../lib/format";
import { methodGroup, methodName, rankerName, variableName, useI18n, RANKER_CODES, VARIABLE_CODES, type MethodGroup, type StringKey } from "../../lib/i18n";
import { compact, RankLines } from "../../components/Charts";
import { Card } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";
import type { VnDetail } from "../../lib/api";


export default function Ranks() {
  const { t, lang } = useI18n();
  const { vn, meta } = useVnContext();
  const codes = Object.keys(vn.ranks);
  const total = meta?.stats.ranked_vns ?? null;
  const scale = { total: total && total > 0 ? total : null };
  const histMethods = meta?.info.history_methods ?? [meta?.info.default_method ?? "borda_grand", "vndb"];
  return (
    <div className="space-y-4">

      <Card title={t("vn.history")}>
        {vn.history.length > 1 ? (
          <>
            <p className="mb-2 text-xs text-ink-3">{t("vn.historyHint")}</p>
            <RankLines
              points={vn.history.map(([d, rank, vndbRank]) => ({ x: d, ys: [rank, vndbRank] }))}
              names={histMethods.map((m) => methodName(m, lang))}
              formatX={dayToDate}
              label={t("vn.history")}
            />
          </>
        ) : (
          <p className="text-sm text-ink-3">{t("vn.historyEmpty")}</p>
        )}
      </Card>
      <VndbHistory history={vn.history} />
      {/* Merged rankings on the left; PONet and the VNDB rating (reference) on the right. */}
      <div className="grid items-start gap-4 md:grid-cols-2">
        <GroupTable group="merged" codes={codes} vn={vn} scale={scale} />
        <div className="space-y-4">
          <GroupTable group="po" codes={codes} vn={vn} scale={scale} />
          <GroupTable group="ref" codes={codes} vn={vn} scale={scale} />
        </div>
      </div>
      <RankitTable vn={vn} scale={scale} />
    </div>
  );
}

/** VNDB rating and votes over time (recorded since they were added to the history). */
function VndbHistory({ history }: { history: VnDetail["history"] }) {
  const { t } = useI18n();
  const pts = history.filter((p) => p[3] != null || p[4] != null);
  if (pts.length < 2) return null;
  return (
    <Card title={t("vn.ratingHistory")}>
      <p className="mb-2 text-xs text-ink-3">{t("vn.ratingHistoryHint")}</p>
      <div className="grid gap-4 md:grid-cols-2">
        <RankLines
          points={pts.map((p) => ({ x: p[0], ys: [p[3] ?? null] }))}
          names={[t("vn.rating")]}
          formatX={dayToDate}
          formatY={(v) => v.toFixed(2)}
          values
          height={160}
          label={t("vn.rating")}
        />
        <RankLines
          points={pts.map((p) => ({ x: p[0], ys: [p[4] ?? null] }))}
          names={[t("rank.col.votes")]}
          formatX={dayToDate}
          formatY={(v) => (v >= 10_000 ? compact(v) : Math.round(v).toLocaleString())}
          values
          height={160}
          label={t("rank.col.votes")}
        />
      </div>
    </Card>
  );
}

type Ranked = { ranks: Record<string, [number, number | null]> };
const top = (rank: number, total: number) => `top ${Math.max(0.1, (rank / total) * 100).toFixed(1)}%`;

type RankScale = { total: number | null };

function RankDisplay({ rank, scale }: { rank: number; scale: RankScale }) {
  return (
    <span className="tabular inline-flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="font-bold">#{rank}</span>
      {scale.total !== null && <span className="text-[11px] text-ink-3">{top(rank, scale.total)}</span>}
    </span>
  );
}

function GroupTable({ group, codes, vn, scale }: { group: MethodGroup; codes: string[]; vn: Ranked; scale: RankScale }) {
  const { t, lang } = useI18n();
  const list = codes.filter((c) => methodGroup(c) === group);
  if (!list.length) return null;
  return (
    <table className="w-full rounded-lg border border-line bg-surface text-sm">
      <caption className="px-3 pt-2 text-left text-xs font-medium text-ink-3">{t(`methods.group.${group}` as StringKey)}</caption>
      <tbody>
        {list.map((c) => (
          <tr key={c} className="border-t border-line first:border-0">
            <td className="px-3 py-1.5">
              <Link to={`/?m=${c}`} className="hover:text-accent-ink">
                {methodName(c, lang)}
              </Link>
            </td>
            <td className="px-3 py-1.5 text-right"><RankDisplay rank={vn.ranks[c][0]} scale={scale} /></td>
            <td className="tabular w-20 px-3 py-1.5 text-right text-xs text-ink-3">{formatScore(vn.ranks[c][1])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}


function RankitTable({ vn, scale }: { vn: Ranked; scale: RankScale }) {
  const { t, lang } = useI18n();
  const vars = VARIABLE_CODES.filter((v) => RANKER_CODES.some((r) => vn.ranks[`${r}_${v}`]));
  const rankers = RANKER_CODES.filter((r) => vars.some((v) => vn.ranks[`${r}_${v}`]));
  if (!vars.length) return null;

  return (
    <section className="overflow-x-auto rounded-lg border border-line bg-surface">
      <table className="w-full min-w-[640px] text-sm">
        <caption className="px-3 pt-2 text-left text-xs font-medium text-ink-3">{t("methods.group.rankit")}</caption>
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="px-3 py-2" />
            {vars.map((v) => (
              <th key={v} className="px-3 py-2 text-right font-medium">
                {variableName(v, lang)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rankers.map((r) => (
            <tr key={r} className="border-t border-line">
              <th scope="row" className="whitespace-nowrap px-3 py-1.5 text-left font-normal">
                {rankerName(r, lang)}
              </th>
              {vars.map((v) => {
                const code = `${r}_${v}`;
                const x = vn.ranks[code];
                return (
                  <td key={v} className="px-3 py-1.5 text-right">
                    {x ? (
                      <Link to={`/?m=${code}`} className="inline-block hover:underline" title={`${methodName(code, lang)} · ${formatScore(x[1])}`}>
                        <RankDisplay rank={x[0]} scale={scale} />
                      </Link>
                    ) : (
                      <span className="text-ink-3">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
