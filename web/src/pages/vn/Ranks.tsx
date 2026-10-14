import { Link } from "react-router";
import { dayToDate, formatScore } from "../../lib/format";
import { methodGroup, methodName, rankerName, variableName, useI18n, RANKER_CODES, VARIABLE_CODES, type MethodGroup, type StringKey } from "../../lib/i18n";
import { RankLines } from "../../components/Charts";
import { Card } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";


export default function Ranks() {
  const { t, lang } = useI18n();
  const { vn, meta } = useVnContext();
  const codes = Object.keys(vn.ranks);
  const total = meta?.stats.ranked_vns ?? null;
  const histMethods = meta?.info.history_methods ?? [meta?.info.default_method ?? "borda_grand", "vndb"];
  return (
    <div className="space-y-4">
      <Card title={t("vn.history")}>
        {vn.history.length > 1 ? (
          <>
            <p className="mb-2 text-xs text-ink-3">{t("vn.historyHint")}</p>
            <RankLines
              points={vn.history.map(([d, ...ys]) => ({ x: d, ys }))}
              names={histMethods.map((m) => methodName(m, lang))}
              formatX={dayToDate}
              label={t("vn.history")}
            />
          </>
        ) : (
          <p className="text-sm text-ink-3">{t("vn.historyEmpty")}</p>
        )}
      </Card>
      <RankitTable vn={vn} total={total} />
      {/* Merged rankings on the left; PONet and the VNDB rating (reference) on the right. */}
      <div className="grid items-start gap-4 md:grid-cols-2">
        <GroupTable group="merged" codes={codes} vn={vn} total={total} />
        <div className="space-y-4">
          <GroupTable group="po" codes={codes} vn={vn} total={total} />
          <GroupTable group="ref" codes={codes} vn={vn} total={total} />
        </div>
      </div>
    </div>
  );
}

type Ranked = { ranks: Record<string, [number, number | null]> };
const top = (rank: number, total: number) => `top ${Math.max(0.1, (rank / total) * 100).toFixed(1)}%`;

function GroupTable({ group, codes, vn, total }: { group: MethodGroup; codes: string[]; vn: Ranked; total: number | null }) {
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
            <td className="tabular px-3 py-1.5 text-right font-medium">#{vn.ranks[c][0]}</td>
            {total && <td className="tabular w-16 px-1 py-1.5 text-right text-xs text-ink-3">{top(vn.ranks[c][0], total)}</td>}
            <td className="tabular w-20 px-3 py-1.5 text-right text-xs text-ink-3">{formatScore(vn.ranks[c][1])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Scientific rankings as a ranker x variable grid, as on the Methods page; the best rank is highlighted. */
function RankitTable({ vn, total }: { vn: Ranked; total: number | null }) {
  const { t, lang } = useI18n();
  const vars = VARIABLE_CODES.filter((v) => RANKER_CODES.some((r) => vn.ranks[`${r}_${v}`]));
  const rankers = RANKER_CODES.filter((r) => vars.some((v) => vn.ranks[`${r}_${v}`]));
  if (!vars.length) return null;
  const best = Math.min(...rankers.flatMap((r) => vars.map((v) => vn.ranks[`${r}_${v}`]?.[0] ?? Infinity)));
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
                      <Link to={`/?m=${code}`} className="group inline-block" title={`${methodName(code, lang)} · ${formatScore(x[1])}`}>
                        <span className={`tabular font-medium group-hover:text-accent-ink ${x[0] === best ? "text-accent-ink" : ""}`}>#{x[0]}</span>
                        {total && <span className="tabular block text-[11px] text-ink-3">{top(x[0], total)}</span>}
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
