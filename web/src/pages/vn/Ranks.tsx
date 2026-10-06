import { Link } from "react-router";
import { dayToDate, formatScore } from "../../lib/format";
import { methodGroup, methodName, useI18n, type MethodGroup, type StringKey } from "../../lib/i18n";
import { RankLines } from "../../components/Charts";
import { Card } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

const GROUPS: MethodGroup[] = ["merged", "po", "rankit", "ref"];

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
      <div className="grid gap-4 md:grid-cols-2">
        {GROUPS.map((g) => {
          const list = codes.filter((c) => methodGroup(c) === g);
          if (!list.length) return null;
          return (
            <table key={g} className="w-full self-start rounded-lg border border-line bg-surface text-sm">
              <caption className="px-3 pt-2 text-left text-xs font-medium text-ink-3">{t(`methods.group.${g}` as StringKey)}</caption>
              <tbody>
                {list.map((c) => (
                  <tr key={c} className="border-t border-line first:border-0">
                    <td className="px-3 py-1.5">
                      <Link to={`/?m=${c}`} className="hover:text-accent-ink">
                        {methodName(c, lang)}
                      </Link>
                    </td>
                    <td className="tabular px-3 py-1.5 text-right font-medium">#{vn.ranks[c][0]}</td>
                    {total && <td className="tabular w-16 px-1 py-1.5 text-right text-xs text-ink-3">top {Math.max(0.1, (vn.ranks[c][0] / total) * 100).toFixed(1)}%</td>}
                    <td className="tabular w-20 px-3 py-1.5 text-right text-xs text-ink-3">{formatScore(vn.ranks[c][1])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          );
        })}
      </div>
    </div>
  );
}
