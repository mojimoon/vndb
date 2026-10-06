import { Link, useParams } from "react-router";
import { all, useMeta, usePair, useVn, type PairResult, type VnDetail } from "../../lib/api";
import { formatInt, pct, titles, year } from "../../lib/format";
import { methodName, useI18n } from "../../lib/i18n";
import { PairedBars } from "../../components/Charts";
import { Status } from "../../components/Status";
import { Card } from "../../components/VnLink";
import { H2HBar } from "../vn/Versus";

export default function CompareVn() {
  const { a, b } = useParams();
  const va = useVn(Number(a));
  const vb = useVn(Number(b));
  const pair = usePair(Number(a), Number(b));
  const meta = useMeta();
  const both = all<[VnDetail, VnDetail, PairResult]>(va, vb, pair);
  if (both.state !== "ok") return <Status value={both} />;
  const [x, y, p] = both.data;
  return <View x={x} y={y} p={p} featured={meta.state === "ok" ? meta.data.info.featured : Object.keys(x.ranks).slice(0, 10)} minCommon={meta.state === "ok" ? meta.data.info.config.min_common_vote : 5} />;
}

function View({ x, y, p, featured, minCommon }: { x: VnDetail; y: VnDetail; p: PairResult; featured: string[]; minCommon: number }) {
  const { t, lang } = useI18n();
  const tx = titles(x, lang).main;
  const ty = titles(y, lang).main;
  const ties = p.common - p.wins - p.losses;
  const rows: [string, (v: VnDetail) => string, (v: VnDetail) => number | null, boolean][] = [
    [t("vn.votes"), (v) => formatInt(v.votes), (v) => v.votes, true],
    [t("vn.rating"), (v) => v.rating?.toFixed(2) ?? "—", (v) => v.rating, true],
    [t("user.mean"), (v) => v.analysis.mean?.toFixed(2) ?? "—", (v) => v.analysis.mean, true],
    [t("vn.ratings.std"), (v) => v.analysis.std?.toFixed(2) ?? "—", () => null, true],
    [t("vn.ratings.sp"), (v) => (v.analysis.sp ? pct(v.analysis.sp.mean) : "—"), (v) => v.analysis.sp?.mean ?? null, true],
    [t("vn.ratings.bias"), (v) => (v.analysis.bias === null ? "—" : (v.analysis.bias > 0 ? "+" : "") + v.analysis.bias.toFixed(2)), (v) => v.analysis.bias, true],
    [t("rank.col.year"), (v) => String(year(v.released) ?? "—"), () => null, true],
  ];
  return (
    <article className="space-y-5">
      <header className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <Link to={`/vn/${x.id}`} className="flex items-center justify-end gap-2 text-right text-lg font-semibold hover:text-accent-ink">
          {tx}
          <Swatch i={0} />
        </Link>
        <Link to={`/compare/vn/${y.id}/${x.id}`} className="rounded-full border border-line bg-surface px-2 py-1 text-xs text-ink-2 hover:text-ink" title={t("compare.swap")}>
          ⇄
        </Link>
        <Link to={`/vn/${y.id}`} className="flex items-center gap-2 text-lg font-semibold hover:text-accent-ink">
          <Swatch i={1} />
          {ty}
        </Link>
      </header>

      <Card title={t("compare.h2h")}>
        {p.common ? (
          <>
            <div className="grid grid-cols-3 text-center">
              <div>
                <div className="tabular text-2xl font-semibold">{p.wins}</div>
                <div className="text-xs text-ink-3">{tx}</div>
              </div>
              <div>
                <div className="tabular text-2xl font-semibold text-ink-2">{ties}</div>
                <div className="text-xs text-ink-3">=</div>
              </div>
              <div>
                <div className="tabular text-2xl font-semibold">{p.losses}</div>
                <div className="text-xs text-ink-3">{ty}</div>
              </div>
            </div>
            <H2HBar wins={p.wins} losses={p.losses} common={p.common} />
            <p className="mt-2 text-center text-xs text-ink-3">{t("vn.common", { n: p.common })}</p>
          </>
        ) : (
          <p className="text-sm text-ink-3">{t("vn.belowThreshold", { n: minCommon })}</p>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("vn.ranks")}>
          <CompareTable
            rows={[
              ...featured.filter((m) => x.ranks[m] && y.ranks[m]).map((m) => [methodName(m, lang), `#${x.ranks[m][0]}`, `#${y.ranks[m][0]}`, x.ranks[m][0] < y.ranks[m][0] ? -1 : x.ranks[m][0] > y.ranks[m][0] ? 1 : 0] as const),
            ]}
          />
        </Card>
        <Card title={t("vn.tab.ratings")}>
          <CompareTable
            rows={rows.map(([label, fmt, num]) => {
              const nx = num(x);
              const ny = num(y);
              return [label, fmt(x), fmt(y), nx === null || ny === null || nx === ny ? 0 : nx > ny ? -1 : 1] as const;
            })}
          />
        </Card>
      </div>
      <Card title={t("vn.ratings.dist")}>
        <PairedBars x={x.analysis.hist.map((_, i) => String(i + 1))} a={x.analysis.hist} b={y.analysis.hist} names={[tx, ty]} label={t("vn.ratings.dist")} />
      </Card>
    </article>
  );
}

export function Swatch({ i }: { i: 0 | 1 }) {
  return <span aria-hidden className="inline-block h-3 w-3 shrink-0 rounded-sm" style={{ background: i === 0 ? "var(--accent)" : "var(--series-2)" }} />;
}

/** label | A | B, with the better side in bold (-1 = A, 1 = B). */
export function CompareTable({ rows }: { rows: readonly (readonly [string, string, string, number])[] }) {
  return (
    <table className="w-full text-sm">
      <tbody>
        {rows.map(([label, a, b, better]) => (
          <tr key={label} className="border-t border-line first:border-0">
            <td className="py-1.5 pr-2 text-ink-2">{label}</td>
            <td className={`tabular w-24 py-1.5 text-right ${better === -1 ? "font-semibold text-ink" : "text-ink-2"}`}>{a}</td>
            <td className={`tabular w-24 py-1.5 text-right ${better === 1 ? "font-semibold text-ink" : "text-ink-2"}`}>{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
