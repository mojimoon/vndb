import { useState } from "react";
import { Link, useParams } from "react-router";
import { all, useCatalogue, useJoint, useMeta, useVn, type Catalogue, type Meta, type VnDetail } from "../../lib/api";
import { decileLabels, formatInt, pct, scoreLabels, titles, year } from "../../lib/format";
import { methodName, useI18n } from "../../lib/i18n";
import { Matrix, PairedBars } from "../../components/Charts";
import { H2HBar } from "../../components/H2H";
import { Status } from "../../components/Status";
import { Card } from "../../components/VnLink";

export default function CompareVn() {
  const { a, b } = useParams();
  const va = useVn(Number(a));
  const vb = useVn(Number(b));
  const meta = useMeta();
  const cat = useCatalogue();
  const both = all<[VnDetail, VnDetail, Meta, Catalogue]>(va, vb, meta, cat);
  if (both.state !== "ok") return <Status value={both} />;
  const [x, y, m] = both.data;
  return <View x={x} y={y} featured={m.info.featured} />;
}

export function Swatch({ i }: { i: 0 | 1 }) {
  return <span aria-hidden className="inline-block h-3 w-3 shrink-0 rounded-sm" style={{ background: i === 0 ? "var(--accent)" : "var(--series-2)" }} />;
}

/** label | A | B, with the better side in bold (-1 = A, 1 = B). */
export function CompareTable({ rows, heads }: { rows: readonly (readonly [string, string, string, number])[]; heads?: [string, string] }) {
  return (
    <table className="w-full text-sm">
      {heads && (
        <thead className="text-xs text-ink-3">
          <tr>
            <th />
            {heads.map((h, i) => (
              <th key={i} className="w-28 py-1 text-right font-medium">
                <span className="inline-flex max-w-28 items-center justify-end gap-1.5">
                  <span className="truncate">{h}</span>
                  <Swatch i={i as 0 | 1} />
                </span>
              </th>
            ))}
          </tr>
        </thead>
      )}
      <tbody>
        {rows.map(([label, a, b, better]) => (
          <tr key={label} className="border-t border-line first:border-0">
            <td className="py-1.5 pr-2 text-ink-2">{label}</td>
            <td className={`tabular w-28 py-1.5 text-right ${better === -1 ? "font-semibold text-ink" : "text-ink-2"}`}>{a}</td>
            <td className={`tabular w-28 py-1.5 text-right ${better === 1 ? "font-semibold text-ink" : "text-ink-2"}`}>{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CompareHeader({ a, b, hrefA, hrefB, swap }: { a: string; b: string; hrefA: string; hrefB: string; swap: string }) {
  const { t } = useI18n();
  return (
    <header className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
      <Link to={hrefA} className="flex items-center justify-end gap-2 text-right text-lg font-semibold hover:text-accent-ink">
        <span className="min-w-0 truncate">{a}</span>
        <Swatch i={0} />
      </Link>
      <Link to={swap} className="rounded-full border border-line bg-surface px-2 py-1 text-xs text-ink-2 hover:text-ink" title={t("compare.swap")}>
        ⇄
      </Link>
      <Link to={hrefB} className="flex items-center gap-2 text-lg font-semibold hover:text-accent-ink">
        <Swatch i={1} />
        <span className="min-w-0 truncate">{b}</span>
      </Link>
    </header>
  );
}

function View({ x, y, featured }: { x: VnDetail; y: VnDetail; featured: string[] }) {
  const { t, lang } = useI18n();
  const joint = useJoint(x.idx, y.idx);
  const [mode, setMode] = useState<"raw" | "sp">("raw");
  const tx = titles(x, lang).main;
  const ty = titles(y, lang).main;
  const cmp = (nx: number | null, ny: number | null, higherBetter = true) =>
    nx === null || ny === null || nx === ny ? 0 : (nx > ny) === higherBetter ? -1 : 1;
  const signed = (v: number | null) => (v === null ? "—" : (v > 0 ? "+" : "") + v.toFixed(2));
  const stats = [
    [t("vn.votes"), formatInt(x.votes), formatInt(y.votes), cmp(x.votes, y.votes)],
    [t("vn.rating"), x.rating?.toFixed(2) ?? "—", y.rating?.toFixed(2) ?? "—", cmp(x.rating, y.rating)],
    [t("user.mean"), x.analysis.mean?.toFixed(2) ?? "—", y.analysis.mean?.toFixed(2) ?? "—", cmp(x.analysis.mean, y.analysis.mean)],
    [t("vn.ratings.std"), x.analysis.std?.toFixed(2) ?? "—", y.analysis.std?.toFixed(2) ?? "—", 0],
    [t("vn.ratings.sp"), x.analysis.sp ? pct(x.analysis.sp.mean) : "—", y.analysis.sp ? pct(y.analysis.sp.mean) : "—", cmp(x.analysis.sp?.mean ?? null, y.analysis.sp?.mean ?? null)],
    [t("vn.ratings.bias"), signed(x.analysis.bias), signed(y.analysis.bias), cmp(x.analysis.bias, y.analysis.bias)],
    [t("rank.col.year"), String(year(x.released) ?? "—"), String(year(y.released) ?? "—"), 0],
  ] as const;
  const ranks = featured
    .filter((m) => x.ranks[m] && y.ranks[m])
    .map((m) => [methodName(m, lang), `#${x.ranks[m][0]}`, `#${y.ranks[m][0]}`, cmp(x.ranks[m][0], y.ranks[m][0], false)] as const);

  return (
    <article className="space-y-5">
      <CompareHeader a={tx} b={ty} hrefA={`/vn/${x.id}`} hrefB={`/vn/${y.id}`} swap={`/compare/vn/${y.id}/${x.id}`} />

      <Card title={t("compare.h2h")}>
        {joint.state !== "ok" ? (
          <Status value={joint} />
        ) : joint.data.common ? (
          <>
            <div className="mb-2 grid grid-cols-3 text-center">
              {[
                [joint.data.wins, tx],
                [joint.data.common - joint.data.wins - joint.data.losses, t("compare.draw")],
                [joint.data.losses, ty],
              ].map(([n, l], i) => (
                <div key={i}>
                  <div className="tabular text-2xl font-semibold">{n}</div>
                  <div className="truncate text-xs text-ink-3">{i === 1 ? l : t("compare.prefers", { x: l })}</div>
                </div>
              ))}
            </div>
            <H2HBar wins={joint.data.wins} losses={joint.data.losses} common={joint.data.common} tall />
            <p className="mt-2 text-center text-xs text-ink-3">{t("vn.common", { n: joint.data.common })}</p>
          </>
        ) : (
          <p className="text-sm text-ink-3">{t("vn.noCommon")}</p>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("vn.ranks")}>
          <CompareTable rows={ranks} heads={[tx, ty]} />
        </Card>
        <Card title={t("vn.tab.ratings")}>
          <CompareTable rows={stats} heads={[tx, ty]} />
        </Card>
        <Card title={t("vn.ratings.dist")}>
          <PairedBars x={scoreLabels} a={x.analysis.hist} b={y.analysis.hist} names={[tx, ty]} label={t("vn.ratings.dist")} />
        </Card>
        <Card title={t("vn.ratings.sp")}>
          <PairedBars x={decileLabels} a={x.analysis.sp?.hist ?? []} b={y.analysis.sp?.hist ?? []} names={[tx, ty]} label={t("vn.ratings.sp")} />
        </Card>
      </div>

      {joint.state === "ok" && joint.data.common > 0 && (
        <Card
          title={t("compare.matrix")}
          aside={
            <div className="inline-flex rounded-md border border-line bg-surface p-0.5 text-xs">
              {(["raw", "sp"] as const).map((k) => (
                <button key={k} type="button" aria-pressed={mode === k} onClick={() => setMode(k)} className={`rounded px-2.5 py-1 ${mode === k ? "bg-surface-2 text-ink" : "text-ink-2"}`}>
                  {t(k === "raw" ? "compare.raw" : "compare.sp")}
                </button>
              ))}
            </div>
          }
        >
          <p className="mb-2 text-xs text-ink-2">{t("compare.matrixHint", { n: joint.data.common })}</p>
          <Matrix
            data={transpose(mode === "raw" ? joint.data.raw : joint.data.sp)}
            labels={mode === "raw" ? scoreLabels : decileLabels}
            xName={tx}
            yName={ty}
            label={t("compare.matrix")}
          />
        </Card>
      )}
    </article>
  );
}

/** joint[a][b] -> rows = B (y axis), columns = A (x axis). */
const transpose = (m: number[][]) => m[0].map((_, c) => m.map((row) => row[c]));
