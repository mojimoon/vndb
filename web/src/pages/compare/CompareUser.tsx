import { useMemo } from "react";
import { Link, useParams } from "react-router";
import { all, useCatalogue, useUser, type Catalogue, type CatalogueItem, type UserData } from "../../lib/api";
import { pct, pearson } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { PairedBars, Scatter } from "../../components/Charts";
import { Status } from "../../components/Status";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { joinVotes, summarize } from "../user/UserLayout";
import { CompareTable, Swatch } from "./CompareVn";

export default function CompareUser() {
  const { a, b } = useParams();
  const ua = useUser(Number(a));
  const ub = useUser(Number(b));
  const cat = useCatalogue();
  const both = all<[UserData, UserData, Catalogue]>(ua, ub, cat);
  if (both.state !== "ok") return <Status value={both} />;
  const [x, y, c] = both.data;
  return <View x={x} y={y} cat={c} />;
}

interface Common {
  vn: CatalogueItem;
  a: number;
  b: number;
}

function View({ x, y, cat }: { x: UserData; y: UserData; cat: Catalogue }) {
  const { t } = useI18n();
  const nx = x.name || `u${x.uid}`;
  const ny = y.name || `u${y.uid}`;
  const d = useMemo(() => {
    const vx = joinVotes(x, cat);
    const vy = joinVotes(y, cat);
    const my = new Map(vy.map((v) => [v.vn.id, v.vote]));
    const mx = new Map(vx.map((v) => [v.vn.id, v.vote]));
    const common: Common[] = vx.filter((v) => my.has(v.vn.id)).map((v) => ({ vn: v.vn, a: v.vote, b: my.get(v.vn.id)! }));
    const r = pearson(common.map((c) => c.a), common.map((c) => c.b));
    const agree = common.length ? common.filter((c) => Math.abs(c.a - c.b) <= 1).length / common.length : null;
    return {
      sx: summarize(vx),
      sy: summarize(vy),
      common,
      r,
      agree,
      disagree: [...common].sort((p, q) => Math.abs(q.a - q.b) - Math.abs(p.a - p.b)).slice(0, 10).filter((c) => c.a !== c.b),
      shared: common.filter((c) => c.a >= 8 && c.b >= 8).sort((p, q) => Math.min(q.a, q.b) - Math.min(p.a, p.b) || q.a + q.b - p.a - p.b).slice(0, 10),
      onlyX: vx.filter((v) => v.vote >= 8 && !my.has(v.vn.id)).sort((p, q) => q.vote - p.vote || q.vn.votes - p.vn.votes).slice(0, 10),
      onlyY: vy.filter((v) => v.vote >= 8 && !mx.has(v.vn.id)).sort((p, q) => q.vote - p.vote || q.vn.votes - p.vn.votes).slice(0, 10),
    };
  }, [x, y, cat]);

  return (
    <article className="space-y-5">
      <header className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <Link to={`/user/${x.uid}`} className="flex items-center justify-end gap-2 text-right text-lg font-semibold hover:text-accent-ink">
          {nx}
          <Swatch i={0} />
        </Link>
        <Link to={`/compare/user/${y.uid}/${x.uid}`} className="rounded-full border border-line bg-surface px-2 py-1 text-xs text-ink-2 hover:text-ink" title={t("compare.swap")}>
          ⇄
        </Link>
        <Link to={`/user/${y.uid}`} className="flex items-center gap-2 text-lg font-semibold hover:text-accent-ink">
          <Swatch i={1} />
          {ny}
        </Link>
      </header>

      <section className="grid grid-cols-3 gap-2">
        <Stat label={t("compare.common")} value={d.common.length} />
        <Stat label={t("compare.pearson")} value={d.r === null ? "—" : d.r.toFixed(2)} />
        <Stat label={t("compare.agreeRate")} value={d.agree === null ? "—" : pct(d.agree)} />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("compare.scatter")}>
          <Scatter points={d.common.map((c) => ({ x: c.a, y: c.b, title: c.vn.title }))} xLabel={nx} yLabel={ny} label={t("compare.scatter")} />
        </Card>
        <div className="space-y-4">
          <Card>
            <CompareTable
              rows={[
                [t("user.votes"), String(d.sx.n), String(d.sy.n), 0],
                [t("user.mean"), d.sx.mean.toFixed(2), d.sy.mean.toFixed(2), 0],
                [t("vn.ratings.std"), d.sx.std.toFixed(2), d.sy.std.toFixed(2), 0],
                [t("user.corr"), d.sx.corr?.toFixed(2) ?? "—", d.sy.corr?.toFixed(2) ?? "—", 0],
              ]}
            />
          </Card>
          <Card title={t("user.dist")}>
            <PairedBars x={d.sx.hist.map((_, i) => String(i + 1))} a={d.sx.hist} b={d.sy.hist} names={[nx, ny]} label={t("user.dist")} />
          </Card>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <VoteList title={t("compare.disagree")} rows={d.disagree.map((c) => ({ vn: c.vn, a: c.a, b: c.b }))} />
        <VoteList title={t("compare.agree")} rows={d.shared.map((c) => ({ vn: c.vn, a: c.a, b: c.b }))} />
        <VoteList title={t("compare.only", { a: nx, b: ny })} rows={d.onlyX.map((v) => ({ vn: v.vn, a: v.vote, b: null }))} />
        <VoteList title={t("compare.only", { a: ny, b: nx })} rows={d.onlyY.map((v) => ({ vn: v.vn, a: null, b: v.vote }))} />
      </div>
    </article>
  );
}

function VoteList({ title, rows }: { title: string; rows: { vn: CatalogueItem; a: number | null; b: number | null }[] }) {
  if (!rows.length) return null;
  return (
    <Card title={title}>
      <table className="w-full text-sm">
        <tbody>
          {rows.map((r) => (
            <tr key={r.vn.id} className="border-t border-line first:border-0">
              <td className="max-w-0 truncate py-1.5 pr-2">
                <VnLink vn={r.vn} released={r.vn.released} />
              </td>
              <td className="tabular w-12 py-1.5 text-right">
                {r.a ?? "—"}
              </td>
              <td className="tabular w-12 py-1.5 text-right text-ink-2">
                {r.b ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
