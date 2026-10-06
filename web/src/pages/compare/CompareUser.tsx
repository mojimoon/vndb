import { useMemo, useState } from "react";
import { useParams } from "react-router";
import { all, useCatalogue, useUser, useUserNotes, type Catalogue, type CatalogueItem, type UserData, type UserNote } from "../../lib/api";
import { pct, pearson, samplePercentiles, scoreLabels, decileLabels, titles } from "../../lib/format";
import { useI18n, type StringKey } from "../../lib/i18n";
import { PairedBars, Scatter } from "../../components/Charts";
import { Modal } from "../../components/Modal";
import { NoteCard } from "../../components/NoteCard";
import { useShowMore } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { Card, Stat, VnLink } from "../../components/VnLink";
import { joinVotes, summarize } from "../user/UserLayout";
import { CompareHeader, CompareTable } from "./CompareVn";

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

type Mode = "raw" | "sp";
interface Scored {
  vn: CatalogueItem;
  raw: number; // 1-10
  sp: number; // 0-1, within the user's own ranked votes
}
interface Row {
  vn: CatalogueItem;
  a: number | null;
  b: number | null;
}

function scored(u: UserData, cat: Catalogue): Scored[] {
  const v = joinVotes(u, cat);
  const sp = samplePercentiles(v.map((x) => x.vote));
  return v.map((x, i) => ({ vn: x.vn, raw: x.vote, sp: sp[i] }));
}

function View({ x, y, cat }: { x: UserData; y: UserData; cat: Catalogue }) {
  const { t, lang } = useI18n();
  const [mode, setMode] = useState<Mode>("raw");
  const nx = x.name || `u${x.uid}`;
  const ny = y.name || `u${y.uid}`;
  const d = useMemo(() => {
    const sx = scored(x, cat);
    const sy = scored(y, cat);
    const my = new Map(sy.map((s) => [s.vn.id, s]));
    const mx = new Map(sx.map((s) => [s.vn.id, s]));
    const common = sx.filter((s) => my.has(s.vn.id)).map((s) => ({ vn: s.vn, a: s, b: my.get(s.vn.id)! }));
    return { sx, sy, mx, my, common };
  }, [x, y, cat]);

  // Everything below depends on the chosen scale.
  const val = (s: Scored) => (mode === "raw" ? s.raw : s.sp);
  const hi = mode === "raw" ? 8 : 0.8; // "rated highly": >= 8 or top 20% of their list
  const lo = mode === "raw" ? 5 : 0.3;
  const fmt = (v: number | null) => (v === null ? "—" : mode === "raw" ? String(v) : pct(v));
  const v = useMemo(() => {
    const pairs = d.common.map((c) => ({ vn: c.vn, a: val(c.a), b: val(c.b) }));
    const r = pearson(pairs.map((p) => p.a), pairs.map((p) => p.b));
    const tol = mode === "raw" ? 1 : 0.1;
    const agree = pairs.length ? pairs.filter((p) => Math.abs(p.a - p.b) <= tol).length / pairs.length : null;
    const byGap = [...pairs].sort((p, q) => Math.abs(q.a - q.b) - Math.abs(p.a - p.b)).filter((p) => p.a !== p.b);
    const only = (src: Scored[], other: Map<number, Scored>, high: boolean) =>
      src
        .filter((s) => !other.has(s.vn.id) && (high ? val(s) >= hi : val(s) <= lo))
        .sort((p, q) => (high ? val(q) - val(p) : val(p) - val(q)) || q.vn.votes - p.vn.votes);
    const hist = (src: Scored[]) => {
      const h = Array(10).fill(0);
      for (const s of src) h[mode === "raw" ? Math.min(9, Math.max(0, Math.floor(s.raw) - 1)) : Math.min(9, Math.floor(s.sp * 10))]++;
      return h;
    };
    return {
      pairs,
      r,
      agree,
      hx: hist(d.sx),
      hy: hist(d.sy),
      lists: [
        ["compare.disagree", byGap.map((p) => ({ vn: p.vn, a: p.a, b: p.b }))],
        ["compare.sharedHigh", pairs.filter((p) => p.a >= hi && p.b >= hi).sort((p, q) => Math.min(q.a, q.b) - Math.min(p.a, p.b))],
        ["compare.sharedLow", pairs.filter((p) => p.a <= lo && p.b <= lo).sort((p, q) => Math.max(p.a, p.b) - Math.max(q.a, q.b))],
        [["compare.onlyHigh", nx, ny], only(d.sx, d.my, true).map((s) => ({ vn: s.vn, a: val(s), b: null }))],
        [["compare.onlyHigh", ny, nx], only(d.sy, d.mx, true).map((s) => ({ vn: s.vn, a: null, b: val(s) }))],
        [["compare.onlyLow", nx, ny], only(d.sx, d.my, false).map((s) => ({ vn: s.vn, a: val(s), b: null }))],
        [["compare.onlyLow", ny, nx], only(d.sy, d.mx, false).map((s) => ({ vn: s.vn, a: null, b: val(s) }))],
      ] as [StringKey | [StringKey, string, string], Row[]][],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, mode]);
  const sumX = summarize(joinVotes(x, cat));
  const sumY = summarize(joinVotes(y, cat));
  // Reviews of both users (one request each), for the 💬 marks and the review modal.
  const notesX = useUserNotes(x.uid);
  const notesY = useUserNotes(y.uid);
  const reviews = useMemo(() => {
    const m = (r: typeof notesX) => new Map<number, UserNote>(r.state === "ok" ? r.data.notes.map((n) => [n.idx, n]) : []);
    return [m(notesX), m(notesY)] as const;
  }, [notesX, notesY]);
  const [open, setOpen] = useState<CatalogueItem | null>(null);

  return (
    <article className="space-y-5">
      <CompareHeader a={nx} b={ny} hrefA={`/user/${x.uid}`} hrefB={`/user/${y.uid}`} swap={`/compare/user/${y.uid}/${x.uid}`} />

      <div className="flex justify-center">
        <div className="inline-flex rounded-md border border-line bg-surface p-0.5 text-sm" role="group" aria-label={t("compare.scale")}>
          {(["raw", "sp"] as const).map((k) => (
            <button key={k} type="button" aria-pressed={mode === k} onClick={() => setMode(k)} className={`rounded px-3 py-1 ${mode === k ? "bg-surface-2 text-ink" : "text-ink-2"}`}>
              {t(k === "raw" ? "compare.raw" : "compare.sp")}
            </button>
          ))}
        </div>
      </div>

      <section className="grid grid-cols-3 gap-2">
        <Stat label={t("compare.common")} value={d.common.length} />
        <Stat label={t("compare.pearson")} value={v.r === null ? "—" : v.r.toFixed(2)} />
        <Stat label={t(mode === "raw" ? "compare.agreeRate" : "compare.agreeRateSp")} value={v.agree === null ? "—" : pct(v.agree)} />
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("compare.scatter")}>
          <Scatter
            points={v.pairs.map((p) => ({ x: mode === "raw" ? p.a : 1 + p.a * 9, y: mode === "raw" ? p.b : 1 + p.b * 9, title: p.vn.title }))}
            xLabel={nx}
            yLabel={ny}
            label={t("compare.scatter")}
          />
          {mode === "sp" && <p className="mt-1 text-center text-xs text-ink-3">{t("compare.spAxis")}</p>}
        </Card>
        <div className="space-y-4">
          <Card>
            <CompareTable
              heads={[nx, ny]}
              rows={[
                [t("user.votes"), String(sumX.n), String(sumY.n), 0],
                [t("user.mean"), sumX.mean.toFixed(2), sumY.mean.toFixed(2), 0],
                [t("vn.ratings.std"), sumX.std.toFixed(2), sumY.std.toFixed(2), 0],
                [t("user.corr"), sumX.corr?.toFixed(2) ?? "—", sumY.corr?.toFixed(2) ?? "—", 0],
                [t("user.corrSci"), sumX.corrSci?.toFixed(2) ?? "—", sumY.corrSci?.toFixed(2) ?? "—", 0],
              ]}
            />
          </Card>
          <Card title={t(mode === "raw" ? "user.dist" : "vn.ratings.sp")}>
            <PairedBars x={mode === "raw" ? scoreLabels : decileLabels} a={v.hx} b={v.hy} names={[nx, ny]} label={t("user.dist")} />
          </Card>
        </div>
      </div>

      <p className="text-sm text-ink-2">{t("compare.clickHint")}</p>
      <div className="grid gap-4 lg:grid-cols-2">
        {v.lists.map(([title, rows]) => {
          const label = Array.isArray(title) ? t(title[0], { a: title[1], b: title[2] }) : t(title);
          return <VoteList key={label + mode} title={label} rows={rows} heads={[nx, ny]} fmt={fmt} reviewed={(id) => reviews[0].has(id) || reviews[1].has(id)} onOpen={setOpen} />;
        })}
      </div>
      {open && (
        <Modal title={titles(open, lang).main} onClose={() => setOpen(null)}>
          <ReviewPair
            vn={open}
            users={[
              { name: nx, s: d.mx.get(open.id), note: reviews[0].get(open.idx) },
              { name: ny, s: d.my.get(open.id), note: reviews[1].get(open.idx) },
            ]}
          />
        </Modal>
      )}
    </article>
  );
}

function VoteList({
  title,
  rows,
  heads,
  fmt,
  reviewed,
  onOpen,
}: {
  title: string;
  rows: Row[];
  heads: [string, string];
  fmt: (v: number | null) => string;
  reviewed: (vnId: number) => boolean;
  onOpen: (vn: CatalogueItem) => void;
}) {
  const { lang } = useI18n();
  const table = (list: Row[]) => (
    <table className="w-full table-fixed text-sm">
      <thead className="text-xs text-ink-2">
        <tr>
          <th />
          {heads.map((h, i) => (
            <th key={i} className="w-20 truncate py-1 text-right font-medium" title={h}>
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {list.map((r) => (
          <tr key={r.vn.id} onClick={() => onOpen(r.vn)} className="cursor-pointer border-t border-line hover:bg-surface-2/60">
            <td className="py-1.5 pr-2">
              <span className="flex min-w-0 items-baseline gap-1.5" title={titles(r.vn, lang).main}>
                <span className="min-w-0 truncate font-medium">{titles(r.vn, lang).main}</span>
                {r.vn.released ? <span className="shrink-0 text-xs text-ink-2">{Math.floor(r.vn.released / 10000)}</span> : null}
                {reviewed(r.vn.id) && <span className="shrink-0 text-xs" aria-hidden>💬</span>}
              </span>
            </td>
            <td className="tabular py-1.5 text-right">{fmt(r.a)}</td>
            <td className="tabular py-1.5 text-right">{fmt(r.b)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  const [visible, more] = useShowMore(rows, 10, title, table);
  if (!rows.length) return null;
  return (
    <Card title={title}>
      {table(visible)}
      {more}
    </Card>
  );
}

/** Both users' votes and reviews of one title. */
function ReviewPair({ vn, users }: { vn: CatalogueItem; users: { name: string; s: Scored | undefined; note: UserNote | undefined }[] }) {
  const { t } = useI18n();
  return (
    <div className="space-y-3">
      <p className="text-sm">
        <VnLink vn={vn} released={vn.released} />
      </p>
      <ul className="space-y-2">
        {users.map((u, i) =>
          u.note ? (
            <NoteCard
              key={i}
              head={u.name}
              vote={u.note.vote}
              date={u.note.date}
              text={u.note.text}
              labels={u.note.labels}
              extra={u.s ? <span>{t("notes.top", { p: Math.max(1, Math.round(100 - u.s.sp * 100)) })}</span> : null}
            />
          ) : (
            <li key={i} className="flex items-baseline justify-between rounded-lg border border-dashed border-line px-4 py-3 text-sm">
              <span className="font-medium">{u.name}</span>
              <span className="text-xs text-ink-2">
                {u.s ? (
                  <>
                    <span className="mr-2 rounded bg-accent-soft px-1.5 py-0.5 font-semibold text-ink tabular">{u.s.raw}</span>
                    {t("compare.noReview")}
                  </>
                ) : (
                  t("compare.notVoted")
                )}
              </span>
            </li>
          ),
        )}
      </ul>
    </div>
  );
}
