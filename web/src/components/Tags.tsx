import { useMemo, useState } from "react";
import { useI18n, type StringKey } from "../lib/i18n";
import { Card } from "./VnLink";

/** One tag of a title or a user, with its TF-IDF score. Tag names are never translated. */
export interface TagRow {
  id: number;
  name: string;
  cat: number; // 0 content, 1 sexual, 2 technical
  spoiler: number; // 0 none, 1 minor, 2 major
  freq: number; // VN page: VNs with the tag on VNDB; user page: the user's titles with it
  sig: number; // significance (VNDB tag rating, 0-3]
  score: number;
  voters?: number; // VN page
  mean?: number; // user page: the user's mean vote (1-10) on titles with the tag
}

/** vn: one title (spoilers, voters); user / dev: summed over a user's votes or a developer's titles. */
export type TagKind = "vn" | "user" | "dev";

type Key = "name" | "cat" | "freq" | "sig" | "voters" | "mean" | "score";

const CLOUD = 35;
const CLOUD_MOBILE = 20;
const PAGE = 50;
const tagUrl = (id: number) => `https://vndb.org/g${id}`;

/**
 * Tag cloud on top (content tags), then the filterable, sortable table. On a
 * title page the table's spoiler setting also applies to the cloud; user pages
 * have no spoiler levels (a tag summed over many titles spoils none of them).
 */
export function TagAnalysis({ rows, kind, hint }: { rows: TagRow[]; kind: TagKind; hint: React.ReactNode }) {
  const { t } = useI18n();
  const [spoil, setSpoil] = useState(0);
  const level = kind === "vn" ? spoil : 2;
  if (!rows.length) return <p className="rounded-lg border border-line bg-surface p-6 text-sm text-ink-2">{t("tags.none")}</p>;
  return (
    <div className="space-y-4">
      <Card title={t("tags.cloud")}>
        <TagCloud rows={rows.filter((r) => r.cat === 0 && r.spoiler <= level)} marks={kind === "vn"} hint={t(kind === "vn" ? "tags.cloudHintVn" : "tags.cloudHint")} />
      </Card>
      <TagTable rows={rows} kind={kind} hint={hint} spoil={level} setSpoil={setSpoil} />
    </div>
  );
}

/** Colour for a significance: grey up to 1, then teal -> blue -> violet at 3 (tokens --sig-0..3 in index.css). */
export const sigColor = (sig: number) => {
  const x = Math.max(0, Math.min(3, (sig - 1) * 1.5));
  const i = Math.min(2, Math.floor(x));
  return `color-mix(in oklab, var(--sig-${i + 1}) ${Math.round((x - i) * 100)}%, var(--sig-${i}))`;
};

/** Scrambles the order (stable per tag) so big and small words mix like in a word cloud. */
const scramble = (id: number) => Math.imul(id ^ 0x9e3779b9, 0x85ebca6b) >>> 0;

/**
 * Word cloud, colour by significance. Size is the geometric mean of the score's
 * position between the lowest and highest shown score and a Zipf-like rank
 * curve, so only a handful of tags are big whether the scores are spread out
 * (user pages) or packed together (title pages, where many tags score alike).
 * Phones show the top CLOUD_MOBILE tags in a narrower size range.
 */
export function TagCloud({ rows, marks, hint }: { rows: TagRow[]; marks: boolean; hint: string }) {
  const { t } = useI18n();
  const top = useMemo(() => [...rows].sort((a, b) => b.score - a.score).slice(0, CLOUD), [rows]);
  if (!top.length) return <p className="text-sm text-ink-3">{t("tags.none")}</p>;
  const hi = top[0].score;
  const lo = top[top.length - 1].score;
  const n = top.length;
  const order = new Map(top.map((r, i) => [r.id, i]));
  const zipfLo = 1 / Math.sqrt(n);
  const zipf = (i: number) => (n > 1 ? (1 / Math.sqrt(i + 1) - zipfLo) / (1 - zipfLo) : 1);
  const lin = (s: number) => (hi > lo ? (s - lo) / (hi - lo) : 1);
  const size = (r: TagRow) => Math.sqrt(zipf(order.get(r.id)!) * lin(r.score));
  const words = [...top].sort((a, b) => scramble(a.id) - scramble(b.id));
  return (
    <>
      <ul
        className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 py-2 [--cmax:2.1rem] [--cmin:0.8rem] sm:gap-x-5 sm:gap-y-2 sm:[--cmax:3.2rem] sm:[--cmin:0.85rem]"
        lang="en"
      >
        {words.map((r) => {
          const f = size(r);
          return (
            <li key={r.id} className={order.get(r.id)! >= CLOUD_MOBILE ? "hidden sm:block" : undefined}>
              <a
                href={tagUrl(r.id)}
                target="_blank"
                rel="noreferrer"
                title={`${r.name} · ${t("tags.col.score")} ${r.score.toFixed(2)} · ${t("tags.col.sig")} ${r.sig.toFixed(2)}`}
                className="block leading-[1.1] tracking-tight hover:underline"
                style={{
                  fontSize: `calc(var(--cmin) + (var(--cmax) - var(--cmin)) * ${f.toFixed(3)})`,
                  fontWeight: f > 0.5 ? 700 : f > 0.2 ? 600 : 500,
                  color: sigColor(r.sig),
                }}
              >
                {r.name}
                {marks && <SpoilerMark level={r.spoiler} />}
              </a>
            </li>
          );
        })}
      </ul>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
        <span className="inline-flex items-center gap-1.5">
          {t("tags.legend.color")} 0
          <span aria-hidden className="h-2 w-20 rounded-full" style={{ background: "linear-gradient(to right in oklab, var(--sig-0) 33.3%, var(--sig-1) 55.6%, var(--sig-2) 77.8%, var(--sig-3))" }} />3
        </span>
        <span>{t("tags.legend.size")}</span>
        <span>{hint}</span>
      </div>
    </>
  );
}

/** VNDB-style spoiler marker after a tag name: yellow "s" (minor), red "S" (major). */
export function SpoilerMark({ level }: { level: number }) {
  const { t } = useI18n();
  if (!level) return null;
  return (
    <sup
      title={t(`tags.spoiler.${level}` as StringKey)}
      aria-label={t(`tags.spoiler.${level}` as StringKey)}
      className={`ml-[0.1em] cursor-help font-bold leading-none ${level === 1 ? "text-[0.7em] text-spoil-minor" : "text-[0.78em] text-down"}`}
    >
      {level === 1 ? "s" : "S"}
    </sup>
  );
}

function TagTable({ rows, kind, hint, spoil, setSpoil }: { rows: TagRow[]; kind: TagKind; hint: React.ReactNode; spoil: number; setSpoil: (s: number) => void }) {
  const { t } = useI18n();
  // Default as on VNDB: content + technical, no spoilers.
  const [cats, setCats] = useState<number[]>([0, 2]);
  const [sort, setSort] = useState<{ k: Key; asc: boolean }>({ k: "score", asc: false });
  const [all, setAll] = useState(false);
  const shown = useMemo(() => {
    const list = rows.filter((r) => cats.includes(r.cat) && r.spoiler <= spoil);
    const dir = sort.asc ? 1 : -1;
    const val = (r: TagRow) => (sort.k === "name" ? r.name : (r[sort.k] ?? -Infinity));
    return list.sort((a, b) => {
      const x = val(a);
      const y = val(b);
      const c = typeof x === "string" ? x.localeCompare(y as string) : (x as number) - (y as number);
      return c * dir || b.score - a.score;
    });
  }, [rows, cats, spoil, sort]);
  const cols: [Key, StringKey, boolean][] = [
    ["name", "tags.col.tag", false],
    ["cat", "tags.col.cat", false],
    ["freq", kind === "vn" ? "tags.col.freq" : "tags.col.titles", true],
    ["sig", "tags.col.sig", true],
    kind === "vn" ? ["voters", "tags.col.voters", true] : ["mean", kind === "dev" ? "tags.col.meanRating" : "tags.col.mean", true],
    ["score", "tags.col.score", true],
  ];
  const head = (k: Key, label: StringKey, num: boolean) => (
    <th key={k} scope="col" aria-sort={sort.k === k ? (sort.asc ? "ascending" : "descending") : undefined} className={`px-3 py-2 font-medium ${num ? "text-right" : ""}`}>
      <button
        type="button"
        onClick={() => setSort({ k, asc: sort.k === k ? !sort.asc : k === "name" || k === "cat" })}
        className={`whitespace-nowrap hover:text-ink ${sort.k === k ? "text-ink" : ""}`}
      >
        {t(label)}
        {sort.k === k ? (sort.asc ? " ↑" : " ↓") : ""}
      </button>
    </th>
  );
  const rowsShown = all ? shown : shown.slice(0, PAGE);
  return (
    <Card title={t("tags.table")}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        {[0, 2, 1].map((c) => {
          const on = cats.includes(c);
          return (
            <button
              key={c}
              type="button"
              aria-pressed={on}
              onClick={() => setCats(on ? cats.filter((x) => x !== c) : [...cats, c])}
              className={`rounded-md border px-2.5 py-1 ${on ? "border-accent bg-accent-soft text-ink" : "border-line bg-surface text-ink-2 hover:bg-surface-2"}`}
            >
              {t(`tags.cat.${c}` as StringKey)}
            </button>
          );
        })}
        {kind === "vn" && (
        <select value={spoil} onChange={(e) => setSpoil(Number(e.target.value))} aria-label={t("tags.col.spoiler")} className="rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink">
          {[0, 1, 2].map((s) => (
            <option key={s} value={s}>
              {t(`tags.spoilUpTo.${s}` as StringKey)}
            </option>
          ))}
        </select>
        )}
        <span className="tabular ml-auto text-ink-3">{t("tags.count", { n: shown.length, total: rows.length })}</span>
      </div>
      <div className="overflow-x-auto overflow-y-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-line text-left text-xs text-ink-3">
            <tr>{cols.map(([k, label, num]) => head(k, label, num))}</tr>
          </thead>
          <tbody>
            {rowsShown.map((r) => (
              <tr key={r.id} className="border-b border-line last:border-0 hover:bg-surface-2/60">
                <td className="px-3 py-1.5" lang="en">
                  <a href={tagUrl(r.id)} target="_blank" rel="noreferrer" className="hover:text-accent-ink hover:underline">
                    {r.name}
                  </a>
                  {kind === "vn" && <SpoilerMark level={r.spoiler} />}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-ink-2">{t(`tags.cat.${r.cat}` as StringKey)}</td>
                <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.freq.toLocaleString()}</td>
                <td className="tabular px-3 py-1.5 text-right font-medium" style={{ color: sigColor(r.sig) }}>
                  {r.sig.toFixed(2)}
                </td>
                {kind === "vn" ? (
                  <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.voters ?? "—"}</td>
                ) : (
                  <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.mean?.toFixed(2) ?? "—"}</td>
                )}
                <td className="tabular px-3 py-1.5 text-right font-medium">{r.score.toFixed(r.score < 1 ? 3 : 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!shown.length && <p className="p-6 text-center text-sm text-ink-3">{t("tags.none")}</p>}
      </div>
      {shown.length > PAGE && (
        <div className="mt-2 flex justify-center">
          <button type="button" onClick={() => setAll(!all)} className="rounded-md border border-line bg-surface px-3 py-1 text-xs font-medium hover:bg-surface-2">
            {all ? t("tags.fewer") : t("common.showAll", { n: shown.length.toLocaleString() })}
          </button>
        </div>
      )}
      <div className="mt-3 text-xs leading-relaxed text-ink-3">{hint}</div>
    </Card>
  );
}
