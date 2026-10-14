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

type Key = "name" | "cat" | "freq" | "sig" | "spoiler" | "voters" | "mean" | "score";

const CLOUD = 60;
const PAGE = 50;
const tagUrl = (id: number) => `https://vndb.org/g${id}`;

/** Tag cloud on top (content tags without spoilers), then the filterable, sortable table. */
export function TagAnalysis({ rows, kind, hint }: { rows: TagRow[]; kind: "vn" | "user"; hint: React.ReactNode }) {
  const { t } = useI18n();
  if (!rows.length) return <p className="rounded-lg border border-line bg-surface p-6 text-sm text-ink-2">{t("tags.none")}</p>;
  return (
    <div className="space-y-4">
      <Card title={t("tags.cloud")}>
        <TagCloud rows={rows.filter((r) => r.cat === 0 && r.spoiler === 0)} />
        <p className="mt-3 text-xs text-ink-3">{t("tags.cloudHint")}</p>
      </Card>
      <TagTable rows={rows} kind={kind} hint={hint} />
    </div>
  );
}

/** Tags sized by score (the top ones only), in alphabetical order. */
export function TagCloud({ rows }: { rows: TagRow[] }) {
  const { t } = useI18n();
  const top = useMemo(() => [...rows].sort((a, b) => b.score - a.score).slice(0, CLOUD), [rows]);
  if (!top.length) return <p className="text-sm text-ink-3">{t("tags.none")}</p>;
  const hi = top[0].score;
  const lo = top[top.length - 1].score;
  const frac = (s: number) => (hi > lo ? Math.sqrt((s - lo) / (hi - lo)) : 1);
  const words = [...top].sort((a, b) => a.name.localeCompare(b.name));
  return (
    <ul className="flex flex-wrap items-baseline justify-center gap-x-4 gap-y-1.5 py-2" lang="en">
      {words.map((r) => {
        const f = frac(r.score);
        return (
          <li key={r.id}>
            <a
              href={tagUrl(r.id)}
              target="_blank"
              rel="noreferrer"
              title={`${r.name} · ${t("tags.col.score")} ${r.score.toFixed(2)}`}
              className="leading-tight hover:text-accent-ink hover:underline"
              style={{
                fontSize: `${(0.8 + f * 1.15).toFixed(2)}rem`,
                fontWeight: f > 0.6 ? 600 : f > 0.3 ? 500 : 400,
                color: `color-mix(in oklab, var(--accent-ink) ${Math.round(35 + f * 65)}%, var(--text-2))`,
              }}
            >
              {r.name}
            </a>
          </li>
        );
      })}
    </ul>
  );
}

function TagTable({ rows, kind, hint }: { rows: TagRow[]; kind: "vn" | "user"; hint: React.ReactNode }) {
  const { t } = useI18n();
  // Default as on VNDB: content + technical, no spoilers.
  const [cats, setCats] = useState<number[]>([0, 2]);
  const [spoil, setSpoil] = useState(0);
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
    ["spoiler", "tags.col.spoiler", true],
    kind === "vn" ? ["voters", "tags.col.voters", true] : ["mean", "tags.col.mean", true],
    ["score", "tags.col.score", true],
  ];
  const head = (k: Key, label: StringKey, num: boolean) => (
    <th key={k} scope="col" aria-sort={sort.k === k ? (sort.asc ? "ascending" : "descending") : undefined} className={`px-3 py-2 font-medium ${num ? "text-right" : ""}`}>
      <button
        type="button"
        onClick={() => setSort({ k, asc: sort.k === k ? !sort.asc : k === "name" || k === "cat" || k === "spoiler" })}
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
        <select value={spoil} onChange={(e) => setSpoil(Number(e.target.value))} aria-label={t("tags.col.spoiler")} className="rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink">
          {[0, 1, 2].map((s) => (
            <option key={s} value={s}>
              {t(`tags.spoilUpTo.${s}` as StringKey)}
            </option>
          ))}
        </select>
        <span className="tabular ml-auto text-ink-3">{t("tags.count", { n: shown.length, total: rows.length })}</span>
      </div>
      <div className="overflow-x-auto">
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
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-ink-2">{t(`tags.cat.${r.cat}` as StringKey)}</td>
                <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.freq.toLocaleString()}</td>
                <td className="tabular px-3 py-1.5 text-right">{r.sig.toFixed(2)}</td>
                <td className={`whitespace-nowrap px-3 py-1.5 text-right ${r.spoiler ? "text-down" : "text-ink-3"}`}>{t(`tags.spoil.${r.spoiler}` as StringKey)}</td>
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
