import { useMemo, useState } from "react";
import { useHistoryDay } from "../../lib/api";
import { dayToDate } from "../../lib/format";
import { methodName, useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { VnLink } from "../../components/VnLink";
import { useRanking } from "./RankingLayout";
import { Delta } from "./RankTable";
import type { Row } from "./data";

const PAGE = 100;
const DAY_MS = 86_400_000;
const dateToDay = (s: string) => Math.round((Date.parse(`${s}T00:00:00Z`) - Date.UTC(2000, 0, 1)) / DAY_MS);

type SortKey = "rank" | "vndb" | "rating" | "votes";

interface HistRow {
  id: number;
  row: Row | undefined; // undefined: no longer ranked
  rank: number;
  vndb: number;
  rating: number; // NaN = none
  votes: number; // 0 = not recorded that day
}

/** The ranking as it stood on a past day: grand rank, VNDB rank / rating / votes then, and the change since. */
export default function History() {
  const { t, lang } = useI18n();
  const { meta, rows, filtered, params, update } = useRanking();
  const days = meta.info.history_days ?? [];
  const today = meta.info.day;
  // Closest recorded day at or before the chosen date (default: ~30 days ago).
  const wanted = params.get("date") ? dateToDay(params.get("date")!) : today - 30;
  const day = days.length ? ([...days].reverse().find((d) => d <= wanted) ?? days[0]) : null;
  const hist = useHistoryDay(day);
  const [sort, setSort] = useState<{ k: SortKey; asc: boolean }>({ k: "rank", asc: true });
  const [limit, setLimit] = useState(PAGE);

  const list = useMemo(() => {
    if (hist.state !== "ok") return [];
    const byId = new Map(rows.map((r) => [r.id, r]));
    const keep = filtered.length === rows.length ? null : new Set(filtered.map((r) => r.id));
    const out: HistRow[] = [];
    const h = hist.data;
    for (let i = 0; i < h.n; i++) {
      const id = h.vid[i];
      if (keep && !keep.has(id)) continue; // filters only apply to titles ranked today
      out.push({ id, row: byId.get(id), rank: h.rank[i], vndb: h.vndbRank[i], rating: h.rating[i], votes: h.votes[i] });
    }
    const val = (r: HistRow) => (sort.k === "rank" ? r.rank : sort.k === "vndb" ? r.vndb : sort.k === "rating" ? (isNaN(r.rating) ? -1 : r.rating) : r.votes);
    const dir = sort.asc ? 1 : -1;
    return out.sort((a, b) => (val(a) - val(b)) * dir || a.rank - b.rank);
  }, [hist, rows, filtered, sort]);

  if (!days.length) return <p className="rounded-lg border border-line bg-surface p-6 text-sm text-ink-2">{t("rank.history.none")}</p>;
  const first = days[0];
  const last = days[days.length - 1];
  const idx = days.indexOf(day!);
  const go = (d: number) => {
    setLimit(PAGE);
    update({ date: dayToDate(d) }, false);
  };
  const hasVotes = list.some((r) => r.votes > 0);
  const head = (k: SortKey, label: string, cls = "") => (
    <th scope="col" aria-sort={sort.k === k ? (sort.asc ? "ascending" : "descending") : undefined} className={`px-3 py-2 text-right font-medium ${cls}`}>
      <button
        type="button"
        onClick={() => setSort({ k, asc: sort.k === k ? !sort.asc : k === "rank" || k === "vndb" })}
        className={`whitespace-nowrap hover:text-ink ${sort.k === k ? "text-ink" : ""}`}
      >
        {label}
        {sort.k === k ? (sort.asc ? " ↑" : " ↓") : ""}
      </button>
    </th>
  );
  const btn = "rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs hover:bg-surface-2 disabled:opacity-40 disabled:hover:bg-surface";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button type="button" className={btn} disabled={idx <= 0} onClick={() => go(days[idx - 1])} aria-label={t("rank.history.prev")}>
          ‹
        </button>
        <input
          type="date"
          min={dayToDate(first)}
          max={dayToDate(last)}
          value={dayToDate(day!)}
          onChange={(e) => e.target.value && go(dateToDay(e.target.value))}
          aria-label={t("rank.history.date")}
          className="rounded-md border border-line bg-surface px-2 py-1 text-sm text-ink"
        />
        <button type="button" className={btn} disabled={idx >= days.length - 1} onClick={() => go(days[idx + 1])} aria-label={t("rank.history.next")}>
          ›
        </button>
        {[7, 30, 90, 365].map((n) =>
          today - n >= first ? (
            <button key={n} type="button" className={btn} onClick={() => go(today - n)}>
              {t("rank.history.ago", { n: n === 365 ? "1y" : `${n}d` })}
            </button>
          ) : null,
        )}
      </div>
      <p className="text-xs leading-relaxed text-ink-3">
        {t("rank.history.hint", { method: methodName(meta.info.default_method, lang), first: dayToDate(first), n: days.length.toLocaleString() })}
      </p>
      {hist.state !== "ok" ? (
        <Status value={hist} />
      ) : (
        <section className="rounded-lg border border-line bg-surface">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-4 py-2.5">
            <h2 className="text-sm font-semibold">{dayToDate(day!)}</h2>
            <span className="tabular text-xs text-ink-3">{t("rank.history.count", { n: list.length.toLocaleString(), total: hist.data.n.toLocaleString() })}</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-line text-left text-xs text-ink-3">
                <tr>
                  {head("rank", methodName(meta.info.default_method, lang), "w-20")}
                  <th scope="col" className="w-16 px-3 py-2 text-right font-medium">
                    {t("rank.history.now")}
                  </th>
                  <th scope="col" className="w-16 px-3 py-2 text-right font-medium">
                    {t("rank.history.diff")}
                  </th>
                  <th scope="col" className="px-3 py-2 font-medium">
                    {t("rank.col.title")}
                  </th>
                  {head("vndb", t("vn.vndbRank"))}
                  {head("rating", t("vn.rating"))}
                  {hasVotes && head("votes", t("rank.col.votes"))}
                </tr>
              </thead>
              <tbody>
                {list.slice(0, limit).map((r) => {
                  const now = r.row ? (r.row.sci_rank ?? r.row.rank) : null;
                  return (
                    <tr key={r.id} className="border-b border-line last:border-0 hover:bg-surface-2/60">
                      <td className="tabular px-3 py-1.5 text-right font-semibold">{r.rank}</td>
                      <td className="tabular px-3 py-1.5 text-right text-xs">{now === null ? <span className="text-ink-3">—</span> : <span className="text-ink-2">{now}</span>}</td>
                      <td className="tabular whitespace-nowrap px-3 py-1.5 text-right text-xs">{now !== null && <Delta d={r.rank - now} />}</td>
                      <td className="min-w-24 px-3 py-1.5 [overflow-wrap:anywhere]">
                        {r.row ? (
                          <VnLink vn={r.row} released={r.row.released} />
                        ) : (
                          <span>
                            <a href={`https://vndb.org/v${r.id}`} target="_blank" rel="noreferrer" className="font-medium hover:text-accent-ink">
                              v{r.id}
                            </a>
                            <span className="ml-1.5 text-xs text-ink-3">{t("rank.history.gone")}</span>
                          </span>
                        )}
                      </td>
                      <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.vndb}</td>
                      <td className="tabular px-3 py-1.5 text-right text-ink-2">{isNaN(r.rating) ? "—" : r.rating.toFixed(2)}</td>
                      {hasVotes && <td className="tabular px-3 py-1.5 text-right text-ink-2">{r.votes ? r.votes.toLocaleString() : "—"}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!list.length && <p className="p-6 text-center text-sm text-ink-3">{t("rank.history.empty")}</p>}
          </div>
          {list.length > limit && (
            <div className="flex justify-center p-2">
              <button type="button" onClick={() => setLimit(limit + 4 * PAGE)} className="rounded-md border border-line bg-surface px-3 py-1 text-xs font-medium hover:bg-surface-2">
                {t("rank.history.more", { n: Math.min(4 * PAGE, list.length - limit), total: list.length.toLocaleString() })}
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
