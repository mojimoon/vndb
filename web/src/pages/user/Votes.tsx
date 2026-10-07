import { useMemo, useState } from "react";
import { normalizeQuery } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useUserContext, type JoinedVote } from "./UserLayout";

type Key = "vote" | "rating" | "diff" | "rank" | "year";

export default function UserVotes() {
  const { t } = useI18n();
  const { votes } = useUserContext();
  const [sort, setSort] = useState<Key>("vote");
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const nq = normalizeQuery(q);
    const key: Record<Key, (v: JoinedVote) => number> = {
      vote: (v) => -v.vote,
      rating: (v) => -(v.vn.rating ?? 0),
      diff: (v) => -(v.diff ?? -99),
      rank: (v) => v.vn.sci_rank ?? Number.MAX_SAFE_INTEGER,
      year: (v) => -(v.vn.released ?? 0),
    };
    return votes.filter((v) => !nq || v.vn.search.includes(nq)).sort((a, b) => key[sort](a) - key[sort](b) || b.vn.votes - a.vn.votes);
  }, [votes, q, sort]);

  const Th = ({ k, children, className = "" }: { k: Key; children: React.ReactNode; className?: string }) => (
    <th className={`px-3 py-2 text-right font-medium ${className}`} aria-sort={sort === k ? "ascending" : undefined}>
      <button type="button" onClick={() => setSort(k)} className={`hover:text-ink ${sort === k ? "text-ink" : ""}`}>
        {children}
        {sort === k && " ↓"}
      </button>
    </th>
  );
  const table = (list: JoinedVote[]) => (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface">
      <table className="w-full table-fixed text-sm">
        <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-2">
          <tr>
            <th className="px-3 py-2 font-medium">{t("rank.col.title")}</th>
            <Th k="vote" className="w-16">{t("user.yourVote")}</Th>
            <Th k="rating" className="w-20">VNDB</Th>
            <Th k="diff" className="hidden w-24 sm:table-cell">{t("user.diff")}</Th>
            <Th k="rank" className="w-28">SciRanking</Th>
          </tr>
        </thead>
        <tbody>
          {list.map((v) => (
            <tr key={v.vn.id} className="border-t border-line">
              <td className="px-3 py-2">
                <VnLink vn={v.vn} released={v.vn.released} truncate />
              </td>
              <td className="tabular px-3 py-2 text-right font-semibold">{v.vote}</td>
              <td className="tabular px-3 py-2 text-right text-ink-2">{v.vn.rating?.toFixed(2) ?? "—"}</td>
              <td className={`tabular hidden px-3 py-2 text-right sm:table-cell ${v.diff === null ? "" : v.diff > 0 ? "text-up" : v.diff < 0 ? "text-down" : "text-ink-2"}`}>
                {v.diff === null ? "—" : `${v.diff > 0 ? "+" : ""}${v.diff.toFixed(1)}`}
              </td>
              <td className="tabular px-3 py-2 text-right text-ink-2">{v.vn.sci_rank === null ? "—" : `#${v.vn.sci_rank}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  const [visible, more] = useShowMore(rows, 50, `${t("user.tab.votes")} (${rows.length})`, table);

  return (
    <div className="space-y-3">
      <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("rank.search")} aria-label={t("rank.search")} className="w-full max-w-sm rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3" />
      {table(visible)}
      {more}
    </div>
  );
}
