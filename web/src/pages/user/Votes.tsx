import { useMemo, useState } from "react";
import { useMeta, useRanks } from "../../lib/api";
import { normalizeQuery } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useShowMore } from "../../components/ShowMore";
import { useUserContext } from "./UserLayout";

type Key = "vote" | "rating" | "diff" | "rank" | "year";

export default function UserVotes() {
  const { t } = useI18n();
  const { votes } = useUserContext();
  const meta = useMeta();
  const ranks = useRanks(meta.state === "ok" ? meta.data.info.default_method : null);
  const [sort, setSort] = useState<Key>("vote");
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const nq = normalizeQuery(q);
    const rank = (id: number) => (ranks.state === "ok" ? ranks.data.get(id)?.rank ?? 1e9 : 1e9);
    const key: Record<Key, (v: (typeof votes)[number]) => number> = {
      vote: (v) => -v.vote,
      rating: (v) => -(v.vn.rating ?? 0),
      diff: (v) => -(v.diff ?? -99),
      rank: (v) => rank(v.vn.id),
      year: (v) => -(v.vn.released ?? 0),
    };
    return votes.filter((v) => !nq || v.vn.search.includes(nq)).sort((a, b) => key[sort](a) - key[sort](b) || b.vn.votes - a.vn.votes);
  }, [votes, q, sort, ranks]);

  const [visible, more] = useShowMore(rows, 50, 100);
  const Th = ({ k, children }: { k: Key; children: React.ReactNode }) => (
    <th className="px-3 py-2 text-right font-medium" aria-sort={sort === k ? "ascending" : undefined}>
      <button type="button" onClick={() => setSort(k)} className={`hover:text-ink ${sort === k ? "text-ink" : ""}`}>
        {children}
        {sort === k && " ↓"}
      </button>
    </th>
  );

  return (
    <div className="space-y-3">
      <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("rank.search")} aria-label={t("rank.search")} className="w-full max-w-sm rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3" />
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="whitespace-nowrap border-b border-line text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-2 font-medium">{t("rank.col.title")}</th>
              <Th k="vote">{t("user.yourVote")}</Th>
              <Th k="rating">VNDB</Th>
              <Th k="diff">{t("user.diff")}</Th>
              <Th k="rank">{t("rank.col.rank")}</Th>
            </tr>
          </thead>
          <tbody>
            {visible.map((v) => (
              <tr key={v.vn.id} className="border-t border-line">
                <td className="max-w-0 truncate px-3 py-2">
                  <VnLink vn={v.vn} released={v.vn.released} />
                </td>
                <td className="tabular px-3 py-2 text-right font-semibold">{v.vote}</td>
                <td className="tabular px-3 py-2 text-right text-ink-2">{v.vn.rating?.toFixed(2) ?? "—"}</td>
                <td className={`tabular px-3 py-2 text-right ${v.diff === null ? "" : v.diff > 0 ? "text-up" : v.diff < 0 ? "text-down" : "text-ink-3"}`}>
                  {v.diff === null ? "—" : `${v.diff > 0 ? "+" : ""}${v.diff.toFixed(1)}`}
                </td>
                <td className="tabular px-3 py-2 text-right text-ink-2">{ranks.state === "ok" ? ranks.data.get(v.vn.id)?.rank ?? "—" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more}
    </div>
  );
}
