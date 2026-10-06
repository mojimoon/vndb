import { useMemo } from "react";
import { Link, Outlet, useOutletContext, useParams } from "react-router";
import { all, useCatalogue, useUser, type Catalogue, type CatalogueItem, type UserData } from "../../lib/api";
import { pearson } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { Tabs } from "../../components/Tabs";

export interface JoinedVote {
  vn: CatalogueItem;
  vote: number; // 1-10 (may be fractional, e.g. 7.5)
  diff: number | null; // vote - VNDB rating
}

export interface UserSummary {
  n: number;
  mean: number;
  std: number;
  corr: number | null;
  generosity: number | null;
  hist: number[];
}

export function joinVotes(user: UserData, cat: Catalogue): JoinedVote[] {
  return user.votes
    .map(([idx, v]) => {
      const vn = cat.byIdx[idx];
      return vn ? { vn, vote: v / 10, diff: vn.rating !== null ? v / 10 - vn.rating : null } : null;
    })
    .filter((x): x is JoinedVote => x !== null);
}

export function summarize(votes: JoinedVote[]): UserSummary {
  const n = votes.length;
  const mean = votes.reduce((s, v) => s + v.vote, 0) / Math.max(n, 1);
  const std = Math.sqrt(votes.reduce((s, v) => s + (v.vote - mean) ** 2, 0) / Math.max(n, 1));
  const rated = votes.filter((v) => v.diff !== null);
  const hist = Array(10).fill(0);
  for (const v of votes) hist[Math.min(10, Math.max(1, Math.floor(v.vote))) - 1]++;
  return {
    n,
    mean,
    std,
    corr: pearson(rated.map((v) => v.vote), rated.map((v) => v.vn.rating!)),
    generosity: rated.length ? rated.reduce((s, v) => s + v.diff!, 0) / rated.length : null,
    hist,
  };
}

export interface UserContext {
  user: UserData;
  cat: Catalogue;
  votes: JoinedVote[];
  summary: UserSummary;
}
export const useUserContext = () => useOutletContext<UserContext>();

export default function UserLayout() {
  const { uid } = useParams();
  const { t } = useI18n();
  const user = useUser(Number(uid));
  const cat = useCatalogue();
  const both = all<[UserData, Catalogue]>(user, cat);
  const ctx = useMemo<UserContext | null>(() => {
    if (both.state !== "ok") return null;
    const [u, c] = both.data;
    const votes = joinVotes(u, c);
    return { user: u, cat: c, votes, summary: summarize(votes) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [both.state, user.data, cat.data]);
  if (!ctx) return <Status value={both} />;
  const base = `/user/${ctx.user.uid}`;
  return (
    <article className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{ctx.user.name || `u${ctx.user.uid}`}</h1>
          <p className="tabular mt-1 text-sm text-ink-3">u{ctx.user.uid}</p>
        </div>
        <div className="flex gap-4 text-sm">
          <a href={`https://vndb.org/u${ctx.user.uid}`} target="_blank" rel="noreferrer" className="text-accent-ink hover:underline">
            {t("user.onVndb")} ↗
          </a>
          <Link to={`/compare?type=user&a=${ctx.user.uid}`} className="text-accent-ink hover:underline">
            {t("user.compare")} →
          </Link>
        </div>
      </header>
      <Tabs
        base={base}
        tabs={[
          { path: "", label: t("user.tab.overview") },
          { path: "votes", label: `${t("user.tab.votes")} (${ctx.votes.length})` },
          { path: "recs", label: t("user.tab.recs") },
          { path: "similar", label: t("user.tab.similar") },
          { path: "notes", label: t("vn.tab.notes") },
        ]}
      />
      <Outlet context={ctx} />
    </article>
  );
}
