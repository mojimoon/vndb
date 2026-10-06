import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { fetchJson, useMeta, type LeaderEntry, type Meta } from "../../lib/api";
import { useI18n, type StringKey } from "../../lib/i18n";
import { useShowMore } from "../../components/ShowMore";

/** "u123", "123", a vndb.org/u123 URL, or a username. */
export async function resolveUser(input: string): Promise<number | null> {
  const s = input.trim();
  const m = /(?:^|\/)u?(\d+)\/?$/i.exec(s);
  if (m) return Number(m[1]);
  if (!s) return null;
  try {
    const r = await fetchJson<{ uid: number }>(`/api/user-lookup?name=${encodeURIComponent(s)}`);
    return r.uid;
  } catch {
    return null;
  }
}

export function UserInput({ onResolved, placeholder }: { onResolved: (uid: number) => void; placeholder?: string }) {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);
  return (
    <form
      className="flex gap-2"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        const uid = await resolveUser(q);
        setBusy(false);
        setError(uid === null);
        if (uid !== null) onResolved(uid);
      }}
    >
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={placeholder ?? t("user.search.placeholder")}
        aria-label={placeholder ?? t("user.search.placeholder")}
        aria-invalid={error}
        className="min-w-0 flex-1 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3"
      />
      <button type="submit" disabled={busy || !q.trim()} className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
        {t("user.search.go")}
      </button>
      {error && <span className="sr-only">{t("user.notFound")}</span>}
    </form>
  );
}

export default function UserSearch() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const meta = useMeta();
  const [notFound, setNotFound] = useState(false);
  const min = meta.state === "ok" ? meta.data.info.config.min_user_votes ?? 5 : 5;
  return (
    <div className="space-y-8">
      <section className="mx-auto max-w-xl space-y-4 pt-4">
        <h1 className="text-2xl font-semibold tracking-tight">{t("user.search.title")}</h1>
        <p className="text-sm text-ink-2">{t("user.search.hint", { n: min })}</p>
        <UserInput
          onResolved={async (uid) => {
            try {
              await fetchJson(`/api/user/${uid}`);
              navigate(`/user/${uid}`);
            } catch {
              setNotFound(true);
            }
          }}
        />
        {notFound && <p className="text-sm text-down">{t("user.notFound")}</p>}
        {meta.state === "ok" && meta.data.stats.user_pages && (
          <p className="tabular text-xs text-ink-3">{t("user.pages", { n: meta.data.stats.user_pages.toLocaleString() })}</p>
        )}
      </section>
      {meta.state === "ok" && meta.data.leaderboards && <Leaderboards lb={meta.data.leaderboards} year={meta.data.info.dump_date.slice(0, 4)} />}
    </div>
  );
}

const BOARDS: [keyof NonNullable<Meta["leaderboards"]>, StringKey, (v: number) => string][] = [
  ["most_votes", "lb.mostVotes", (v) => v.toLocaleString()],
  ["most_votes_year", "lb.mostVotesYear", (v) => v.toLocaleString()],
  ["highest_mean", "lb.highestMean", (v) => v.toFixed(2)],
  ["lowest_mean", "lb.lowestMean", (v) => v.toFixed(2)],
  ["most_mainstream", "lb.mainstream", (v) => v.toFixed(3)],
  ["most_contrarian", "lb.contrarian", (v) => v.toFixed(3)],
];

function Leaderboards({ lb, year }: { lb: NonNullable<Meta["leaderboards"]>; year: string }) {
  const { t } = useI18n();
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold">{t("lb.title")}</h2>
      <p className="text-sm text-ink-2">{t("lb.hint", { n: lb.min_votes })}</p>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {BOARDS.map(([key, title, fmt]) => (
          <Board key={key} title={t(title, { y: year })} rows={lb[key] as LeaderEntry[]} fmt={fmt} />
        ))}
      </div>
    </section>
  );
}

function Board({ title, rows, fmt }: { title: string; rows: LeaderEntry[]; fmt: (v: number) => string }) {
  const { t } = useI18n();
  const [visible, more] = useShowMore(rows, 10, 20);
  return (
    <div className="rounded-lg border border-line bg-surface p-3">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <ol className="text-sm">
        {visible.map(([uid, name, value, n, page], i) => (
          <li key={uid} className="grid grid-cols-[1.5rem_1fr_auto] items-baseline gap-2 border-t border-line py-1 first:border-0">
            <span className="tabular text-xs text-ink-3">{i + 1}</span>
            {page ? (
              <Link to={`/user/${uid}`} className="truncate hover:text-accent-ink">
                {name || `u${uid}`}
              </Link>
            ) : (
              <a href={`https://vndb.org/u${uid}`} target="_blank" rel="noreferrer" className="truncate hover:text-accent-ink">
                {name || `u${uid}`} ↗
              </a>
            )}
            <span className="tabular text-right" title={t("lb.rankedVotes", { n })}>
              {fmt(value)}
            </span>
          </li>
        ))}
      </ol>
      {more}
    </div>
  );
}
