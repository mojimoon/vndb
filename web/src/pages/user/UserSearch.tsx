import { useState } from "react";
import { useNavigate } from "react-router";
import { fetchJson, useMeta } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

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
    <div className="mx-auto max-w-xl space-y-4 py-6">
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
        <p className="tabular text-xs text-ink-3">{meta.data.stats.user_pages.toLocaleString()} users</p>
      )}
    </div>
  );
}
