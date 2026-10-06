import { Link } from "react-router";
import { useI18n } from "../../lib/i18n";
import { useUserContext } from "./UserLayout";

export default function UserSimilar() {
  const { t } = useI18n();
  const { user } = useUserContext();
  const max = Math.max(0.01, ...user.similar.map((s) => s.sim));
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("user.similarHint")}</p>
      <ol className="space-y-2">
        {user.similar.map((s, i) => (
          <li key={s.uid} className="grid grid-cols-[1.5rem_1fr] items-baseline gap-2 rounded-lg border border-line bg-surface px-3 py-2.5">
            <span className="tabular text-xs text-ink-3">{i + 1}</span>
            <div className="min-w-0">
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <Link to={`/user/${s.uid}`} className="truncate font-medium hover:text-accent-ink">
                  {s.name || `u${s.uid}`}
                </Link>
                <span className="flex shrink-0 items-baseline gap-3 text-xs text-ink-3">
                  <span className="tabular">
                    {t("user.sim")} {s.sim.toFixed(3)} · {t("user.common")} {s.common}
                  </span>
                  <Link to={`/compare/user/${user.uid}/${s.uid}`} className="text-accent-ink hover:underline" aria-label={t("user.compare")}>
                    ⇄
                  </Link>
                </span>
              </div>
              <div className="mt-1.5 h-1.5 rounded-r-[4px]" style={{ width: `${(s.sim / max) * 100}%`, background: "var(--accent)" }} />
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
