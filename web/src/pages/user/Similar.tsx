import { Link, useNavigate } from "react-router";
import { useI18n } from "../../lib/i18n";
import { H2HBar } from "../../components/H2H";
import { Fact } from "../vn/Similar";
import { useUserContext } from "./UserLayout";

export default function UserSimilar() {
  const { t } = useI18n();
  const { user } = useUserContext();
  const navigate = useNavigate();
  const me = user.name || `u${user.uid}`;
  const max = Math.max(0.01, ...user.similar.map((s) => s.sim));
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">
        {t("user.similarHint")} {t("sim.open")}.
      </p>
      <ol className="grid gap-2 md:grid-cols-2">
        {user.similar.map((s, i) => {
          const href = `/compare/user/${user.uid}/${s.uid}`;
          const hasH2h = s.higher !== undefined && s.common > 0;
          return (
            <li
              key={s.uid}
              role="link"
              tabIndex={0}
              onClick={() => navigate(href)}
              onKeyDown={(e) => e.key === "Enter" && navigate(href)}
              title={t("sim.open")}
              className="cursor-pointer rounded-lg border border-line bg-surface px-3 py-2.5 transition-colors hover:border-accent hover:bg-surface-2/60"
            >
              <div className="flex items-baseline gap-2 text-sm">
                <span className="tabular w-5 shrink-0 text-xs text-ink-2">{i + 1}</span>
                <Link to={`/user/${s.uid}`} onClick={(e) => e.stopPropagation()} className="min-w-0 truncate font-medium hover:text-accent-ink hover:underline">
                  {s.name || `u${s.uid}`}
                </Link>
              </div>
              <dl className="mt-1 grid grid-cols-3 gap-2 pl-7 text-xs">
                <Fact k={t("sim.similarity")} v={s.sim.toFixed(3)} />
                <Fact k={t("user.common")} v={s.common.toLocaleString()} />
                <Fact k={t("user.votes")} v={s.votes?.toLocaleString() ?? "—"} />
              </dl>
              <div className="mt-1.5 ml-7 h-1.5 rounded-r-[4px]" style={{ width: `calc((100% - 1.75rem) * ${s.sim / max})`, background: "var(--accent)" }} />
              {hasH2h && (
                <div className="mt-2 pl-7">
                  <H2HBar wins={s.higher} losses={s.lower} common={s.common} />
                  <p className="mt-1 text-[11px] text-ink-2 tabular">{t("sim.userH2h", { a: me, h: s.higher, e: s.equal, l: s.lower })}</p>
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
