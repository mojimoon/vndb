import { Link, useNavigate } from "react-router";
import { titles, year } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { H2HBar } from "../../components/H2H";
import { useVnContext } from "./VnLayout";

export default function Similar() {
  const { t, lang } = useI18n();
  const { vn, cat } = useVnContext();
  const navigate = useNavigate();
  const max = Math.max(0.01, ...vn.similar.map((s) => s.sim));
  const self = titles(vn, lang).main;
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">
        {t("vn.similarHint")} {t("sim.open")}.
      </p>
      {vn.similar.length === 0 ? (
        <p className="text-sm text-ink-2">{t("vn.h2h.none")}</p>
      ) : (
        <ol className="grid gap-2 md:grid-cols-2">
          {vn.similar.map((s, i) => {
            const o = cat.byId.get(s.id);
            const href = `/compare/vn/${vn.id}/${s.id}`;
            const name = o ? titles(o, lang).main : `v${s.id}`;
            return (
              <li
                key={s.id}
                role="link"
                tabIndex={0}
                onClick={() => navigate(href)}
                onKeyDown={(e) => e.key === "Enter" && navigate(href)}
                title={t("sim.open")}
                className="cursor-pointer rounded-lg border border-line bg-surface px-3 py-2.5 transition-colors hover:border-accent hover:bg-surface-2/60"
              >
                <div className="flex items-baseline gap-2 text-sm">
                  <span className="tabular w-5 shrink-0 text-xs text-ink-2">{i + 1}</span>
                  <Link to={`/vn/${s.id}`} onClick={(e) => e.stopPropagation()} className="min-w-0 truncate font-medium hover:text-accent-ink hover:underline">
                    {name}
                  </Link>
                  {o && year(o.released) && <span className="shrink-0 text-xs text-ink-2">{year(o.released)}</span>}
                </div>
                {o && (
                  <dl className="mt-1 grid grid-cols-4 gap-2 pl-7 text-xs">
                    <Fact k="SciRanking" v={o.sci_rank === null ? "—" : `#${o.sci_rank}`} />
                    <Fact k={t("rank.col.rating")} v={o.rating?.toFixed(2) ?? "—"} />
                    <Fact k={t("rank.col.votes")} v={o.votes.toLocaleString()} />
                    <Fact k={t("sim.similarity")} v={s.sim.toFixed(3)} />
                  </dl>
                )}
                <div className="mt-1.5 ml-7 h-1.5 rounded-r-[4px]" style={{ width: `calc((100% - 1.75rem) * ${s.sim / max})`, background: "var(--accent)" }} />
                {s.common > 0 && s.wins + s.losses > 0 && (
                  <div className="mt-2 pl-7">
                    <div className="mb-1 flex justify-between gap-2 text-[11px] text-ink-2">
                      <span className="truncate">{t("compare.prefers", { x: self })}</span>
                      <span className="truncate text-right">{t("compare.prefers", { x: name })}</span>
                    </div>
                    <H2HBar wins={s.wins} losses={s.losses} common={s.common} />
                    <p className="mt-1 text-[11px] text-ink-2 tabular">{t("vn.h2h.row", { w: s.wins, t: s.common - s.wins - s.losses, l: s.losses, n: s.common })}</p>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

export function Fact({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="truncate text-ink-2">{k}</dt>
      <dd className="tabular font-medium text-ink">{v}</dd>
    </div>
  );
}
