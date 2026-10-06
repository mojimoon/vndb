import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

export default function Similar() {
  const { t } = useI18n();
  const { vn, others } = useVnContext();
  const max = Math.max(0.01, ...vn.similar.map((s) => s.sim));
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("vn.similarHint")}</p>
      {vn.similar.length === 0 ? (
        <p className="text-sm text-ink-3">{t("vn.h2h.none")}</p>
      ) : (
        <ol className="space-y-2">
          {vn.similar.map((s, i) => {
            const o = others.get(s.id);
            return (
              <li key={s.id} className="grid grid-cols-[1.5rem_1fr] items-baseline gap-2 rounded-lg border border-line bg-surface px-3 py-2.5">
                <span className="tabular text-xs text-ink-3">{i + 1}</span>
                <div className="min-w-0">
                  <div className="flex items-baseline justify-between gap-3 text-sm">
                    {o ? <VnLink vn={o} released={o.released} className="truncate" /> : <span>v{s.id}</span>}
                    <span className="tabular shrink-0 text-xs text-ink-3">
                      {s.sim.toFixed(3)} · {t("vn.common", { n: s.common })}
                    </span>
                  </div>
                  <div className="mt-1.5 h-1.5 rounded-r-[4px]" style={{ width: `${(s.sim / max) * 100}%`, background: "var(--accent)" }} />
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
