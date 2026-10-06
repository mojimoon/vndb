import { useI18n } from "../../lib/i18n";
import { VnLink } from "../../components/VnLink";
import { useUserContext } from "./UserLayout";

export default function UserRecs() {
  const { t } = useI18n();
  const { user, cat } = useUserContext();
  const recs = user.recs.map((r) => ({ ...r, vn: cat.byIdx[r.idx] })).filter((r) => r.vn);
  return (
    <section className="space-y-3">
      <p className="max-w-3xl text-sm text-ink-2">{t("user.recsHint", { k: 30 })}</p>
      {recs.length === 0 ? (
        <p className="text-sm text-ink-3">{t("user.noRecs")}</p>
      ) : (
        <ol className="grid gap-2 md:grid-cols-2">
          {recs.map((r, i) => (
            <li key={r.idx} className="grid grid-cols-[1.5rem_1fr_auto] items-center gap-3 rounded-lg border border-line bg-surface px-3 py-2.5">
              <span className="tabular text-xs text-ink-3">{i + 1}</span>
              <div className="min-w-0">
                <VnLink vn={r.vn} released={r.vn.released} showSub />
                <div className="text-xs text-ink-3">
                  VNDB {r.vn.rating?.toFixed(2) ?? "—"} · {t("user.support", { n: r.support })}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-ink-3">{t("user.pred")}</div>
                <div className="tabular text-lg font-semibold">{Math.min(10, r.pred).toFixed(1)}</div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
