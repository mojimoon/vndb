import { useNavigate, useSearchParams } from "react-router";
import { useCatalogue, type CatalogueItem } from "../../lib/api";
import { titles } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { VnPicker } from "../../components/VnPicker";
import { UserInput } from "../user/UserSearch";

export default function ComparePicker() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const type = params.get("type") === "user" ? "user" : "vn";
  const cat = useCatalogue();
  const a = Number(params.get("a")) || null;
  const b = Number(params.get("b")) || null;
  const set = (k: "a" | "b", v: number | null) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, String(v));
    else next.delete(k);
    setParams(next, { replace: true });
    const other = k === "a" ? b : a;
    if (v && other) navigate(`/compare/${type}/${k === "a" ? v : other}/${k === "a" ? other : v}`);
  };
  const vnName = (id: number | null) => {
    if (!id) return null;
    const it: CatalogueItem | undefined = cat.state === "ok" ? cat.data.byId.get(id) : undefined;
    return it ? titles(it, lang).main : `v${id}`;
  };

  return (
    <div className="mx-auto max-w-2xl space-y-5 py-4">
      <h1 className="text-2xl font-semibold tracking-tight">{t("compare.title")}</h1>
      <p className="text-sm text-ink-2">{t("compare.hint")}</p>
      <div className="inline-flex rounded-md border border-line bg-surface p-0.5 text-sm">
        {(["vn", "user"] as const).map((k) => (
          <button key={k} type="button" aria-pressed={type === k} onClick={() => setParams({ type: k }, { replace: true })} className={`rounded px-3 py-1 ${type === k ? "bg-surface-2 text-ink" : "text-ink-2"}`}>
            {t(k === "vn" ? "compare.vn" : "compare.user")}
          </button>
        ))}
      </div>
      {(["a", "b"] as const).map((k) => {
        const v = k === "a" ? a : b;
        return (
          <div key={k} className="space-y-1.5">
            <div className="text-xs font-medium text-ink-3">{t(k === "a" ? "compare.pickA" : "compare.pickB")}</div>
            {v ? (
              <div className="flex items-center justify-between rounded-md border border-line bg-surface px-3 py-2 text-sm">
                <span className="font-medium">{type === "vn" ? vnName(v) : `u${v}`}</span>
                <button type="button" onClick={() => set(k, null)} className="text-ink-3 hover:text-ink" aria-label={t("rank.clear")}>
                  ×
                </button>
              </div>
            ) : type === "vn" ? (
              <VnPicker onPick={(it) => set(k, it.id)} exclude={k === "a" ? b ?? undefined : a ?? undefined} />
            ) : (
              <UserInput onResolved={(uid) => set(k, uid)} />
            )}
          </div>
        );
      })}
    </div>
  );
}
