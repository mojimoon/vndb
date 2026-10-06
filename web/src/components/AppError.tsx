import { useRouteError } from "react-router";
import { useI18n } from "../lib/i18n";

/** Shown instead of React Router's default crash screen. */
export default function AppError() {
  const err = useRouteError();
  const { t } = useI18n();
  console.error(err);
  return (
    <div className="mx-auto max-w-lg px-4 py-24 text-center">
      <h1 className="text-xl font-semibold">{t("common.crash")}</h1>
      <p className="mt-2 text-sm text-ink-2">{t("common.crashHint")}</p>
      <button type="button" onClick={() => location.reload()} className="mt-6 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white">
        {t("common.reload")}
      </button>
      <p className="mt-6 break-words text-xs text-ink-3">{err instanceof Error ? err.message : String(err)}</p>
    </div>
  );
}
