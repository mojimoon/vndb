import { ApiError, type Loadable } from "../lib/api";
import { useI18n } from "../lib/i18n";

/** Loading / error placeholder for a Loadable. Returns null once data is ready. */
export function Status({ value }: { value: Loadable<unknown> }) {
  const { t } = useI18n();
  if (value.state === "ok") return null;
  if (value.state === "loading")
    return (
      <div className="flex items-center gap-2 py-16 text-sm text-ink-3" role="status">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
        {t("common.loading")}
      </div>
    );
  const err = value.error;
  const msg =
    err instanceof ApiError && err.status === 503
      ? t("common.nodata")
      : err instanceof ApiError && err.status === 404
        ? t("common.notfound")
        : `${t("common.error")}: ${err.message}`;
  return (
    <div className="my-10 rounded-lg border border-line bg-surface p-6 text-sm text-ink-2" role="alert">
      {msg}
    </div>
  );
}
