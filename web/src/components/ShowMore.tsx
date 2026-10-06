import { useState } from "react";
import { useI18n } from "../lib/i18n";

/** First `initial` items, growing by `step` with a "show more" button. */
export function useShowMore<T>(items: T[], initial = 10, step = 20) {
  const [n, setN] = useState(initial);
  const visible = items.slice(0, n);
  const more =
    items.length > n ? (
      <ShowMoreButton shown={visible.length} total={items.length} onMore={() => setN(n + step)} onAll={() => setN(items.length)} />
    ) : null;
  return [visible, more] as const;
}

export function ShowMoreButton({ shown, total, onMore, onAll }: { shown: number; total: number; onMore: () => void; onAll?: () => void }) {
  const { t } = useI18n();
  return (
    <div className="mt-2 flex items-center justify-center gap-3 text-xs">
      <span className="tabular text-ink-3">
        {shown} / {total}
      </span>
      <button type="button" onClick={onMore} className="rounded-md border border-line bg-surface px-3 py-1 text-ink-2 hover:bg-surface-2 hover:text-ink">
        {t("common.more")}
      </button>
      {onAll && total - shown > 20 && (
        <button type="button" onClick={onAll} className="text-accent-ink hover:underline">
          {t("common.all")}
        </button>
      )}
    </div>
  );
}
