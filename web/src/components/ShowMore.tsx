import { useState } from "react";
import { useI18n } from "../lib/i18n";
import { Modal, useIncremental } from "./Modal";

/**
 * First `initial` items inline; when there are more, a "show all" button opens
 * the complete list in a modal. `render` draws a list of items (used for both).
 */
export function useShowMore<T>(items: T[], initial: number, title: React.ReactNode, render: (items: T[]) => React.ReactNode) {
  const [open, setOpen] = useState(false);
  const visible = items.slice(0, initial);
  const more =
    items.length > initial ? (
      <>
        <ShowMoreButton shown={visible.length} total={items.length} onMore={() => setOpen(true)} />
        {open && (
          <Modal title={title} onClose={() => setOpen(false)} wide>
            <AllItems items={items} render={render} />
          </Modal>
        )}
      </>
    ) : null;
  return [visible, more] as const;
}

function AllItems<T>({ items, render }: { items: T[]; render: (items: T[]) => React.ReactNode }) {
  const [shown, sentinel] = useIncremental(items, 100);
  return (
    <>
      {render(shown)}
      {sentinel}
    </>
  );
}

export function ShowMoreButton({ shown, total, onMore }: { shown: number; total: number; onMore: () => void }) {
  const { t } = useI18n();
  return (
    <div className="mt-2 flex items-center justify-center gap-3 text-xs">
      <span className="tabular text-ink-2">
        {shown} / {total}
      </span>
      <button type="button" onClick={onMore} className="rounded-md border border-line bg-surface px-3 py-1 font-medium text-ink hover:bg-surface-2">
        {t("common.showAll", { n: total.toLocaleString() })}
      </button>
    </div>
  );
}
