import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../lib/i18n";

/** Centered dialog over a dimmed page; Escape, the × button or a click outside close it. */
export function Modal({ title, onClose, children, wide = false }: { title: React.ReactNode; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  const { t } = useI18n();
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panel.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [onClose]);
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/45 p-3 pt-[6vh] sm:p-6 sm:pt-[8vh]" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        className={`flex max-h-[86vh] w-full flex-col overflow-hidden rounded-xl border border-line bg-bg shadow-2xl outline-none ${wide ? "max-w-5xl" : "max-w-3xl"}`}
      >
        <header className="flex items-center justify-between gap-3 border-b border-line bg-surface px-4 py-3">
          <h2 className="min-w-0 truncate text-base font-semibold">{title}</h2>
          <button type="button" onClick={onClose} aria-label={t("common.close")} className="rounded-md px-2 py-1 text-lg leading-none text-ink-2 hover:bg-surface-2 hover:text-ink">
            ×
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

/** Renders `step` more items each time the sentinel scrolls into view (long lists in modals). */
export function useIncremental<T>(items: T[], step = 100) {
  const [n, setN] = useState(step);
  const sentinel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = sentinel.current;
    if (!el || n >= items.length) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setN((x) => x + step));
    io.observe(el);
    return () => io.disconnect();
  }, [n, items.length, step]);
  return [items.slice(0, n), n < items.length ? <div ref={sentinel} className="h-8" /> : null] as const;
}
