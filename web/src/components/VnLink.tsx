import { Link } from "react-router";
import type { TitleFields } from "../lib/api";
import { titles, year } from "../lib/format";
import { useI18n } from "../lib/i18n";

export function VnLink({ vn, released, showSub = false, className = "" }: { vn: TitleFields; released?: number | null; showSub?: boolean; className?: string }) {
  const { lang } = useI18n();
  const { main, sub } = titles(vn, lang);
  const y = year(released);
  return (
    <span className={`min-w-0 ${className}`}>
      <Link to={`/vn/${vn.id}`} className="font-medium hover:text-accent-ink">
        {main}
      </Link>
      {y && <span className="ml-1.5 text-xs text-ink-3">{y}</span>}
      {showSub && sub && <span className="block truncate text-xs text-ink-3">{sub}</span>}
    </span>
  );
}

export function Card({ title, children, className = "", aside }: { title?: string; children: React.ReactNode; className?: string; aside?: React.ReactNode }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-4 ${className}`}>
      {(title || aside) && (
        <div className="mb-2 flex items-baseline justify-between gap-3">
          {title && <h2 className="text-sm font-semibold">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3" title={hint}>
      <div className="text-xs text-ink-3">{label}</div>
      <div className="tabular mt-1 text-xl font-semibold">{value}</div>
    </div>
  );
}
