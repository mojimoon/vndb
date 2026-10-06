import { useState } from "react";
import { Link } from "react-router";
import { useI18n } from "../lib/i18n";

const LONG = 400;

/** One list note ("short review"): author / title line, vote, list status, date, expandable text. */
export function NoteCard({ head, vote, date, text, labels = 0, extra }: { head: React.ReactNode; vote: number; date: number; text: string; labels?: number; extra?: React.ReactNode }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const long = text.length > LONG;
  const d = date ? `${Math.floor(date / 10000)}-${String(Math.floor(date / 100) % 100).padStart(2, "0")}-${String(date % 100).padStart(2, "0")}` : "";
  return (
    <li className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-sm">
        <span className="min-w-0 truncate font-medium">{head}</span>
        <span className="flex shrink-0 flex-wrap items-baseline gap-x-2 gap-y-1 text-xs text-ink-2 tabular">
          {[1, 2, 3, 4, 5, 6]
            .filter((k) => labels & (1 << (k - 1)))
            .map((k) => (
              <span key={k} className="rounded border border-line px-1.5 py-px">
                {t(`label.${k}` as never)}
              </span>
            ))}
          {extra}
          {vote > 0 && <span className="rounded bg-accent-soft px-1.5 py-0.5 font-semibold text-ink">{vote / 10}</span>}
          {d}
        </span>
      </div>
      <p className="whitespace-pre-line break-words text-sm leading-relaxed text-ink">{long && !open ? `${text.slice(0, LONG)}…` : text}</p>
      {long && (
        <button type="button" onClick={() => setOpen(!open)} className="mt-1 text-xs text-accent-ink hover:underline">
          {open ? "−" : "+"}
        </button>
      )}
    </li>
  );
}

export const UserNameLink = ({ uid, name, hasPage }: { uid: number; name: string; hasPage: boolean }) =>
  hasPage ? (
    <Link to={`/user/${uid}`} className="hover:text-accent-ink">
      {name || `u${uid}`}
    </Link>
  ) : (
    <a href={`https://vndb.org/u${uid}`} target="_blank" rel="noreferrer" className="hover:text-accent-ink">
      {name || `u${uid}`} ↗
    </a>
  );
