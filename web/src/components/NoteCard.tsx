import { useState } from "react";
import { Link } from "react-router";

const LONG = 400;

/** One list note ("short review"): author / title line, vote, date, expandable text. */
export function NoteCard({ head, vote, date, text }: { head: React.ReactNode; vote: number; date: number; text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > LONG;
  const d = date ? `${Math.floor(date / 10000)}-${String(Math.floor(date / 100) % 100).padStart(2, "0")}-${String(date % 100).padStart(2, "0")}` : "";
  return (
    <li className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-sm">
        <span className="min-w-0 truncate font-medium">{head}</span>
        <span className="flex shrink-0 items-baseline gap-3 text-xs text-ink-3 tabular">
          {vote > 0 && <span className="rounded bg-surface-2 px-1.5 py-0.5 font-semibold text-ink">{vote / 10}</span>}
          {d}
        </span>
      </div>
      <p className="whitespace-pre-line break-words text-sm leading-relaxed text-ink-2">{long && !open ? `${text.slice(0, LONG)}…` : text}</p>
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
