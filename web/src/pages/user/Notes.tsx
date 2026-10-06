import { useMemo, useState } from "react";
import { useUserNotes, type UserNote } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { NoteCard } from "../../components/NoteCard";
import { useShowMore } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { VnLink } from "../../components/VnLink";
import { LabelChips } from "../vn/Ratings";
import { useUserContext } from "./UserLayout";

const SORTS = ["date", "vote", "sp"] as const;
type Sort = (typeof SORTS)[number];

export default function UserNotes() {
  const { t } = useI18n();
  const { user, cat } = useUserContext();
  const res = useUserNotes(user.uid);
  const [sort, setSort] = useState<Sort>("date");
  const [asc, setAsc] = useState(false);
  const [st, setSt] = useState(0);
  const notes = useMemo(() => {
    if (res.state !== "ok") return [];
    const key = (n: UserNote) => (sort === "date" ? n.date : sort === "vote" ? (n.vote > 0 ? n.vote : null) : n.sp ?? null);
    return res.data.notes
      .filter((n) => cat.byIdx[n.idx] && (!st || (n.labels ?? 0) & st))
      .sort((x, y) => {
        const a = key(x);
        const b = key(y);
        if (a === null || b === null) return a === b ? 0 : a === null ? 1 : -1;
        return asc ? a - b : b - a;
      });
  }, [res, cat, sort, asc, st]);
  const list = (items: UserNote[]) => (
    <ul className="space-y-2">
      {items.map((n) => {
        const vn = cat.byIdx[n.idx];
        return (
          <NoteCard
            key={n.idx}
            head={<VnLink vn={vn} released={vn.released} truncate />}
            vote={n.vote}
            date={n.date}
            text={n.text}
            labels={n.labels}
            extra={n.sp !== null && n.sp !== undefined ? <span title={t("notes.sort.sp")}>{t("notes.top", { p: Math.max(1, 100 - n.sp) })}</span> : null}
          />
        );
      })}
    </ul>
  );
  const [visible, more] = useShowMore(notes, 20, `${t("vn.tab.notes")} (${notes.length})`, list);
  if (res.state !== "ok") return <Status value={res} />;
  const total = res.data.notes.filter((n) => cat.byIdx[n.idx]).length;
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("notes.userHint")}</p>
      {total > 0 && (
        <div className="flex flex-wrap items-end gap-x-4 gap-y-3 rounded-lg border border-line bg-surface p-3">
          <label className="block">
            <span className="mb-1 block text-xs text-ink-2">{t("notes.sort")}</span>
            <span className="flex gap-1">
              <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} className="rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink">
                {SORTS.map((s) => (
                  <option key={s} value={s}>
                    {t(`notes.sort.${s}`)}
                  </option>
                ))}
              </select>
              <button type="button" onClick={() => setAsc(!asc)} className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-ink hover:bg-surface-2" aria-label={t(asc ? "notes.asc" : "notes.desc")} title={t(asc ? "notes.asc" : "notes.desc")}>
                {asc ? "↑" : "↓"}
              </button>
            </span>
          </label>
          <div>
            <span className="mb-1 block text-xs text-ink-2">{t("notes.status")}</span>
            <LabelChips mask={st} onChange={setSt} />
          </div>
        </div>
      )}
      {total > 0 && <p className="text-sm font-medium text-ink tabular">{t("notes.matched", { m: notes.length, n: total })}</p>}
      {notes.length === 0 ? <p className="text-sm text-ink-2">{t("notes.none")}</p> : list(visible)}
      {more}
    </section>
  );
}
