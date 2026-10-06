import { useUserNotes } from "../../lib/api";
import { titles } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { NoteCard } from "../../components/NoteCard";
import { useShowMore } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { VnLink } from "../../components/VnLink";
import { useUserContext } from "./UserLayout";

export default function UserNotes() {
  const { t, lang } = useI18n();
  const { user, cat } = useUserContext();
  const res = useUserNotes(user.uid);
  const notes = res.state === "ok" ? res.data.notes.filter((n) => cat.byIdx[n.idx]) : [];
  const [visible, more] = useShowMore(notes, 20, 30);
  if (res.state !== "ok") return <Status value={res} />;
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("notes.userHint")}</p>
      {notes.length === 0 ? (
        <p className="text-sm text-ink-3">{t("notes.none")}</p>
      ) : (
        <ul className="space-y-2">
          {visible.map((n) => {
            const vn = cat.byIdx[n.idx];
            return <NoteCard key={n.idx} head={<span title={titles(vn, lang).main}><VnLink vn={vn} released={vn.released} /></span>} vote={n.vote} date={n.date} text={n.text} />;
          })}
        </ul>
      )}
      {more}
    </section>
  );
}
