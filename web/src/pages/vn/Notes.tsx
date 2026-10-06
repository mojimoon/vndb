import { useState } from "react";
import { useVnNotes, type Note } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { NoteCard, UserNameLink } from "../../components/NoteCard";
import { ShowMoreButton } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { useVnContext } from "./VnLayout";

export default function Notes() {
  const { t } = useI18n();
  const { vn } = useVnContext();
  const [pages, setPages] = useState(1);
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("notes.hint")}</p>
      {vn.analysis.notes ? (
        <ul className="space-y-2">
          {Array.from({ length: pages }, (_, p) => (
            <NotesPage key={p} idx={vn.idx} page={p} last={p === pages - 1} onMore={() => setPages(pages + 1)} />
          ))}
        </ul>
      ) : (
        <p className="text-sm text-ink-3">{t("notes.none")}</p>
      )}
    </section>
  );
}

function NotesPage({ idx, page, last, onMore }: { idx: number; page: number; last: boolean; onMore: () => void }) {
  const res = useVnNotes(idx, page);
  if (res.state !== "ok") return <Status value={res} />;
  const { notes, total, pageSize } = res.data;
  const shown = Math.min(total, (page + 1) * pageSize);
  return (
    <>
      {notes.map((n: Note, i) => (
        <NoteCard key={`${page}-${i}`} head={<UserNameLink uid={n.uid} name={n.name} hasPage={n.hasPage} />} vote={n.vote} date={n.date} text={n.text} />
      ))}
      {last && shown < total && <ShowMoreButton shown={shown} total={total} onMore={onMore} />}
    </>
  );
}
