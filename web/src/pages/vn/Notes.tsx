import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { useVnNotes, type Note, type NotesQuery } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { Modal } from "../../components/Modal";
import { NoteCard, UserNameLink } from "../../components/NoteCard";
import { ShowMoreButton } from "../../components/ShowMore";
import { Status } from "../../components/Status";
import { LabelChips } from "./Ratings";
import { useVnContext } from "./VnLayout";

const SORTS = ["date", "vote", "sp"] as const;

/** Reviews (list notes) of one VN; sorting and filtering run in the worker. */
export default function Notes() {
  const { t } = useI18n();
  const { vn } = useVnContext();
  const [params, setParams] = useSearchParams();
  const [all, setAll] = useState(false);
  const sort = (SORTS as readonly string[]).includes(params.get("sort") ?? "") ? (params.get("sort") as NotesQuery["sort"]) : "date";
  const num = (k: string) => (params.get(k) ? Number(params.get(k)) || null : null);
  const q: NotesQuery = { sort, dir: params.get("dir") === "asc" ? "asc" : "desc", st: Number(params.get("st")) || 0, minv: num("minv"), maxv: num("maxv") };
  const set = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "" || v === "0") next.delete(k);
      else next.set(k, v);
    }
    setParams(next, { replace: true });
  };
  const first = useVnNotes(vn.analysis.notes ? vn.idx : null, 0, q);
  const input = "w-20 rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink placeholder:text-ink-3";
  const filtered = q.st || q.minv || q.maxv;

  if (!vn.analysis.notes) return <p className="text-sm text-ink-2">{t("notes.none")}</p>;
  return (
    <section className="space-y-3">
      <p className="text-sm text-ink-2">{t("notes.hint")}</p>
      <div className="space-y-3 rounded-lg border border-line bg-surface p-3">
        <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
          <label className="block">
            <span className="mb-1 block text-xs text-ink-2">{t("notes.sort")}</span>
            <span className="flex gap-1">
              <select value={sort} onChange={(e) => set({ sort: e.target.value === "date" ? null : e.target.value })} className="rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink">
                {SORTS.map((s) => (
                  <option key={s} value={s}>
                    {t(`notes.sort.${s}`)}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => set({ dir: q.dir === "asc" ? null : "asc" })}
                className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-ink hover:bg-surface-2"
                title={t(q.dir === "asc" ? "notes.asc" : "notes.desc")}
                aria-label={t(q.dir === "asc" ? "notes.asc" : "notes.desc")}
              >
                {q.dir === "asc" ? "↑" : "↓"}
              </button>
            </span>
          </label>
          <label className="block">
            <span className="mb-1 block text-xs text-ink-2">{t("notes.userVotes")}</span>
            <span className="flex items-center gap-1">
              <input type="number" inputMode="numeric" min={0} step={10} value={params.get("minv") ?? ""} onChange={(e) => set({ minv: e.target.value })} placeholder="min" aria-label={`${t("notes.userVotes")} min`} className={input} />
              <span className="text-ink-3">–</span>
              <input type="number" inputMode="numeric" min={0} step={10} value={params.get("maxv") ?? ""} onChange={(e) => set({ maxv: e.target.value })} placeholder="max" aria-label={`${t("notes.userVotes")} max`} className={input} />
            </span>
          </label>
          {filtered ? (
            <button type="button" onClick={() => set({ st: null, minv: null, maxv: null })} className="pb-2 text-xs text-accent-ink hover:underline">
              {t("common.reset")}
            </button>
          ) : null}
        </div>
        <div>
          <span className="mb-1 block text-xs text-ink-2">
            {t("notes.status")} <span className="text-ink-3">({t("ratings.statusHint")})</span>
          </span>
          <LabelChips mask={q.st ?? 0} onChange={(m) => set({ st: String(m) })} />
        </div>
      </div>

      {first.state !== "ok" ? (
        <Status value={first} />
      ) : (
        <>
          <p className="text-sm font-medium text-ink tabular">{t("notes.matched", { m: first.data.matched.toLocaleString(), n: first.data.total.toLocaleString() })}</p>
          {first.data.notes.length === 0 ? (
            <p className="text-sm text-ink-2">{t("notes.none")}</p>
          ) : (
            <ul className="space-y-2">
              {first.data.notes.map((n, i) => (
                <VnNote key={i} n={n} />
              ))}
            </ul>
          )}
          {first.data.matched > first.data.notes.length && (
            <ShowMoreButton shown={first.data.notes.length} total={first.data.matched} onMore={() => setAll(true)} />
          )}
          {all && (
            <Modal title={`${t("vn.tab.notes")} · ${t("notes.matched", { m: first.data.matched.toLocaleString(), n: first.data.total.toLocaleString() })}`} onClose={() => setAll(false)} wide>
              <AllNotes idx={vn.idx} q={q} pages={Math.ceil(first.data.matched / first.data.pageSize)} />
            </Modal>
          )}
        </>
      )}
    </section>
  );
}

export function VnNote({ n }: { n: Note }) {
  const { t } = useI18n();
  return (
    <NoteCard
      head={<UserNameLink uid={n.uid} name={n.name} hasPage={n.hasPage} />}
      vote={n.vote}
      date={n.date}
      text={n.text}
      labels={n.labels}
      extra={
        <>
          <span title={t("notes.userVotes")}>{t("notes.votesBy", { n: n.nvotes.toLocaleString() })}</span>
          {n.sp !== null && <span title={t("notes.sort.sp")}>{t("notes.top", { p: Math.max(1, 100 - n.sp) })}</span>}
        </>
      }
    />
  );
}

/** Every page of the query, loaded one page at a time as the list scrolls. */
function AllNotes({ idx, q, pages }: { idx: number; q: NotesQuery; pages: number }) {
  const [n, setN] = useState(1);
  const sentinel = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(0); // pages loaded so far
  useEffect(() => {
    const el = sentinel.current;
    if (!el || n >= pages || ready < n) return;
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setN((x) => Math.min(pages, x + 1)));
    io.observe(el);
    return () => io.disconnect();
  }, [n, pages, ready]);
  return (
    <>
      <ul className="space-y-2">
        {Array.from({ length: n }, (_, p) => (
          <NotesPage key={p} idx={idx} page={p} q={q} onLoad={() => setReady((r) => Math.max(r, p + 1))} />
        ))}
      </ul>
      {n < pages && <div ref={sentinel} className="h-8" />}
    </>
  );
}

function NotesPage({ idx, page, q, onLoad }: { idx: number; page: number; q: NotesQuery; onLoad: () => void }) {
  const res = useVnNotes(idx, page, q);
  const ok = res.state === "ok";
  useEffect(() => {
    if (ok) onLoad();
  }, [ok]); // eslint-disable-line react-hooks/exhaustive-deps
  if (res.state !== "ok") return <Status value={res} />;
  return (
    <>
      {res.data.notes.map((n, i) => (
        <VnNote key={`${page}-${i}`} n={n} />
      ))}
    </>
  );
}
