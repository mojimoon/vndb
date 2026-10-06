import { useMemo, useState } from "react";
import { useCatalogue, type CatalogueItem } from "../lib/api";
import { normalizeQuery, titles, year } from "../lib/format";
import { useI18n } from "../lib/i18n";

/** Search-as-you-type over the catalogue. */
export function VnPicker({ onPick, placeholder, exclude }: { onPick: (vn: CatalogueItem) => void; placeholder?: string; exclude?: number }) {
  const { t, lang } = useI18n();
  const cat = useCatalogue();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const results = useMemo(() => {
    if (cat.state !== "ok" || !q.trim()) return [];
    const nq = normalizeQuery(q);
    const id = /^v?(\d+)$/i.exec(q.trim());
    return cat.data.items
      .filter((it) => it.id !== exclude && ((id && it.id === Number(id[1])) || (nq && it.search.includes(nq))))
      .sort((a, b) => b.votes - a.votes)
      .slice(0, 8);
  }, [cat, q, exclude]);

  return (
    <div className="relative">
      <input
        type="search"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder={placeholder ?? t("vn.pickVn")}
        aria-label={placeholder ?? t("vn.pickVn")}
        className="w-full rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3"
      />
      {open && results.length > 0 && (
        <ul className="absolute z-30 mt-1 max-h-80 w-full overflow-y-auto rounded-md border border-line bg-surface shadow-lg" role="listbox">
          {results.map((it) => {
            const { main, sub } = titles(it, lang);
            return (
              <li key={it.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={false}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => {
                    onPick(it);
                    setQ("");
                    setOpen(false);
                  }}
                  className="block w-full px-3 py-2 text-left text-sm hover:bg-surface-2"
                >
                  <span className="font-medium">{main}</span>
                  <span className="ml-2 text-xs text-ink-3">
                    {sub ? `${sub} · ` : ""}
                    {year(it.released) ?? ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
