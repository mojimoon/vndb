import { Link } from "react-router";
import type { CatalogueItem } from "../lib/api";
import { titles, year } from "../lib/format";
import { useI18n } from "../lib/i18n";

/** Wins / draws / losses as one bar with the shares written inside. */
export function H2HBar({ wins, losses, common, tall = false }: { wins: number; losses: number; common: number; tall?: boolean }) {
  const { t } = useI18n();
  const ties = common - wins - losses;
  const segs = [
    { n: wins, bg: "var(--win)", fg: "#fff", key: "w" },
    { n: ties, bg: "var(--tie)", fg: "var(--text)", key: "d" },
    { n: losses, bg: "var(--loss)", fg: "#fff", key: "l" },
  ];
  const label = t("vn.h2h.row", { w: wins, t: ties, l: losses, n: common });
  return (
    <div className={`flex gap-[2px] overflow-hidden rounded-md ${tall ? "h-7" : "h-5"}`} role="img" aria-label={label} title={label}>
      {segs.map((s) =>
        s.n > 0 ? (
          <div
            key={s.key}
            className="flex items-center justify-center overflow-hidden text-[11px] font-medium tabular"
            style={{ width: `${(s.n / common) * 100}%`, background: s.bg, color: s.fg }}
          >
            {s.n / common >= 0.09 ? `${Math.round((s.n / common) * 100)}%` : ""}
          </div>
        ) : null,
      )}
    </div>
  );
}

/** A clickable head-to-head row: opens the full comparison of the two titles. */
export function H2HRow({ self, other, wins, losses, common }: { self: number; other: CatalogueItem | undefined; otherId: number; wins: number; losses: number; common: number }) {
  const { t, lang } = useI18n();
  const ties = common - wins - losses;
  const id = other?.id;
  const name = other ? titles(other, lang).main : `v${id}`;
  return (
    <li>
      <Link to={`/compare/vn/${self}/${id}`} className="block rounded-lg border border-line bg-surface px-3 py-2.5 hover:border-accent hover:bg-surface-2/60">
        <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
          <span className="min-w-0 truncate font-medium">
            {name}
            {other && year(other.released) && <span className="ml-1.5 text-xs font-normal text-ink-3">{year(other.released)}</span>}
          </span>
          <span className="tabular shrink-0 text-xs text-ink-3">{t("vn.h2h.row", { w: wins, t: ties, l: losses, n: common })}</span>
        </div>
        <H2HBar wins={wins} losses={losses} common={common} />
      </Link>
    </li>
  );
}
