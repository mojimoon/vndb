import { useMemo, useState } from "react";
import { Link } from "react-router";
import { usePair, type CatalogueItem, type OtherVn, type VnDetail } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { Card, VnLink } from "../../components/VnLink";
import { VnPicker } from "../../components/VnPicker";
import { useVnContext } from "./VnLayout";

// Same categories and keys as pipeline/vndb_rank/neighbors.py.
const CATEGORIES = ["popular", "ahead", "behind", "contested", "tied"] as const;
type Category = (typeof CATEGORIES)[number];
type Neighbor = VnDetail["neighbors"][number];

const KEYS: Record<Category, (n: Neighbor) => number> = {
  popular: (n) => n.common,
  ahead: (n) => (n.wins / n.common) * Math.log10(n.common),
  behind: (n) => (n.losses / n.common) * Math.log10(n.common),
  contested: (n) => (n.wins + n.losses > 0 ? (1 - Math.abs(n.wins - n.losses) / (n.wins + n.losses)) * Math.log10(n.common) : 0),
  tied: (n) => ((n.common - n.wins - n.losses) / n.common) * Math.log10(n.common),
};

export default function Versus() {
  const { t } = useI18n();
  const { vn, others, meta } = useVnContext();
  const [cat, setCat] = useState<Category>("popular");
  const [pick, setPick] = useState<CatalogueItem | null>(null);
  const list = useMemo(() => [...vn.neighbors].sort((a, b) => KEYS[cat](b) - KEYS[cat](a) || b.common - a.common).slice(0, 10), [vn.neighbors, cat]);

  return (
    <div className="space-y-6">
      <Card title={t("vn.compareAny")}>
        <VnPicker onPick={setPick} exclude={vn.id} />
        {pick && <AnyPair a={vn.id} b={pick} minCommon={meta?.info.config.min_common_vote ?? 5} />}
      </Card>

      <section>
        <h2 className="text-lg font-semibold">{t("vn.h2h")}</h2>
        <p className="mt-1 text-sm text-ink-2">{t("vn.h2hHint")}</p>
        <div className="mt-3 flex flex-wrap gap-1" role="tablist">
          {CATEGORIES.map((c) => (
            <button
              key={c}
              type="button"
              role="tab"
              aria-selected={cat === c}
              onClick={() => setCat(c)}
              className={`rounded-full px-3 py-1 text-sm ${cat === c ? "bg-ink text-bg" : "bg-surface-2 text-ink-2 hover:text-ink"}`}
            >
              {t(`vn.h2h.${c}`)}
            </button>
          ))}
        </div>
        {list.length === 0 ? (
          <p className="mt-4 text-sm text-ink-3">{t("vn.h2h.none")}</p>
        ) : (
          <ul className="mt-4 space-y-2" role="tabpanel">
            {list.map((n) => (
              <H2HRow key={n.id} n={n} other={others.get(n.id)} self={vn.id} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function AnyPair({ a, b, minCommon }: { a: number; b: CatalogueItem; minCommon: number }) {
  const { t } = useI18n();
  const pair = usePair(a, b.id);
  if (pair.state !== "ok") return <Status value={pair} />;
  const p = pair.data;
  return (
    <div className="mt-3 space-y-2">
      {p.common ? (
        <ul>
          <H2HRow n={{ id: b.id, wins: p.wins, losses: p.losses, common: p.common }} other={{ ...b, rank: null }} self={a} />
        </ul>
      ) : (
        <p className="text-sm text-ink-3">{t("vn.belowThreshold", { n: minCommon })}</p>
      )}
      <Link to={`/compare/vn/${a}/${b.id}`} className="inline-block text-sm text-accent-ink hover:underline">
        {t("vn.fullCompare")}
      </Link>
    </div>
  );
}

export function H2HBar({ wins, losses, common }: { wins: number; losses: number; common: number }) {
  const { t } = useI18n();
  const ties = common - wins - losses;
  const pctW = (x: number) => `${(x / common) * 100}%`;
  const label = t("vn.h2h.row", { w: wins, t: ties, l: losses, n: common });
  return (
    <div className="mt-2 flex h-2 gap-[2px] overflow-hidden rounded-full" role="img" aria-label={label} title={label}>
      {wins > 0 && <div style={{ width: pctW(wins), background: "var(--win)" }} />}
      {ties > 0 && <div style={{ width: pctW(ties), background: "var(--tie)" }} />}
      {losses > 0 && <div style={{ width: pctW(losses), background: "var(--loss)" }} />}
    </div>
  );
}

function H2HRow({ n, other, self }: { n: Neighbor; other?: OtherVn; self: number }) {
  const { t } = useI18n();
  const ties = n.common - n.wins - n.losses;
  return (
    <li className="rounded-lg border border-line bg-surface px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        {other ? <VnLink vn={other} released={other.released} className="truncate" /> : <span>v{n.id}</span>}
        <span className="flex shrink-0 items-baseline gap-3">
          <span className="tabular text-xs text-ink-3">{t("vn.h2h.row", { w: n.wins, t: ties, l: n.losses, n: n.common })}</span>
          <Link to={`/compare/vn/${self}/${n.id}`} className="text-xs text-accent-ink hover:underline" aria-label={t("compare.title")}>
            ⇄
          </Link>
        </span>
      </div>
      <H2HBar wins={n.wins} losses={n.losses} common={n.common} />
    </li>
  );
}
