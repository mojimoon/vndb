import { useMemo, useState } from "react";
import { Link } from "react-router";
import { useJoint, type CatalogueItem, type VnDetail } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { H2HRow } from "../../components/H2H";
import { Status } from "../../components/Status";
import { Card } from "../../components/VnLink";
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
  const { vn, cat } = useVnContext();
  const [cat_, setCat] = useState<Category>("popular");
  const [pick, setPick] = useState<CatalogueItem | null>(null);
  const list = useMemo(() => [...vn.neighbors].sort((a, b) => KEYS[cat_](b) - KEYS[cat_](a) || b.common - a.common).slice(0, 10), [vn.neighbors, cat_]);

  return (
    <div className="space-y-6">
      <Card title={t("vn.compareAny")}>
        <VnPicker onPick={setPick} exclude={vn.id} />
        {pick && <AnyPair self={vn} other={pick} />}
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
              aria-selected={cat_ === c}
              onClick={() => setCat(c)}
              className={`rounded-full px-3 py-1 text-sm ${cat_ === c ? "bg-ink text-bg" : "bg-surface-2 text-ink-2 hover:text-ink"}`}
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
              <H2HRow key={n.id} self={vn.id} other={cat.byId.get(n.id)} otherId={n.id} wins={n.wins} losses={n.losses} common={n.common} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function AnyPair({ self, other }: { self: VnDetail; other: CatalogueItem }) {
  const { t } = useI18n();
  const joint = useJoint(self.idx, other.idx);
  if (joint.state !== "ok") return <Status value={joint} />;
  const j = joint.data;
  return (
    <div className="mt-3 space-y-2">
      {j.common ? (
        <ul>
          <H2HRow self={self.id} other={other} otherId={other.id} wins={j.wins} losses={j.losses} common={j.common} />
        </ul>
      ) : (
        <p className="text-sm text-ink-3">{t("vn.noCommon")}</p>
      )}
      <Link to={`/compare/vn/${self.id}/${other.id}`} className="inline-block text-sm text-accent-ink hover:underline">
        {t("vn.fullCompare")}
      </Link>
    </div>
  );
}
