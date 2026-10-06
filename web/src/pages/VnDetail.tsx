import { useMemo, useState } from "react";
import { Link, useParams } from "react-router";
import { useApi, useMeta, type OtherVn, type VnDetail as Detail } from "../lib/api";
import { coverUrl, formatDate, formatInt, formatScore, titles, year } from "../lib/format";
import { langName, methodGroup, methodName, relationName, useI18n, type MethodGroup, type StringKey } from "../lib/i18n";
import { Status } from "../components/Status";

export default function VnDetail() {
  const { id } = useParams();
  const vn = useApi<Detail>(`/api/vn/${Number(id)}`);
  const meta = useMeta();
  if (vn.state !== "ok") return <Status value={vn} />;
  const featured = meta.state === "ok" ? meta.data.info.featured : [];
  return <DetailView vn={vn.data} featured={featured} />;
}

function DetailView({ vn, featured }: { vn: Detail; featured: string[] }) {
  const { t, lang } = useI18n();
  const { main, sub } = titles(vn, lang);
  const others = useMemo(() => new Map(vn.others.map((o) => [o.id, o])), [vn.others]);
  const dev = lang === "zh" ? vn.dev : vn.dev_latin ?? vn.dev;

  const facts: [string, React.ReactNode][] = [
    [t("vn.developer"), vn.dev_id ? <Link className="hover:text-accent-ink hover:underline" to={`/?dev=${vn.dev_id}`}>{dev}</Link> : "—"],
    [t("vn.released"), formatDate(vn.released)],
    [t("rank.olang"), vn.olang ? langName(vn.olang, lang) : "—"],
    [t("vn.votes"), formatInt(vn.votes)],
    [t("vn.rating"), vn.rating?.toFixed(2) ?? "—"],
    [t("vn.average"), vn.average?.toFixed(2) ?? "—"],
  ];

  return (
    <article className="space-y-8">
      <header className="flex flex-col gap-5 sm:flex-row">
        {vn.image && <Cover id={vn.image} sexual={vn.image_sexual} alt={main} />}
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight">{main}</h1>
          {sub && <p className="mt-1 text-ink-2">{sub}</p>}
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {facts.map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs text-ink-3">{k}</dt>
                <dd className="tabular">{v}</dd>
              </div>
            ))}
          </dl>
          <a href={`https://vndb.org/v${vn.id}`} target="_blank" rel="noreferrer" className="mt-4 inline-block text-sm text-accent-ink hover:underline">
            {t("vn.onVndb")} ↗
          </a>
        </div>
      </header>

      <Ranks ranks={vn.ranks} featured={featured} />
      <HeadToHead vn={vn} others={others} />

      {vn.relations.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold">{t("vn.relations")}</h2>
          <ul className="grid gap-2 sm:grid-cols-2">
            {vn.relations.map((r) => {
              const o = others.get(r.id);
              return (
                <li key={r.id} className="rounded-lg border border-line bg-surface px-3 py-2 text-sm">
                  <span className="mr-2 text-xs text-ink-3">{relationName(r.relation, lang)}</span>
                  {o ? <VnLink vn={o} /> : <span>v{r.id}</span>}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </article>
  );
}

function Cover({ id, sexual, alt }: { id: number; sexual: number | null; alt: string }) {
  const { t } = useI18n();
  // Blur anything flagged as more than mildly suggestive (or unflagged) until asked.
  const risky = sexual === null || sexual >= 0.5;
  const [show, setShow] = useState(!risky);
  return (
    <div className="relative aspect-[2/3] w-40 shrink-0 self-start overflow-hidden rounded-lg border border-line bg-surface-2">
      <img src={coverUrl(id)} alt={alt} loading="lazy" referrerPolicy="no-referrer" className={`h-full w-full object-cover ${show ? "" : "scale-110 blur-xl"}`} />
      {!show && (
        <button type="button" onClick={() => setShow(true)} className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-bg/40 p-2 text-center text-xs text-ink">
          <span>{t("vn.nsfwCover")}</span>
          <span className="rounded bg-surface px-2 py-0.5 font-medium">{t("vn.showCover")}</span>
        </button>
      )}
    </div>
  );
}

const GROUPS: MethodGroup[] = ["merged", "po", "rankit", "ref"];

function Ranks({ ranks, featured }: { ranks: Detail["ranks"]; featured: string[] }) {
  const { t, lang } = useI18n();
  const [all, setAll] = useState(false);
  const codes = Object.keys(ranks);
  const shown = featured.filter((m) => ranks[m]);
  return (
    <section>
      <h2 className="mb-3 text-lg font-semibold">{t("vn.ranks")}</h2>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {shown.map((m) => (
          <Link key={m} to={`/?m=${m}`} className="rounded-lg border border-line bg-surface px-3 py-2.5 hover:border-accent">
            <div className="truncate text-xs text-ink-3" title={methodName(m, lang)}>
              {methodName(m, lang)}
            </div>
            <div className="tabular mt-0.5 text-xl font-semibold">#{ranks[m][0]}</div>
          </Link>
        ))}
      </div>
      <button type="button" onClick={() => setAll(!all)} className="mt-3 text-sm text-accent-ink hover:underline" aria-expanded={all}>
        {all ? t("vn.lessMethods") : t("vn.allMethods", { n: codes.length })}
      </button>
      {all && (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          {GROUPS.map((g) => {
            const list = codes.filter((c) => methodGroup(c) === g);
            if (!list.length) return null;
            return (
              <table key={g} className="w-full self-start rounded-lg border border-line bg-surface text-sm">
                <caption className="px-3 pt-2 text-left text-xs font-medium text-ink-3">{t(`methods.group.${g}` as StringKey)}</caption>
                <tbody>
                  {list.map((c) => (
                    <tr key={c} className="border-t border-line first:border-0">
                      <td className="px-3 py-1.5">
                        <Link to={`/?m=${c}`} className="hover:text-accent-ink">
                          {methodName(c, lang)}
                        </Link>
                      </td>
                      <td className="tabular px-3 py-1.5 text-right font-medium">#{ranks[c][0]}</td>
                      <td className="tabular w-24 px-3 py-1.5 text-right text-xs text-ink-3">{formatScore(ranks[c][1])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            );
          })}
        </div>
      )}
    </section>
  );
}

// Same categories and keys as pipeline/vndb_rank/neighbors.py.
const CATEGORIES = ["popular", "ahead", "behind", "contested", "tied"] as const;
type Category = (typeof CATEGORIES)[number];
type Neighbor = Detail["neighbors"][number];

const KEYS: Record<Category, (n: Neighbor) => number> = {
  popular: (n) => n.common,
  ahead: (n) => (n.wins / n.common) * Math.log10(n.common),
  behind: (n) => (n.losses / n.common) * Math.log10(n.common),
  contested: (n) => (n.wins + n.losses > 0 ? (1 - Math.abs(n.wins - n.losses) / (n.wins + n.losses)) * Math.log10(n.common) : 0),
  tied: (n) => ((n.common - n.wins - n.losses) / n.common) * Math.log10(n.common),
};

function HeadToHead({ vn, others }: { vn: Detail; others: Map<number, OtherVn> }) {
  const { t } = useI18n();
  const [cat, setCat] = useState<Category>("popular");
  const list = useMemo(() => [...vn.neighbors].sort((a, b) => KEYS[cat](b) - KEYS[cat](a) || b.common - a.common).slice(0, 10), [vn.neighbors, cat]);

  return (
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
          {list.map((n) => {
            const o = others.get(n.id);
            return <H2HRow key={n.id} n={n} other={o} />;
          })}
        </ul>
      )}
    </section>
  );
}

function H2HRow({ n, other }: { n: Neighbor; other?: OtherVn }) {
  const { t } = useI18n();
  const ties = n.common - n.wins - n.losses;
  const pct = (x: number) => `${(x / n.common) * 100}%`;
  const label = t("vn.h2h.row", { w: n.wins, t: ties, l: n.losses, n: n.common });
  return (
    <li className="rounded-lg border border-line bg-surface px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="min-w-0 truncate">{other ? <VnLink vn={other} /> : `v${n.id}`}</span>
        <span className="tabular shrink-0 text-xs text-ink-3">{label}</span>
      </div>
      {/* wins | ties | losses, with 2px surface gaps between segments */}
      <div className="mt-2 flex h-2 gap-[2px] overflow-hidden rounded-full" role="img" aria-label={label} title={label}>
        {n.wins > 0 && <div style={{ width: pct(n.wins), background: "var(--win)" }} />}
        {ties > 0 && <div style={{ width: pct(ties), background: "var(--tie)" }} />}
        {n.losses > 0 && <div style={{ width: pct(n.losses), background: "var(--loss)" }} />}
      </div>
    </li>
  );
}

function VnLink({ vn }: { vn: OtherVn }) {
  const { lang } = useI18n();
  const { main } = titles(vn, lang);
  const y = year(vn.released);
  return (
    <Link to={`/vn/${vn.id}`} className="hover:text-accent-ink">
      {main}
      {y && <span className="ml-1.5 text-xs text-ink-3">{y}</span>}
    </Link>
  );
}
