import { useState } from "react";
import { Link, Outlet, useOutletContext, useParams } from "react-router";
import { all, useCatalogue, useMeta, useVn, type Catalogue, type CatalogueItem, type Meta, type VnDetail } from "../../lib/api";
import { coverUrl, devName, formatDate, formatInt, titles } from "../../lib/format";
import { langName, useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { Tabs } from "../../components/Tabs";

export interface VnContext {
  vn: VnDetail;
  meta: Meta;
  cat: Catalogue;
}
export const useVnContext = () => useOutletContext<VnContext>();

export default function VnLayout() {
  const { id } = useParams();
  const vn = useVn(Number(id));
  const meta = useMeta();
  const cat = useCatalogue();
  const { t } = useI18n();
  const both = all<[VnDetail, Meta, Catalogue]>(vn, meta, cat);
  if (both.state !== "ok") return <Status value={both} />;
  const [v, m, c] = both.data;
  const base = `/vn/${v.id}`;
  return (
    <article className="space-y-6">
      <Header vn={v} item={c.byId.get(v.id)} />
      <Tabs
        base={base}
        tabs={[
          { path: "", label: t("vn.tab.overview") },
          { path: "ratings", label: t("vn.tab.ratings") },
          { path: "ranks", label: t("vn.tab.ranks") },
          { path: "versus", label: t("vn.tab.versus") },
          { path: "similar", label: t("vn.tab.similar") },
          { path: "notes", label: `${t("vn.tab.notes")}${v.analysis.notes ? ` (${v.analysis.notes})` : ""}` },
        ]}
      />
      <Outlet context={{ vn: v, meta: m, cat: c } satisfies VnContext} />
    </article>
  );
}

function Header({ vn, item }: { vn: VnDetail; item?: CatalogueItem }) {
  const { t, lang } = useI18n();
  const { main, sub } = titles(vn, lang);
  const dev = item ? devName(item, lang) : null;
  const facts: [string, React.ReactNode][] = [
    [t("vn.developer"), vn.dev_id ? <Link className="hover:text-accent-ink hover:underline" to={`/dev/${vn.dev_id}`}>{dev}</Link> : "—"],
    [t("vn.released"), formatDate(vn.released)],
    [t("rank.olang"), vn.olang ? langName(vn.olang, lang) : "—"],
    [t("vn.votes"), formatInt(vn.votes)],
    [t("vn.rating"), vn.rating?.toFixed(2) ?? "—"],
    [t("rank.col.length"), vn.length ? t(`length.${vn.length}` as never) : "—"],
  ];
  return (
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
        <div className="mt-4 flex flex-wrap gap-4 text-sm">
          <a href={`https://vndb.org/v${vn.id}`} target="_blank" rel="noreferrer" className="text-accent-ink hover:underline">
            {t("vn.onVndb")} ↗
          </a>
          <Link to={`/compare?type=vn&a=${vn.id}`} className="text-accent-ink hover:underline">
            {t("compare.title")} →
          </Link>
        </div>
      </div>
    </header>
  );
}

function Cover({ id, sexual, alt }: { id: number; sexual: number | null; alt: string }) {
  const { t } = useI18n();
  // Blur anything flagged as more than mildly suggestive (or unflagged) until asked.
  const risky = sexual === null || sexual >= 0.5;
  const [show, setShow] = useState(!risky);
  return (
    <div className="relative aspect-[2/3] w-36 shrink-0 self-start overflow-hidden rounded-lg border border-line bg-surface-2 sm:w-40">
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
