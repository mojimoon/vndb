import { useMemo } from "react";
import { all, useTags, useVnTagTable, type TagDict, type VnTagTable } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { TagAnalysis, type TagRow } from "../../components/Tags";
import type { DevStats } from "./data";

/**
 * What a developer's titles are about. TF: a tag's significance summed over the
 * developer's ranked titles, divided by their number; IDF as on a title page
 * (ln(N / df) over all of VNDB), so tags most titles carry weigh less.
 */
export function DevTags({ d }: { d: DevStats }) {
  const { t } = useI18n();
  const data = all<[TagDict, VnTagTable]>(useTags(), useVnTagTable(true));
  const rows = useMemo<TagRow[]>(() => {
    if (data.state !== "ok" || !d.vns.length) return [];
    const [dict, tab] = data.data;
    const acc = new Map<number, { n: number; sig: number; rated: number; rating: number }>();
    for (const v of d.vns) {
      for (let k = tab.offsets[v.idx]; k < tab.offsets[v.idx + 1]; k++) {
        const a = acc.get(tab.tag[k]) ?? { n: 0, sig: 0, rated: 0, rating: 0 };
        a.n++;
        a.sig += tab.rating[k];
        if (v.rating !== null) {
          a.rated++;
          a.rating += v.rating;
        }
        acc.set(tab.tag[k], a);
      }
    }
    const out: TagRow[] = [];
    for (const [id, a] of acc) {
      const info = dict.byId.get(id);
      if (!info) continue;
      out.push({ id, name: info.name, cat: info.cat, spoiler: 0, freq: a.n, sig: a.sig / a.n, mean: a.rated ? a.rating / a.rated : undefined, score: (a.sig / d.vns.length) * info.idf });
    }
    return out;
  }, [data, d]);
  if (data.state === "error") return null; // a snapshot without tags
  if (data.state !== "ok") return <Status value={data} />;
  return <TagAnalysis rows={rows} kind="dev" hint={t("tags.hintDev", { n: d.vns.length.toLocaleString() })} />;
}
