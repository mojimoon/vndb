import { useMemo } from "react";
import { all, useTags, useVnTagTable, type TagDict, type VnTagTable } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { TagAnalysis, type TagRow } from "../../components/Tags";
import { useUserContext } from "./UserLayout";

/**
 * The user's taste in tags. TF: a tag's significance summed over the user's
 * voted titles, divided by the number of votes. IDF is weighted by votes
 * (ln(V / V_t) over ranked titles), so tags that most lists share weigh less.
 */
export default function UserTags() {
  const { t } = useI18n();
  const { votes } = useUserContext();
  const data = all<[TagDict, VnTagTable]>(useTags(), useVnTagTable(true));
  const rows = useMemo<TagRow[]>(() => {
    if (data.state !== "ok" || !votes.length) return [];
    const [dict, tab] = data.data;
    const acc = new Map<number, { n: number; sig: number; vote: number }>();
    for (const v of votes) {
      const i = v.vn.idx;
      for (let k = tab.offsets[i]; k < tab.offsets[i + 1]; k++) {
        const a = acc.get(tab.tag[k]) ?? { n: 0, sig: 0, vote: 0 };
        a.n++;
        a.sig += tab.rating[k];
        a.vote += v.vote;
        acc.set(tab.tag[k], a);
      }
    }
    const out: TagRow[] = [];
    for (const [id, a] of acc) {
      const info = dict.byId.get(id);
      if (!info || info.idfVotes === null) continue;
      out.push({ id, name: info.name, cat: info.cat, spoiler: 0, freq: a.n, sig: a.sig / a.n, mean: a.vote / a.n, score: (a.sig / votes.length) * info.idfVotes });
    }
    return out;
  }, [data, votes]);
  if (data.state !== "ok") return <Status value={data} />;
  return <TagAnalysis rows={rows} kind="user" hint={t("tags.hintUser", { n: votes.length.toLocaleString() })} />;
}
