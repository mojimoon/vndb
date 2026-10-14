import { useMemo } from "react";
import { useTags } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { Status } from "../../components/Status";
import { TagAnalysis, type TagRow } from "../../components/Tags";
import { useVnContext } from "./VnLayout";

/** The title's VNDB tags weighted by TF-IDF: significance x ln(N / VNs with the tag). */
export default function VnTags() {
  const { t } = useI18n();
  const { vn } = useVnContext();
  const dict = useTags();
  const rows = useMemo<TagRow[]>(() => {
    if (dict.state !== "ok") return [];
    return vn.tags.flatMap(([id, sig, spoiler, voters]) => {
      const info = dict.data.byId.get(id);
      return info ? [{ id, name: info.name, cat: info.cat, spoiler, freq: info.df, sig, voters, score: sig * info.idf }] : [];
    });
  }, [vn, dict]);
  if (dict.state !== "ok") return <Status value={dict} />;
  return <TagAnalysis rows={rows} kind="vn" hint={t("tags.hintVn", { n: dict.data.nVns.toLocaleString() })} />;
}
