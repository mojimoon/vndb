import { useEffect } from "react";
import { useI18n } from "../lib/i18n";

/** Sets the document title to "part · part · VNDB SciRanking" while mounted. */
export function Title({ parts }: { parts: (string | null | undefined)[] }) {
  const { t } = useI18n();
  const title = [...parts.filter(Boolean), t("site.name")].join(" - ");
  useEffect(() => {
    document.title = title;
    return () => {
      document.title = t("site.name");
    };
  }, [title, t]);
  return null;
}
