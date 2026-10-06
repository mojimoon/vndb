import { methodGroup, methodName, useI18n, type MethodGroup } from "../lib/i18n";

const GROUP_ORDER: MethodGroup[] = ["merged", "po", "rankit", "ref"];

export function MethodSelect({
  methods,
  featured,
  value,
  onChange,
  id,
  placeholder,
  compact = false,
}: {
  methods: string[];
  featured: string[];
  value: string;
  onChange: (m: string) => void;
  id?: string;
  placeholder?: string;
  compact?: boolean;
}) {
  const { t, lang } = useI18n();
  const all = methods;
  return (
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full rounded-md border border-line bg-surface text-ink ${compact ? "px-2 py-1 text-xs" : "px-3 py-2 text-sm"}`}
    >
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      <optgroup label={t("rank.featured")}>
        {featured.map((m) => (
          <option key={`f-${m}`} value={m}>
            {methodName(m, lang)}
          </option>
        ))}
      </optgroup>
      {GROUP_ORDER.map((g) => {
        const items = all.filter((m) => methodGroup(m) === g && !featured.includes(m));
        if (!items.length) return null;
        return (
          <optgroup key={g} label={t(`methods.group.${g}`)}>
            {items.map((m) => (
              <option key={m} value={m}>
                {methodName(m, lang)}
              </option>
            ))}
          </optgroup>
        );
      })}
    </select>
  );
}
