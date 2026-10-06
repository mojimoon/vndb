import { useState } from "react";
import { Link } from "react-router";
import { useMeta, type Meta } from "../lib/api";
import {
  methodDesc, methodGroup, methodName, methodShort, rankerName, variableName, useI18n,
  RANKER_CODES, VARIABLE_CODES, type Lang, type MethodGroup, type StringKey,
} from "../lib/i18n";
import { Status } from "../components/Status";

const INTRO: Record<Lang, string[]> = {
  zh: [
    "偏序网络（Partial Order Network, PONet）排名最初由 Bangumi 用户 @eyecandy 提出：不看作品本身的分数分布，而是根据同一用户对不同作品的评分来判断作品之间的相对优劣。如果同时给 A 和 B 打分的人里多数认为 A 更好，就认为 A 比 B 好。",
    "对每一对作品 A、B，设有 n 名用户同时评分，其中 x 人给 A 更高分、y 人给 B 更高分。由这些「比赛结果」可以算出多种得分：PONet 系列直接对积分取平均；科学排名（rankit）把每对作品当作一场比赛，用 Massey、Colley、Markov 等经典体育排名方法求解；最后用 Borda 计数合并多种方法的结果。",
    "样本百分位（sample percentile）把每位用户的评分换算成它在该用户全部评分中的位置（0–1），从而消除「有人打分偏高、有人偏低」的影响。",
  ],
  en: [
    "Partial-order-network (PONet) ranking was proposed by Bangumi user @eyecandy. Instead of looking at each title's score distribution, it compares titles through the users who rated both: if most people who voted on A and B rated A higher, A is considered better.",
    "For every pair A, B with n common voters, x rated A higher and y rated B higher. PONet methods average these pairwise results directly; the scientific-ranking methods (rankit) treat each pair as a game and solve it with classic sports rating systems such as Massey, Colley and Markov chains; Borda counts then merge several methods into one list.",
    "Sample percentiles turn each vote into its position within that user's own votes (0–1), which removes the effect of some people scoring generously and others harshly.",
  ],
};

const GROUPS: MethodGroup[] = ["merged", "po", "rankit", "ref"];

export default function Methods() {
  const meta = useMeta();
  const { t, lang } = useI18n();
  if (meta.state !== "ok") return <Status value={meta} />;
  const { info, kendall } = meta.data;
  const all = [...info.methods, ...(info.methods.includes("vndb") ? [] : ["vndb"])];

  return (
    <div className="space-y-10">
      <section className="max-w-3xl space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">{t("methods.title")}</h1>
        {INTRO[lang].map((p) => (
          <p key={p.slice(0, 20)} className="leading-relaxed text-ink-2">
            {p}
          </p>
        ))}
        <dl className="mt-4 grid max-w-lg grid-cols-[1fr_auto] gap-x-6 gap-y-1 rounded-lg border border-line bg-surface p-4 text-sm">
          <dt className="col-span-2 mb-1 text-xs font-medium text-ink-3">{t("methods.params")}</dt>
          <dt className="text-ink-2">{t("methods.minVote")}</dt>
          <dd className="tabular text-right font-medium">{info.config.min_vote}</dd>
          <dt className="text-ink-2">{t("methods.minCommon")}</dt>
          <dd className="tabular text-right font-medium">{info.config.min_common_vote}</dd>
        </dl>
      </section>

      <Heatmap kendall={kendall} />

      {GROUPS.map((g) => {
        const list = all.filter((m) => methodGroup(m) === g);
        if (!list.length) return null;
        if (g === "rankit") return <RankitGrid key={g} methods={list} />;
        return (
          <section key={g}>
            <h2 className="mb-3 text-lg font-semibold">{t(`methods.group.${g}` as StringKey)}</h2>
            <ul className="grid gap-2 md:grid-cols-2">
              {list.map((m) => (
                <li key={m} className="rounded-lg border border-line bg-surface px-4 py-3">
                  <Link to={`/?m=${m}`} className="font-medium hover:text-accent-ink">
                    {methodName(m, lang)}
                  </Link>
                  <code className="ml-2 text-xs text-ink-3">{m}</code>
                  <p className="mt-1 text-sm text-ink-2">{methodDesc(m, lang)}</p>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

function RankitGrid({ methods }: { methods: string[] }) {
  const { t, lang } = useI18n();
  const have = new Set(methods);
  return (
    <section>
      <h2 className="text-lg font-semibold">{t("methods.group.rankit")}</h2>
      <p className="mt-1 mb-3 max-w-3xl text-sm text-ink-2">{t("methods.rankitHint")}</p>
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-2" />
              {VARIABLE_CODES.map((v) => (
                <th key={v} className="px-3 py-2 font-medium">
                  {variableName(v, lang)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {RANKER_CODES.map((r) => (
              <tr key={r} className="border-t border-line">
                <th scope="row" className="whitespace-nowrap px-3 py-2 text-left font-medium">
                  {rankerName(r, lang)}
                </th>
                {VARIABLE_CODES.map((v) => {
                  const code = `${r}_${v}`;
                  return (
                    <td key={v} className="px-3 py-2">
                      {have.has(code) ? (
                        <Link to={`/?m=${code}`} className="font-mono text-xs text-accent-ink hover:underline" title={methodName(code, lang)}>
                          {code}
                        </Link>
                      ) : (
                        <span className="text-ink-3">—</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Heatmap({ kendall }: { kendall: Meta["kendall"] }) {
  const { t, lang } = useI18n();
  const [hover, setHover] = useState<[number, number] | null>(null);
  const { methods, matrix } = kendall;
  const off = matrix.flatMap((row, i) => row.filter((_, j) => j < i));
  const lo = Math.min(...off, 1);
  const frac = (v: number) => (lo < 1 ? Math.max(0, (v - lo) / (1 - lo)) : 1);
  // Sequential single-hue ramp from the lowest observed tau to 1.
  const color = (v: number) => `color-mix(in oklab, var(--seq-1) ${Math.round(frac(v) * 100)}%, var(--seq-0))`;
  const h = hover ? `${methodName(methods[hover[0]], lang)} × ${methodName(methods[hover[1]], lang)}: τ = ${matrix[hover[0]][hover[1]].toFixed(3)}` : null;

  return (
    <section>
      <h2 className="text-lg font-semibold">{t("methods.agreement")}</h2>
      <p className="mt-1 text-sm text-ink-2">{t("methods.agreementHint")}</p>
      <div className="mt-2 h-5 text-xs text-ink-2 tabular" aria-live="polite">
        {h}
      </div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[2px] text-[11px]" onMouseLeave={() => setHover(null)}>
          <tbody>
            {methods.map((m, i) => (
              <tr key={m}>
                <th scope="row" className="whitespace-nowrap pr-2 text-right font-normal text-ink-2">
                  {methodShort(m, lang)} <span className="tabular text-ink-3">{i + 1}</span>
                </th>
                {methods.slice(0, i).map((m2, j) => {
                  const v = matrix[i][j];
                  return (
                    <td
                      key={m2}
                      onMouseEnter={() => setHover([i, j])}
                      className={`tabular h-8 w-9 min-w-9 rounded-[3px] text-center ${hover && hover[0] === i && hover[1] === j ? "outline-2 outline-ink" : ""}`}
                      style={{ background: color(v), color: frac(v) > 0.55 ? "var(--surface)" : "var(--text)" }}
                      title={`${methodName(m, lang)} × ${methodName(m2, lang)}: ${v.toFixed(3)}`}
                    >
                      {v.toFixed(2).replace(/^0/, "")}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <th />
              {methods.slice(0, -1).map((m, j) => (
                <th key={m} scope="col" className="tabular pt-1 font-normal text-ink-3" title={methodName(m, lang)}>
                  {j + 1}
                </th>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center gap-2 text-xs text-ink-3">
        <span className="tabular">τ {lo.toFixed(2)}</span>
        <span className="h-2 w-32 rounded-full" style={{ background: "linear-gradient(to right, var(--seq-0), var(--seq-1))" }} />
        <span>1</span>
      </div>
    </section>
  );
}
