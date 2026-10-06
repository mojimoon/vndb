import { useMemo } from "react";
import { useSearchParams } from "react-router";
import { useVoters, type Analysis, type Voters } from "../../lib/api";
import { decileLabels, pct, scoreLabels } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { BarChart, HBars, cumulativeLine } from "../../components/Charts";
import { Status } from "../../components/Status";
import { Card, Stat } from "../../components/VnLink";
import { useVnContext } from "./VnLayout";

const LABELS = [1, 2, 3, 4, 5, 6];

interface VoterFilter {
  st: number; // list-label mask, any of
  vmin: number | null;
  vmax: number | null;
  cvmin: number | null;
  cvmax: number | null;
  csmin: number | null;
  csmax: number | null;
}
const KEYS = ["st", "vmin", "vmax", "cvmin", "cvmax", "csmin", "csmax"] as const;

function readFilter(p: URLSearchParams): VoterFilter {
  const num = (k: string) => {
    const v = p.get(k);
    return v === null || v === "" || Number.isNaN(Number(v)) ? null : Number(v);
  };
  return { st: Number(p.get("st")) || 0, vmin: num("vmin"), vmax: num("vmax"), cvmin: num("cvmin"), cvmax: num("cvmax"), csmin: num("csmin"), csmax: num("csmax") };
}

const isActive = (f: VoterFilter) => f.st !== 0 || KEYS.slice(1).some((k) => f[k] !== null);

/** The precomputed per-VN analysis, recomputed over the voters that pass the filter. */
function analyse(v: Voters, f: VoterFilter): Analysis {
  const inRange = (x: number, lo: number | null, hi: number | null) => (lo === null || x >= lo) && (hi === null || x <= hi);
  const hist = Array(10).fill(0);
  const spHist = Array(10).fill(0);
  const labels: Record<string, number> = {};
  const years = new Map<number, [number, number]>();
  let n = 0;
  let s1 = 0;
  let s2 = 0;
  let sp = 0;
  let bias = 0;
  for (let i = 0; i < v.n; i++) {
    if (f.st && !(v.labels[i] & f.st)) continue;
    if (!inRange(v.nvotes[i], f.vmin, f.vmax)) continue;
    if ((f.cvmin !== null || f.cvmax !== null) && (Number.isNaN(v.cv[i]) || !inRange(v.cv[i], f.cvmin, f.cvmax))) continue;
    if ((f.csmin !== null || f.csmax !== null) && (Number.isNaN(v.cs[i]) || !inRange(v.cs[i], f.csmin, f.csmax))) continue;
    const x = v.vote[i] / 10;
    n++;
    s1 += x;
    s2 += x * x;
    hist[Math.min(10, Math.max(1, Math.floor(x))) - 1]++;
    sp += v.sp[i] / 200;
    spHist[Math.min(9, Math.floor(v.sp[i] / 20))]++;
    bias += (v.vote[i] - v.umean[i]) / 10;
    for (const k of LABELS) if (v.labels[i] & (1 << (k - 1))) labels[k] = (labels[k] ?? 0) + 1;
    if (v.year[i]) {
      const e = years.get(v.year[i]) ?? [0, 0];
      years.set(v.year[i], [e[0] + 1, e[1] + x]);
    }
  }
  const mean = n ? s1 / n : null;
  return {
    n,
    hist,
    mean,
    std: n && mean !== null ? Math.sqrt(Math.max(0, s2 / n - mean * mean)) : null,
    years: [...years.entries()].sort((a, b) => a[0] - b[0]).map(([y, [c, s]]) => [y, c, Math.round((s / c) * 100) / 100]),
    labels,
    sp: n ? { mean: sp / n, hist: spHist } : null,
    bias: n ? bias / n : null,
  };
}

export default function Ratings() {
  const { vn } = useVnContext();
  const [params, setParams] = useSearchParams();
  const f = readFilter(params);
  const active = isActive(f);
  const voters = useVoters(active ? vn.idx : null);
  const filtered = useMemo(() => (active && voters.state === "ok" ? analyse(voters.data, f) : null), [active, voters, params]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (patch: Partial<Record<(typeof KEYS)[number], string | null>>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "" || v === "0") next.delete(k);
      else next.set(k, v);
    }
    setParams(next, { replace: true });
  };

  return (
    <div className="space-y-4">
      <FilterBar f={f} raw={(k) => params.get(k) ?? ""} set={set} reset={() => set(Object.fromEntries(KEYS.map((k) => [k, null])))} active={active} />
      {active && voters.state !== "ok" ? (
        <Status value={voters} />
      ) : (
        <Body a={filtered ?? vn.analysis} total={vn.analysis.n} rating={vn.rating} filtered={!!filtered} />
      )}
    </div>
  );
}

function FilterBar({ f, raw, set, reset, active }: { f: VoterFilter; raw: (k: string) => string; set: (p: Partial<Record<(typeof KEYS)[number], string | null>>) => void; reset: () => void; active: boolean }) {
  const { t } = useI18n();
  const input = "w-full min-w-0 rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink placeholder:text-ink-3";
  const range = (label: string, lo: (typeof KEYS)[number], hi: (typeof KEYS)[number], step: number, min?: number, max?: number) => (
    <label className="block">
      <span className="mb-1 block text-xs text-ink-2">{label}</span>
      <span className="flex items-center gap-1">
        <input type="number" inputMode="decimal" step={step} min={min} max={max} value={raw(lo)} onChange={(e) => set({ [lo]: e.target.value })} placeholder="min" aria-label={`${label} min`} className={input} />
        <span className="text-ink-3">–</span>
        <input type="number" inputMode="decimal" step={step} min={min} max={max} value={raw(hi)} onChange={(e) => set({ [hi]: e.target.value })} placeholder="max" aria-label={`${label} max`} className={input} />
      </span>
    </label>
  );
  return (
    <Card
      title={t("ratings.filter")}
      aside={
        active ? (
          <button type="button" onClick={reset} className="text-xs text-accent-ink hover:underline">
            {t("common.reset")}
          </button>
        ) : undefined
      }
    >
      <div className="space-y-3">
        <div>
          <span className="mb-1 block text-xs text-ink-2">
            {t("notes.status")} <span className="text-ink-3">({t("ratings.statusHint")})</span>
          </span>
          <LabelChips mask={f.st} onChange={(m) => set({ st: String(m) })} />
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {range(t("ratings.userVotes"), "vmin", "vmax", 10, 0)}
          {range(t("ratings.corrVndb"), "cvmin", "cvmax", 0.1, -1, 1)}
          {range(t("ratings.corrSci"), "csmin", "csmax", 0.1, -1, 1)}
        </div>
        <p className="text-xs text-ink-2">{t("ratings.filterHint")}</p>
      </div>
    </Card>
  );
}

/** Toggle chips for the six VNDB list labels, as a bit mask. */
export function LabelChips({ mask, onChange }: { mask: number; onChange: (mask: number) => void }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-wrap gap-1.5">
      {LABELS.map((k) => {
        const bit = 1 << (k - 1);
        const on = (mask & bit) !== 0;
        return (
          <button
            key={k}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(mask ^ bit)}
            className={`rounded-full border px-3 py-1 text-xs ${on ? "border-accent bg-accent-soft font-medium text-ink" : "border-line bg-surface text-ink-2 hover:text-ink"}`}
          >
            {t(`label.${k}` as never)}
          </button>
        );
      })}
    </div>
  );
}

function Body({ a, total, rating, filtered }: { a: Analysis; total: number; rating: number | null; filtered: boolean }) {
  const { t } = useI18n();
  const labels = LABELS.map((k) => ({ label: t(`label.${k}` as never), value: a.labels[String(k)] ?? 0 })).filter((d) => d.value > 0);
  if (filtered && a.n === 0) return <p className="py-8 text-center text-sm text-ink-2">{t("ratings.noMatch")}</p>;
  return (
    <>
      {filtered && <p className="text-sm font-medium text-ink">{t("ratings.matched", { m: a.n.toLocaleString(), n: total.toLocaleString() })}</p>}
      <section className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label={t("vn.ratings.count")} value={a.n.toLocaleString()} />
        <Stat label={t("vn.meanStd")} value={a.mean !== null && a.std !== null ? `${a.mean.toFixed(2)} ± ${a.std.toFixed(2)}` : "—"} />
        <Stat label={t("vn.ratings.bias")} value={a.bias !== null ? (a.bias > 0 ? "+" : "") + a.bias.toFixed(2) : "—"} />
        <Stat label={t("vn.rating")} value={rating?.toFixed(2) ?? "—"} />
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={t("vn.ratings.dist")}>
          <BarChart data={a.hist.map((y, i) => ({ x: scoreLabels[i], y }))} label={t("vn.ratings.dist")} barName={t("vn.votes")} line={cumulativeLine(a.hist, t("chart.cumulative"))} />
        </Card>
        {a.sp && (
          <Card title={t("vn.ratings.sp")}>
            <p className="mb-2 text-xs text-ink-2">{t("vn.ratings.spHint", { p: pct(a.sp.mean) })}</p>
            <BarChart data={a.sp.hist.map((y, i) => ({ x: decileLabels[i], y }))} label={t("vn.ratings.sp")} barName={t("vn.votes")} line={cumulativeLine(a.sp.hist, t("chart.cumulative"))} />
          </Card>
        )}
        {a.years.length > 1 && (
          <Card title={t("vn.ratings.byYear")} className="lg:col-span-2">
            <BarChart
              data={a.years.map(([y, n]) => ({ x: String(y), y: n }))}
              label={t("vn.ratings.byYear")}
              barName={t("stats.count")}
              line={{ name: t("user.mean"), values: a.years.map(([, , m]) => m), format: (v) => v.toFixed(2) }}
              height={200}
            />
          </Card>
        )}
        <Card title={t("vn.ratings.labels")}>
          <HBars data={labels} />
        </Card>
        {a.bias !== null && (
          <Card title={t("vn.ratings.bias")}>
            <p className="text-sm text-ink-2">{t("vn.ratings.biasHint", { b: (a.bias > 0 ? "+" : "") + a.bias.toFixed(2) })}</p>
            <BiasGauge value={a.bias} />
          </Card>
        )}
      </div>
    </>
  );
}

/** -3 .. +3 scale with the value marked; diverging from a neutral zero. */
function BiasGauge({ value }: { value: number }) {
  const clamp = Math.max(-3, Math.min(3, value));
  const pos = ((clamp + 3) / 6) * 100;
  return (
    <div className="mt-4">
      <div className="relative h-2 rounded-full" style={{ background: "linear-gradient(to right, var(--loss), var(--tie) 50%, var(--win))" }}>
        <span className="absolute -top-1.5 h-5 w-1 -translate-x-1/2 rounded bg-ink" style={{ left: `${pos}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-ink-2 tabular">
        <span>−3</span>
        <span>0</span>
        <span>+3</span>
      </div>
    </div>
  );
}
