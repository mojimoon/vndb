import { useLayoutEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";

/** Width of the chart's container in CSS pixels, so SVG text keeps its size
 *  instead of scaling with the card. */
function useWidth(fallback = 600) {
  const ref = useRef<HTMLElement>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** 1234 -> "1.2k", 0.153 -> "0.15" for axis ticks. */
export function compact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
  if (Number.isInteger(v)) return String(v);
  return String(+v.toFixed(2));
}

/** ~3 round tick values from 0 to max (inclusive of a round top). */
function niceTicks(max: number): number[] {
  if (max <= 0) return [0];
  const raw = max / 3;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) ?? raw;
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
  return ticks;
}

function YAxis({ ticks, y, x, format = compact, width }: { ticks: number[]; y: (v: number) => number; x: number; format?: (v: number) => string; width: number }) {
  return (
    <g>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={x} x2={width} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeWidth={1} />
          <text x={x - 6} y={y(v) + 4} textAnchor="end" fontSize={11} fill="var(--text-2)">
            {format(v)}
          </text>
        </g>
      ))}
    </g>
  );
}

/** Single-series bar chart with a hover readout. Values are magnitudes, so the
 *  y axis starts at zero; bars get 4px rounded tops and 2px gaps. */
export interface SecondaryLine {
  name: string;
  values: (number | null)[];
  format: (v: number) => string;
  /** Fixed axis range; fitted to the values when omitted. */
  domain?: [number, number];
}

/** Bars on the left axis, optionally with a second series drawn as a line on
 *  its own right axis (secondary color, labelled, with a legend). */
export function BarChart({
  data,
  format = (v) => v.toLocaleString(),
  height = 180,
  label,
  barName,
  line,
}: {
  data: { x: string; y: number }[];
  format?: (v: number) => string;
  height?: number;
  label: string;
  barName?: string;
  line?: SecondaryLine;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth();
  const H = height;
  const padL = 40;
  const padR = line ? 44 : 0;
  const padB = 22;
  const padT = 8;
  const ticks = niceTicks(Math.max(1, ...data.map((d) => d.y)));
  const max = ticks[ticks.length - 1];
  const y = (v: number) => padT + (H - padB - padT) * (1 - v / max);
  const bw = (W - padL - padR) / data.length;
  const h = hover !== null ? data[hover] : null;
  // Label every bar when there are few, otherwise about 8 labels.
  const every = data.length <= 12 ? 1 : Math.ceil(data.length / 8);

  // Secondary axis
  const lv = line?.values.filter((v): v is number => v !== null) ?? [];
  let [lo, hi] = line?.domain ?? [Math.min(...lv), Math.max(...lv)];
  if (!line?.domain && lv.length) {
    const pad = (hi - lo) * 0.1 || 0.5;
    [lo, hi] = [Math.floor((lo - pad) * 2) / 2, Math.ceil((hi + pad) * 2) / 2];
  }
  const y2 = (v: number) => padT + (H - padB - padT) * (1 - (v - lo) / (hi - lo || 1));
  const cx = (i: number) => padL + i * bw + bw / 2;

  return (
    <figure ref={ref as React.RefObject<HTMLElement>} className="relative">
      <div className="mb-1 flex min-h-5 flex-wrap items-center justify-between gap-x-3 text-xs text-ink-2 tabular" aria-live="polite">
        {line ? <Legend items={[barName ?? label, line.name]} /> : <span />}
        <span>
          {h ? (
            <>
              <span className="font-medium text-ink">{h.x}</span>: {format(h.y)}
              {line && line.values[hover!] !== null && (
                <>
                  {" "}
                  · {line.name} {line.format(line.values[hover!]!)}
                </>
              )}
            </>
          ) : null}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        <YAxis ticks={ticks} y={y} x={padL} width={W - padR} />
        {line && lv.length > 0 && (
          <g>
            {[lo, (lo + hi) / 2, hi].map((v) => (
              <text key={v} x={W - padR + 6} y={y2(v) + 4} fontSize={11} fill="var(--series-2)">
                {line.format(v)}
              </text>
            ))}
          </g>
        )}
        {data.map((d, i) => {
          const bh = H - padB - y(d.y);
          const x = padL + i * bw + 1;
          const w = Math.max(1, bw - 2);
          const top = H - padB - bh;
          const r = Math.min(4, w / 2, bh);
          return (
            <g key={d.x} onMouseEnter={() => setHover(i)}>
              <rect x={padL + i * bw} y={0} width={bw} height={H - padB} fill="transparent" />
              <path
                d={`M${x},${H - padB} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top + r} V${H - padB} Z`}
                fill="var(--accent)"
                opacity={hover === null || hover === i ? 1 : 0.55}
              />
              {i % every === 0 && (
                <text x={x + w / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-2)">
                  {d.x}
                </text>
              )}
            </g>
          );
        })}
        {line && lv.length > 0 && (
          <g pointerEvents="none">
            <polyline
              points={line.values.map((v, i) => (v === null ? null : `${cx(i)},${y2(v)}`)).filter(Boolean).join(" ")}
              fill="none"
              stroke="var(--series-2)"
              strokeWidth={2}
            />
            {line.values.map((v, i) =>
              v === null ? null : <circle key={i} cx={cx(i)} cy={y2(v)} r={hover === i ? 5 : 3} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />,
            )}
          </g>
        )}
      </svg>
    </figure>
  );
}

/** Running share of the total, as a 0-100% secondary line. */
export function cumulativeLine(values: number[], name: string): SecondaryLine {
  const total = values.reduce((a, b) => a + b, 0) || 1;
  let run = 0;
  return {
    name,
    values: values.map((v) => (run += v) / total),
    format: (v) => `${Math.round(v * 100)}%`,
    domain: [0, 1],
  };
}

/** 10x10 matrix of counts over two same-scale axes (e.g. the joint vote
 *  distribution of two titles). Cells are tinted by which side of the diagonal
 *  they fall on: above = the y quantity is higher (secondary color), below =
 *  the x quantity is higher (accent), on it = equal (neutral); the shade grows
 *  with the cell's share. */
export function Matrix({
  data,
  labels,
  xName,
  yName,
  label,
  xLabels,
  yLabels,
}: {
  data: number[][]; // data[row = y][col = x]
  labels: string[];
  xName: string;
  yName: string;
  label: string;
  xLabels?: string[];
  yLabels?: string[];
}) {
  const { t } = useI18n();
  const [hover, setHover] = useState<[number, number] | null>(null);
  const total = data.flat().reduce((a, b) => a + b, 0) || 1;
  const share = (r: number, c: number) => data[r][c] / total;
  const max = Math.max(1e-9, ...data.flatMap((row, r) => row.map((_, c) => share(r, c))));
  const xl = xLabels ?? labels;
  const yl = yLabels ?? labels;
  const n = yl.length;
  const hue = (r: number, c: number) => (r > c ? "var(--series-2)" : r < c ? "var(--accent)" : "var(--diag)");
  const side = (() => {
    let above = 0;
    let below = 0;
    let diag = 0;
    data.forEach((row, r) => row.forEach((v, c) => (r > c ? (above += v) : r < c ? (below += v) : (diag += v))));
    return { above: above / total, below: below / total, diag: diag / total };
  })();
  return (
    <figure>
      <div className="mb-1 min-h-5 text-xs text-ink-2 tabular" aria-live="polite">
        {hover && (
          <>
            {xName} <span className="font-medium text-ink">{xl[hover[1]]}</span> · {yName} <span className="font-medium text-ink">{yl[hover[0]]}</span>:{" "}
            <span className="font-medium text-ink">{data[hover[0]][hover[1]].toLocaleString()}</span> ({(share(hover[0], hover[1]) * 100).toFixed(1)}%)
          </>
        )}
      </div>
      <div className="flex items-center justify-center gap-1 overflow-x-auto">
        <div className="shrink-0 text-xs font-medium text-ink-2 [writing-mode:vertical-rl] rotate-180">{yName} →</div>
        <table className="border-separate border-spacing-[2px] text-[10px]" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
          <tbody>
            {Array.from({ length: n }, (_, k) => n - 1 - k).map((r) => (
              <tr key={r}>
                <th scope="row" className="pr-1 text-right font-normal text-ink-2 tabular">
                  {yl[r]}
                </th>
                {xl.map((_, c) => {
                  const f = share(r, c) / max;
                  return (
                    <td
                      key={c}
                      onMouseEnter={() => setHover([r, c])}
                      className={`h-7 w-8 min-w-7 rounded-[3px] text-center tabular ${hover && hover[0] === r && hover[1] === c ? "outline-2 outline-ink" : ""}`}
                      style={{
                        background: data[r][c] ? `color-mix(in oklab, ${hue(r, c)} ${Math.round(12 + f * 88)}%, var(--surface-2))` : "var(--surface-2)",
                        color: f > 0.5 ? "#fff" : "var(--text)",
                      }}
                    >
                      {data[r][c] ? data[r][c] : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr>
              <th />
              {xl.map((l) => (
                <th key={l} className="pt-1 font-normal text-ink-2 tabular">
                  {l}
                </th>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-1 text-center text-xs font-medium text-ink-2">{xName} →</div>
      <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-ink-2">
        {(
          [
            ["var(--series-2)", t("chart.above", { y: yName }), side.above],
            ["var(--diag)", t("chart.diag"), side.diag],
            ["var(--accent)", t("chart.below", { x: xName }), side.below],
          ] as const
        ).map(([color, text, v]) => (
          <span key={text} className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
            {text} <span className="tabular font-medium text-ink">{(v * 100).toFixed(0)}%</span>
          </span>
        ))}
      </div>
    </figure>
  );
}

/** Dot + line chart for a mean over time; the y range is fitted to the data. */
export function LineChart({
  data,
  format,
  label,
  height = 160,
}: {
  data: { x: string; y: number }[];
  format: (v: number) => string;
  label: string;
  height?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth();
  if (!data.length) return null;
  const H = height;
  const padL = 40;
  const padB = 22;
  const padT = 10;
  const ys = data.map((d) => d.y);
  const lo = Math.floor(Math.min(...ys) * 2) / 2;
  const hi = Math.max(Math.ceil(Math.max(...ys) * 2) / 2, lo + 0.5);
  const step = (W - padL) / data.length;
  const px = (i: number) => padL + i * step + step / 2;
  const py = (v: number) => padT + (H - padB - padT) * (1 - (v - lo) / (hi - lo || 1));
  const every = Math.ceil(data.length / 8);
  const h = hover !== null ? data[hover] : null;
  return (
    <figure ref={ref as React.RefObject<HTMLElement>}>
      <div className="mb-1 h-5 text-xs text-ink-2 tabular" aria-live="polite">
        {h ? (
          <>
            <span className="font-medium text-ink">{h.x}</span>: {format(h.y)}
          </>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        <YAxis ticks={[lo, (lo + hi) / 2, hi]} y={py} x={padL} width={W} format={format} />
        {hover !== null && <line x1={px(hover)} x2={px(hover)} y1={padT} y2={H - padB} stroke="var(--text-3)" strokeWidth={1} />}
        <polyline points={data.map((d, i) => `${px(i)},${py(d.y)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {data.map((d, i) => (
          <g key={d.x} onMouseEnter={() => setHover(i)}>
            <rect x={padL + i * step} y={0} width={step} height={H} fill="transparent" />
            <circle cx={px(i)} cy={py(d.y)} r={hover === i ? 5 : 3} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            {i % every === 0 && (
              <text x={px(i)} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-2)">
                {d.x}
              </text>
            )}
          </g>
        ))}
      </svg>
    </figure>
  );
}

const SERIES = ["var(--accent)", "var(--series-2)"];

export function Legend({ items }: { items: string[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {items.map((label, i) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: SERIES[i] }} />
          {label}
        </span>
      ))}
    </div>
  );
}

/** Rank over time for up to two series; rank 1 is at the top. */
export function RankLines({
  points,
  names,
  formatX,
  label,
  height = 200,
}: {
  points: { x: number; ys: (number | null)[] }[];
  names: string[];
  formatX: (x: number) => string;
  label: string;
  height?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth();
  if (!points.length) return null;
  const H = height;
  const padL = 40;
  const padB = 22;
  const padT = 10;
  const all = points.flatMap((p) => p.ys.filter((y): y is number => y !== null));
  const lo = Math.max(1, Math.min(...all));
  const hi = Math.max(...all, lo + 1);
  const x0 = points[0].x;
  const x1 = Math.max(points[points.length - 1].x, x0 + 1);
  const px = (x: number) => padL + ((W - padL - 8) * (x - x0)) / (x1 - x0);
  const py = (r: number) => padT + ((H - padB - padT) * (r - lo)) / (hi - lo);
  const h = hover !== null ? points[hover] : null;
  const ticks = [lo, Math.round((lo + hi) / 2), hi];
  return (
    <figure ref={ref as React.RefObject<HTMLElement>} className="space-y-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Legend items={names} />
        <div className="h-5 text-xs text-ink-2 tabular" aria-live="polite">
          {h && (
            <>
              <span className="font-medium text-ink">{formatX(h.x)}</span>
              {h.ys.map((y, i) => (y === null ? null : <span key={i}> · {names[i]} #{y}</span>))}
            </>
          )}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {ticks.map((r) => (
          <g key={r}>
            <line x1={padL} x2={W} y1={py(r)} y2={py(r)} stroke="var(--border)" strokeWidth={1} />
            <text x={padL - 6} y={py(r) + 4} textAnchor="end" fontSize={11} fill="var(--text-2)">
              #{r}
            </text>
          </g>
        ))}
        {hover !== null && <line x1={px(points[hover].x)} x2={px(points[hover].x)} y1={padT} y2={H - padB} stroke="var(--text-3)" strokeWidth={1} />}
        {names.map((_, s) => {
          const pts = points.filter((p) => p.ys[s] !== null);
          return (
            <g key={s}>
              <polyline points={pts.map((p) => `${px(p.x)},${py(p.ys[s]!)}`).join(" ")} fill="none" stroke={SERIES[s]} strokeWidth={2} />
              {pts.length <= 15 && pts.map((p) => <circle key={p.x} cx={px(p.x)} cy={py(p.ys[s]!)} r={3} fill={SERIES[s]} stroke="var(--surface)" strokeWidth={2} />)}
            </g>
          );
        })}
        {points.map((p, i) => {
          const left = i === 0 ? padL : (px(points[i - 1].x) + px(p.x)) / 2;
          const right = i === points.length - 1 ? W : (px(p.x) + px(points[i + 1].x)) / 2;
          return <rect key={p.x} x={left} y={0} width={Math.max(1, right - left)} height={H} fill="transparent" onMouseEnter={() => setHover(i)} />;
        })}
        <text x={padL} y={H - 6} fontSize={11} fill="var(--text-2)">
          {formatX(points[0].x)}
        </text>
        <text x={W} y={H - 6} textAnchor="end" fontSize={11} fill="var(--text-2)">
          {formatX(points[points.length - 1].x)}
        </text>
      </svg>
    </figure>
  );
}

/** Horizontal bars with labels and values (for categorical breakdowns). */
export function HBars({ data, format = (v) => v.toLocaleString() }: { data: { label: string; value: number }[]; format?: (v: number) => string }) {
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <ul className="space-y-1.5">
      {data.map((d) => (
        <li key={d.label} className="grid grid-cols-[6rem_1fr_auto] items-center gap-2 text-sm" title={`${d.label}: ${format(d.value)}`}>
          <span className="truncate text-ink-2">{d.label}</span>
          <span className="h-2.5 rounded-r-[4px]" style={{ width: `${(d.value / max) * 100}%`, minWidth: d.value ? 2 : 0, background: "var(--accent)" }} />
          <span className="tabular text-xs text-ink-2">{format(d.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Two series side by side per category, as shares of each series' total. */
export function PairedBars({ x, a, b, names, label }: { x: string[]; a: number[]; b: number[]; names: [string, string]; label: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const ta = a.reduce((s, v) => s + v, 0) || 1;
  const tb = b.reduce((s, v) => s + v, 0) || 1;
  const sa = a.map((v) => v / ta);
  const sb = b.map((v) => v / tb);
  const ticks = niceTicks(Math.max(0.01, ...sa, ...sb));
  const max = ticks[ticks.length - 1];
  const [ref, W] = useWidth();
  const H = 180;
  const padL = 40;
  const padB = 22;
  const padT = 8;
  const yOf = (v: number) => padT + (H - padB - padT) * (1 - v / max);
  const gw = (W - padL) / x.length;
  const bw = (gw - 6) / 2;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <figure ref={ref as React.RefObject<HTMLElement>} className="space-y-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Legend items={names} />
        <div className="h-5 text-xs text-ink-2 tabular" aria-live="polite">
          {hover !== null && (
            <>
              <span className="font-medium text-ink">{x[hover]}</span> · {names[0]} {pct(sa[hover])} · {names[1]} {pct(sb[hover])}
            </>
          )}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        <YAxis ticks={ticks} y={yOf} x={padL} width={W} format={(v) => `${+(v * 100).toFixed(1)}%`} />
        {x.map((label, i) => {
          const ha = H - padB - yOf(sa[i]);
          const hb = H - padB - yOf(sb[i]);
          const gx = padL + i * gw + 2;
          return (
            <g key={label} onMouseEnter={() => setHover(i)}>
              <rect x={padL + i * gw} y={0} width={gw} height={H} fill="transparent" />
              <rect x={gx} y={H - padB - ha} width={bw} height={ha} rx={3} fill={SERIES[0]} />
              <rect x={gx + bw + 2} y={H - padB - hb} width={bw} height={hb} rx={3} fill={SERIES[1]} />
              <text x={padL + i * gw + gw / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-2)">
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

/** Scatter of two users' votes on common titles (1-10 on both axes). */
export function Scatter({
  points,
  xLabel,
  yLabel,
  label,
}: {
  points: { x: number; y: number; title: string }[];
  xLabel: string;
  yLabel: string;
  label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const S = 320;
  const pad = 30;
  const top = 22;
  const p = (v: number) => pad + ((S - pad - 8) * (v - 1)) / 9;
  const q = (v: number) => S - pad - ((S - pad - top) * (v - 1)) / 9;
  const h = hover !== null ? points[hover] : null;
  // Deterministic jitter so identical votes don't hide each other.
  const jitter = (i: number, k: number) => (((i * 9301 + k * 49297) % 233280) / 233280 - 0.5) * 0.35;
  return (
    <figure>
      <div className="mb-1 h-5 truncate text-xs text-ink-2 tabular" aria-live="polite">
        {h && (
          <>
            <span className="font-medium text-ink">{h.title}</span> · {xLabel} {h.x} · {yLabel} {h.y}
          </>
        )}
      </div>
      <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto w-full max-w-sm" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {[1, 4, 7, 10].map((v) => (
          <g key={v}>
            <line x1={p(v)} x2={p(v)} y1={top} y2={S - pad} stroke="var(--border)" />
            <line x1={pad} x2={S - 8} y1={q(v)} y2={q(v)} stroke="var(--border)" />
            <text x={p(v)} y={S - pad + 14} textAnchor="middle" fontSize={10} fill="var(--text-2)">{v}</text>
            <text x={pad - 6} y={q(v) + 3} textAnchor="end" fontSize={10} fill="var(--text-2)">{v}</text>
          </g>
        ))}
        <line x1={p(1)} y1={q(1)} x2={p(10)} y2={q(10)} stroke="var(--text-3)" strokeDasharray="3 3" />
        <text x={S - 8} y={S - 4} textAnchor="end" fontSize={10} fill="var(--text-2)">{xLabel} →</text>
        <text x={pad} y={11} fontSize={10} fill="var(--text-2)">↑ {yLabel}</text>
        {points.map((pt, i) => (
          <circle
            key={i}
            cx={p(pt.x + jitter(i, 1))}
            cy={q(pt.y + jitter(i, 2))}
            r={hover === i ? 6 : 4}
            fill="var(--accent)"
            fillOpacity={0.7}
            stroke="var(--surface)"
            strokeWidth={1.5}
            onMouseEnter={() => setHover(i)}
          />
        ))}
      </svg>
    </figure>
  );
}
