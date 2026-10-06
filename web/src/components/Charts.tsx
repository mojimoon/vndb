import { useState } from "react";

/** Single-series bar chart with a hover readout. Values are magnitudes, so the
 *  y axis starts at zero; bars get 4px rounded tops and 2px gaps. */
export function BarChart({
  data,
  format = (v) => v.toLocaleString(),
  height = 180,
  label,
}: {
  data: { x: string; y: number }[];
  format?: (v: number) => string;
  height?: number;
  label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...data.map((d) => d.y));
  const W = 600;
  const H = height;
  const padB = 22;
  const padT = 8;
  const bw = W / data.length;
  const ticks = [0, 0.5, 1].map((f) => f * max);
  const h = hover !== null ? data[hover] : null;
  // Label every bar when there are few, otherwise about 8 labels.
  const every = data.length <= 12 ? 1 : Math.ceil(data.length / 8);

  return (
    <figure className="relative">
      <div className="mb-1 h-5 text-xs text-ink-2 tabular" aria-live="polite">
        {h ? (
          <>
            <span className="font-medium text-ink">{h.x}</span>: {format(h.y)}
          </>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {ticks.map((v) => {
          const y = padT + (H - padB - padT) * (1 - v / max);
          return <line key={v} x1={0} x2={W} y1={y} y2={y} stroke="var(--border)" strokeWidth={1} />;
        })}
        {data.map((d, i) => {
          const bh = ((H - padB - padT) * d.y) / max;
          const x = i * bw + 1;
          const w = Math.max(1, bw - 2);
          const y = H - padB - bh;
          const r = Math.min(4, w / 2, bh);
          return (
            <g key={d.x} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} tabIndex={-1}>
              <rect x={i * bw} y={0} width={bw} height={H - padB} fill="transparent" />
              <path
                d={`M${x},${H - padB} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${H - padB} Z`}
                fill="var(--accent)"
                opacity={hover === null || hover === i ? 1 : 0.55}
              />
              {i % every === 0 && (
                <text x={x + w / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-3)">
                  {d.x}
                </text>
              )}
            </g>
          );
        })}
      </svg>
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
  if (!data.length) return null;
  const W = 600;
  const H = height;
  const padB = 22;
  const padT = 10;
  const ys = data.map((d) => d.y);
  const lo = Math.floor(Math.min(...ys) * 2) / 2;
  const hi = Math.ceil(Math.max(...ys) * 2) / 2 || lo + 1;
  const step = W / data.length;
  const px = (i: number) => i * step + step / 2;
  const py = (v: number) => padT + (H - padB - padT) * (1 - (v - lo) / (hi - lo || 1));
  const every = Math.ceil(data.length / 8);
  const h = hover !== null ? data[hover] : null;
  return (
    <figure>
      <div className="mb-1 h-5 text-xs text-ink-2 tabular" aria-live="polite">
        {h ? (
          <>
            <span className="font-medium text-ink">{h.x}</span>: {format(h.y)}
          </>
        ) : (
          <span className="text-ink-3">
            {format(lo)} – {format(hi)}
          </span>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {[lo, (lo + hi) / 2, hi].map((v) => (
          <line key={v} x1={0} x2={W} y1={py(v)} y2={py(v)} stroke="var(--border)" strokeWidth={1} />
        ))}
        {hover !== null && <line x1={px(hover)} x2={px(hover)} y1={padT} y2={H - padB} stroke="var(--text-3)" strokeWidth={1} />}
        <polyline points={data.map((d, i) => `${px(i)},${py(d.y)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {data.map((d, i) => (
          <g key={d.x} onMouseEnter={() => setHover(i)}>
            <rect x={i * step} y={0} width={step} height={H} fill="transparent" />
            <circle cx={px(i)} cy={py(d.y)} r={hover === i ? 5 : 3} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            {i % every === 0 && (
              <text x={px(i)} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-3)">
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
  if (!points.length) return null;
  const W = 900;
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
    <figure className="space-y-1">
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
            <text x={padL - 6} y={py(r) + 4} textAnchor="end" fontSize={11} fill="var(--text-3)">
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
        <text x={padL} y={H - 6} fontSize={11} fill="var(--text-3)">
          {formatX(points[0].x)}
        </text>
        <text x={W} y={H - 6} textAnchor="end" fontSize={11} fill="var(--text-3)">
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
  const max = Math.max(0.01, ...sa, ...sb);
  const W = 600;
  const H = 180;
  const padB = 22;
  const padT = 8;
  const gw = W / x.length;
  const bw = (gw - 6) / 2;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <figure className="space-y-1">
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
        <line x1={0} x2={W} y1={H - padB} y2={H - padB} stroke="var(--border)" />
        {x.map((label, i) => {
          const ha = ((H - padB - padT) * sa[i]) / max;
          const hb = ((H - padB - padT) * sb[i]) / max;
          const gx = i * gw + 2;
          return (
            <g key={label} onMouseEnter={() => setHover(i)}>
              <rect x={i * gw} y={0} width={gw} height={H} fill="transparent" />
              <rect x={gx} y={H - padB - ha} width={bw} height={ha} rx={3} fill={SERIES[0]} />
              <rect x={gx + bw + 2} y={H - padB - hb} width={bw} height={hb} rx={3} fill={SERIES[1]} />
              <text x={i * gw + gw / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="var(--text-3)">
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
            <text x={p(v)} y={S - pad + 14} textAnchor="middle" fontSize={10} fill="var(--text-3)">{v}</text>
            <text x={pad - 6} y={q(v) + 3} textAnchor="end" fontSize={10} fill="var(--text-3)">{v}</text>
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
