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
