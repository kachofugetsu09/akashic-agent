import { useEffect, useId, useState } from "react";
import { cn } from "./cn";

// Hand-rolled SVG pie — a 2-slice hit/miss filled pie with a glossy, dimensional
// finish (radial sheen + drop shadow + rim) and a sweep-in animation on mount.
// Lightweight (no charting lib), themed with the industrial tokens, and exposed
// via the dashboard SDK so plugins render it as a shared React component.
export function Pie({
  rate,
  hit,
  miss,
  title,
  hitLabel = "命中",
  missLabel = "未命中",
  size = 168,
  className,
}: {
  rate: number | null;
  hit: number;
  miss: number;
  title?: string;
  hitLabel?: string;
  missLabel?: string;
  size?: number;
  className?: string;
}) {
  const uid = useId().replace(/:/g, "");

  // 1. Resolve the ratio: explicit rate wins, else derive from hit/miss totals.
  const total = hit + miss;
  const ratio = rate != null ? Math.max(0, Math.min(1, rate)) : total > 0 ? hit / total : 0;
  const pct = Math.round(ratio * 1000) / 10;

  // 2. Sweep-in: animate the drawn fraction 0 -> 1 on mount (easeOutCubic).
  const [drawn, setDrawn] = useState(0);
  useEffect(() => {
    let raf = 0;
    let start = 0;
    const dur = 750;
    const tick = (now: number): void => {
      if (!start) start = now;
      const t = Math.min(1, (now - start) / dur);
      setDrawn(1 - Math.pow(1 - t, 3));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [ratio]);
  const shown = ratio * drawn;

  // 3. Pie geometry: hit wedge drawn clockwise from 12 o'clock over the miss disc.
  const cx = size / 2;
  const r = size / 2 - 3;
  const angle = shown * 2 * Math.PI;
  const ex = cx + r * Math.sin(angle);
  const ey = cx - r * Math.cos(angle);
  const largeArc = shown > 0.5 ? 1 : 0;
  const slice = `M ${cx} ${cx} L ${cx} ${cx - r} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey} Z`;

  const fmt = (n: number): string => new Intl.NumberFormat("en-US").format(Math.round(n));

  return (
    <div className={cn("flex flex-col items-center gap-3", className)}>
      {title && (
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">{title}</span>
      )}
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ filter: "drop-shadow(0 6px 14px rgba(0,0,0,0.55))" }}
      >
        <defs>
          <radialGradient id={`hit-${uid}`} cx="36%" cy="30%" r="78%">
            <stop offset="0%" stopColor="rgb(var(--color-success-rgb))" stopOpacity="1" />
            <stop offset="100%" stopColor="rgb(var(--color-success-rgb))" stopOpacity="0.72" />
          </radialGradient>
          <radialGradient id={`miss-${uid}`} cx="36%" cy="30%" r="80%">
            <stop offset="0%" stopColor="rgb(var(--color-surface-3-rgb))" stopOpacity="1" />
            <stop offset="100%" stopColor="rgb(var(--color-bg-rgb))" stopOpacity="1" />
          </radialGradient>
          <radialGradient id={`gloss-${uid}`} cx="32%" cy="24%" r="62%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.22" />
            <stop offset="55%" stopColor="#ffffff" stopOpacity="0.04" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx={cx} cy={cx} r={r} fill={`url(#miss-${uid})`} />
        {shown >= 0.999 ? (
          <circle cx={cx} cy={cx} r={r} fill={`url(#hit-${uid})`} />
        ) : shown > 0.001 ? (
          <path d={slice} fill={`url(#hit-${uid})`} />
        ) : null}
        {/* glossy sheen for the dimensional / glass finish */}
        <circle cx={cx} cy={cx} r={r} fill={`url(#gloss-${uid})`} />
        {/* crisp rim + inner top highlight */}
        <circle cx={cx} cy={cx} r={r} fill="none" stroke="rgba(0,0,0,0.35)" strokeWidth={1.5} />
        <circle cx={cx} cy={cx} r={r - 1} fill="none" stroke="rgba(255,255,255,0.10)" strokeWidth={1} />
      </svg>
      <div className="flex w-full items-center justify-center gap-4 font-mono text-[11px] tabular-nums">
        <span className="flex items-center gap-1.5 text-success">
          <span className="h-2 w-2 rounded-full bg-success" />
          {hitLabel} {fmt(hit)} · {pct}%
        </span>
        <span className="flex items-center gap-1.5 text-muted">
          <span className="h-2 w-2 rounded-full bg-surface-3" />
          {missLabel} {fmt(miss)} · {Math.round((100 - pct) * 10) / 10}%
        </span>
      </div>
    </div>
  );
}
