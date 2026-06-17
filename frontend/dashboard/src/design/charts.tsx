import { useEffect, useId, useState, type ReactNode } from "react";
import { cn } from "./cn";

// Shared accent palette for the monitoring atoms below — resolved to the
// industrial RGB-triplet tokens so opacity blending stays theme-aware.
export type ChartTone = "accent" | "success" | "warning" | "danger" | "muted";

const TONE_RGB: Record<ChartTone, string> = {
  accent: "var(--color-accent-rgb)",
  success: "var(--color-success-rgb)",
  warning: "var(--color-warning-rgb)",
  danger: "var(--color-danger-rgb)",
  muted: "var(--color-muted-rgb)",
};

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

// MetricTile — a KPI card: a big tabular-nums value with an optional unit, a
// secondary line (delta / context), and an inline sparkline. The workhorse of
// the monitoring overview.
export function MetricTile({
  label,
  value,
  unit,
  sub,
  tone = "accent",
  spark,
  className,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  sub?: ReactNode;
  tone?: ChartTone;
  spark?: number[];
  className?: string;
}) {
  return (
    <div className={cn("relative overflow-hidden rounded-lg border border-border bg-surface p-4 shadow-lift-sm", className)}>
      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">{label}</span>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="font-mono text-[26px] font-semibold leading-none tracking-tight tabular-nums text-fg">{value}</span>
        {unit && <span className="font-mono text-[12px] text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-1.5 font-mono text-[11px] tabular-nums text-muted">{sub}</div>}
      {spark && spark.length > 1 && (
        <Sparkline data={spark} tone={tone} className="mt-3 w-full" height={28} />
      )}
    </div>
  );
}

// Sparkline — a normalized SVG area+line trend, no axes. Fills its container
// width via a preserveAspectRatio="none" viewBox.
export function Sparkline({
  data,
  tone = "accent",
  height = 32,
  className,
}: {
  data: number[];
  tone?: ChartTone;
  height?: number;
  className?: string;
}) {
  const uid = useId().replace(/:/g, "");
  const w = 100;
  const h = 32;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const span = max - min || 1;
  const step = data.length > 1 ? w / (data.length - 1) : w;
  const pts = data.map((v, i) => {
    const x = i * step;
    const y = h - ((v - min) / span) * h;
    return [x, Math.max(1, Math.min(h - 1, y))] as const;
  });
  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
  const area = `${line} L ${w} ${h} L 0 ${h} Z`;
  const rgb = TONE_RGB[tone];
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{ height }}
      className={cn("block", className)}
    >
      <defs>
        <linearGradient id={`spark-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={`rgb(${rgb})`} stopOpacity="0.32" />
          <stop offset="100%" stopColor={`rgb(${rgb})`} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#spark-${uid})`} />
      <path d={line} fill="none" stroke={`rgb(${rgb})`} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// BarChart — a bucketed bar series with a baseline grid line and per-bar hover
// titles. Used for token / error / iteration trends over time.
export function BarChart({
  points,
  height = 140,
  tone = "accent",
  valueFmt = (n: number) => String(n),
  className,
}: {
  points: { label: string; value: number }[];
  height?: number;
  tone?: ChartTone;
  valueFmt?: (n: number) => string;
  className?: string;
}) {
  const rgb = TONE_RGB[tone];
  const max = Math.max(...points.map((p) => p.value), 1);
  if (points.length === 0) {
    return (
      <div className={cn("flex items-center justify-center text-[12px] text-subtle", className)} style={{ height }}>
        暂无数据
      </div>
    );
  }
  return (
    <div className={cn("relative", className)} style={{ height }}>
      <div className="absolute inset-x-0 bottom-5 top-2 border-b border-border" />
      <div className="absolute inset-x-0 bottom-5 top-2 flex items-end gap-[2px]">
        {points.map((p, i) => {
          const frac = p.value / max;
          return (
            <div
              key={`${p.label}-${i}`}
              className="group relative flex-1 cursor-default"
              style={{ height: "100%" }}
              title={`${p.label} · ${valueFmt(p.value)}`}
            >
              <div
                className="absolute bottom-0 w-full rounded-t-[2px] transition-[height] duration-300"
                style={{
                  height: `${Math.max(frac * 100, p.value > 0 ? 2 : 0)}%`,
                  background: `linear-gradient(to top, rgb(${rgb} / 0.55), rgb(${rgb} / 0.95))`,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="absolute inset-x-0 bottom-0 flex justify-between font-mono text-[9px] tabular-nums text-subtle">
        <span>{points[0]?.label}</span>
        {points.length > 2 && <span>{points[Math.floor(points.length / 2)]?.label}</span>}
        <span>{points[points.length - 1]?.label}</span>
      </div>
    </div>
  );
}
