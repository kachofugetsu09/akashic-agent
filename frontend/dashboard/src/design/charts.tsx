import { useEffect, useId, useState, type ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart as RBarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
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

const toneColor = (tone: ChartTone): string => `rgb(${TONE_RGB[tone]})`;

const AXIS_TICK = { fontSize: 10, fill: "rgba(138,138,143,0.6)", fontFamily: "JetBrains Mono, monospace" };
const GRID_STROKE = "rgba(255,255,255,0.07)";

// Hand-rolled SVG pie — a 2-slice hit/miss filled pie with a glossy, dimensional
// finish (radial sheen + drop shadow + rim) and a sweep-in animation on mount.
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

// MetricTile — a KPI card: a big tabular-nums value, an optional delta badge and
// unit, a secondary line, and an inline sparkline. The workhorse of the
// monitoring overview. Matches the superlog density (36px value, 14px radius).
export function MetricTile({
  label,
  value,
  unit,
  delta,
  sub,
  tone = "accent",
  spark,
  className,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  delta?: number | null;
  sub?: ReactNode;
  tone?: ChartTone;
  spark?: number[];
  className?: string;
}) {
  return (
    <div className={cn("relative overflow-hidden rounded-2xl border border-border bg-surface p-5 shadow-lift-sm", className)}>
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">{label}</span>
        {typeof delta === "number" && (
          <span className={cn("font-mono text-[11px] tabular-nums", delta >= 0 ? "text-success" : "text-danger")}>
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="font-sans text-4xl font-semibold leading-none tracking-tight tabular-nums text-fg">{value}</span>
        {unit && <span className="font-mono text-[11px] text-subtle">{unit}</span>}
      </div>
      {sub && <div className="mt-2 font-mono text-[11px] tabular-nums text-muted">{sub}</div>}
      {spark && spark.length > 1 && (
        <Sparkline data={spark} tone={tone} className="mt-4 w-full" height={40} />
      )}
    </div>
  );
}

// Sparkline — a normalized SVG area+line trend, no axes. Fills its container
// width via a preserveAspectRatio="none" viewBox.
export function Sparkline({
  data,
  tone = "accent",
  height = 40,
  className,
}: {
  data: number[];
  tone?: ChartTone;
  height?: number;
  className?: string;
}) {
  const uid = useId().replace(/:/g, "");
  const w = 100;
  const h = 40;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const span = max - min || 1;
  const step = data.length > 1 ? w / (data.length - 1) : w;
  const pts = data.map((v, i) => {
    const x = i * step;
    const y = h - ((v - min) / span) * (h - 2) - 1;
    return [x, Math.max(1, Math.min(h - 1, y))] as const;
  });
  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
  const area = `${line} L ${w} ${h} L 0 ${h} Z`;
  const color = toneColor(tone);
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{ height }}
      className={cn("block", className)}
    >
      <defs>
        <linearGradient id={`spark-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#spark-${uid})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Tooltip styled to the industrial tokens, shared by all recharts surfaces.
const TOOLTIP_CONTENT_STYLE = {
  background: "rgb(var(--color-surface-2-rgb))",
  border: "1px solid var(--color-border-strong)",
  borderRadius: 8,
  fontSize: 11,
  fontFamily: "JetBrains Mono, monospace",
  padding: "6px 10px",
  boxShadow: "0 8px 24px -6px rgba(0,0,0,0.6)",
};

// TrendChart — a recharts area/bar time series with a dashed horizontal grid,
// muted axis ticks (Y + sparse X), and a themed floating tooltip. This is what
// gives the monitoring page its precision-instrument feel.
export function TrendChart({
  data,
  kind = "area",
  tone = "accent",
  height = 170,
  valueFmt = (n: number) => String(n),
  className,
  empty,
}: {
  data: { label: string; value: number }[];
  kind?: "area" | "bar";
  tone?: ChartTone;
  height?: number;
  valueFmt?: (n: number) => string;
  className?: string;
  empty?: ReactNode;
}) {
  const uid = useId().replace(/:/g, "");
  const color = toneColor(tone);
  const allZero = data.every((d) => d.value === 0);
  if (data.length === 0 || (allZero && empty)) {
    return (
      <div className={cn("flex items-center justify-center text-[12px] text-subtle", className)} style={{ height }}>
        {empty ?? "暂无数据"}
      </div>
    );
  }
  const axisProps = {
    tick: AXIS_TICK,
    axisLine: false as const,
    tickLine: false as const,
  };
  return (
    <div className={className} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        {kind === "area" ? (
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={`trend-${uid}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.28" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
            <XAxis dataKey="label" {...axisProps} minTickGap={28} />
            <YAxis {...axisProps} width={42} tickFormatter={valueFmt} />
            <Tooltip
              cursor={{ stroke: GRID_STROKE }}
              contentStyle={TOOLTIP_CONTENT_STYLE}
              labelStyle={{ color: "rgb(var(--color-subtle-rgb))", marginBottom: 2 }}
              itemStyle={{ color: "rgb(var(--color-fg-rgb))" }}
              formatter={(v) => [valueFmt(Number(v)), ""] as [string, string]}
            />
            <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill={`url(#trend-${uid})`} dot={false} activeDot={{ r: 3, fill: color }} />
          </AreaChart>
        ) : (
          <RBarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={`bar-${uid}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.95" />
                <stop offset="100%" stopColor={color} stopOpacity="0.5" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
            <XAxis dataKey="label" {...axisProps} minTickGap={28} />
            <YAxis {...axisProps} width={42} tickFormatter={valueFmt} />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
              contentStyle={TOOLTIP_CONTENT_STYLE}
              labelStyle={{ color: "rgb(var(--color-subtle-rgb))", marginBottom: 2 }}
              itemStyle={{ color: "rgb(var(--color-fg-rgb))" }}
              formatter={(v) => [valueFmt(Number(v)), ""] as [string, string]}
            />
            <Bar dataKey="value" fill={`url(#bar-${uid})`} radius={[2, 2, 0, 0]} maxBarSize={28} />
          </RBarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
