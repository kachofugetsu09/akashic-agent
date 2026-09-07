import { type ReactNode } from "react";
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
  accent: "var(--ak-color-action-primary-rgb)",
  success: "var(--ak-color-status-success-rgb)",
  warning: "var(--ak-color-status-warning-rgb)",
  danger: "var(--ak-color-status-error-rgb)",
  muted: "var(--ak-color-text-secondary-rgb)",
};

const toneColor = (tone: ChartTone): string => `rgb(${TONE_RGB[tone]})`;

const AXIS_TICK = { fontSize: 10, fill: "rgb(var(--ak-color-text-muted-rgb) / 0.72)", fontFamily: "var(--sans)" };
const GRID_STROKE = "rgb(var(--ak-color-border-default-rgb) / 0.72)";

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
    <div className={cn("relative overflow-hidden border border-border bg-surface p-4", className)}>
      <div className="flex items-center justify-between">
        <span className="font-sans text-[11px] font-medium tracking-wide text-muted">{label}</span>
        {typeof delta === "number" && (
          <span className={cn("font-sans text-[11px] tabular-nums", delta >= 0 ? "text-success" : "text-danger")}>
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="font-sans text-4xl font-semibold leading-none tracking-tight tabular-nums text-fg">{value}</span>
        {unit && <span className="font-sans text-[11px] text-subtle">{unit}</span>}
      </div>
      {sub && <div className="mt-2 font-sans text-[11px] tabular-nums text-muted">{sub}</div>}
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
      <path d={area} fill={color} opacity="0.08" />
      <path d={line} fill="none" stroke={color} strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Tooltip styled to the industrial tokens, shared by all recharts surfaces.
const TOOLTIP_CONTENT_STYLE = {
  background: "rgb(var(--ak-color-bg-surface-low-rgb))",
  border: "1px solid var(--ak-color-border-strong)",
  borderRadius: 8,
  fontSize: 11,
  fontFamily: "var(--sans)",
  padding: "6px 10px",
  boxShadow: "0 2px 8px var(--ak-color-shadow)",
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
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
            <XAxis dataKey="label" {...axisProps} minTickGap={28} />
            <YAxis {...axisProps} width={42} tickFormatter={valueFmt} />
            <Tooltip
              cursor={{ stroke: GRID_STROKE }}
              contentStyle={TOOLTIP_CONTENT_STYLE}
              labelStyle={{ color: "rgb(var(--ak-color-text-muted-rgb))", marginBottom: 2 }}
              itemStyle={{ color: "rgb(var(--ak-color-text-primary-rgb))" }}
              formatter={(v) => [valueFmt(Number(v)), ""] as [string, string]}
            />
            <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill={color} fillOpacity={0.08} dot={false} activeDot={{ r: 3, fill: color }} />
          </AreaChart>
        ) : (
          <RBarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={GRID_STROKE} />
            <XAxis dataKey="label" {...axisProps} minTickGap={28} />
            <YAxis {...axisProps} width={42} tickFormatter={valueFmt} />
            <Tooltip
              cursor={{ fill: "rgb(var(--ak-color-bg-surface-high-rgb) / 0.4)" }}
              contentStyle={TOOLTIP_CONTENT_STYLE}
              labelStyle={{ color: "rgb(var(--ak-color-text-muted-rgb))", marginBottom: 2 }}
              itemStyle={{ color: "rgb(var(--ak-color-text-primary-rgb))" }}
              formatter={(v) => [valueFmt(Number(v)), ""] as [string, string]}
            />
            <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} maxBarSize={28} />
          </RBarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
