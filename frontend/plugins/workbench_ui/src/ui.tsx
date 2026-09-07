import { MaterialButton } from "@akashic/web-ui-v1";
import { type ReactNode } from "react";
import { cn } from "./cn";
import { renderMarkdown } from "./format";

export function Grid({
  children,
  columns = 2,
  className,
}: {
  children: ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}): React.ReactElement {
  return <div className={cn("ak-plugin-grid", `ak-plugin-grid-${columns}`, className)}>{children}</div>;
}

export function Markdown({ children, className }: { children: unknown; className?: string }): React.ReactElement {
  return <div className={cn("ak-markdown", className)} dangerouslySetInnerHTML={{ __html: renderMarkdown(children) }} />;
}

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text || (!text.startsWith("{") && !text.startsWith("[") && !text.startsWith('"'))) return value;
  try {
    return parseMaybeJson(JSON.parse(text));
  } catch {
    return value;
  }
}

function JsonScalar({ value }: { value: unknown }): React.ReactElement {
  if (typeof value === "string") {
    if (value.length > 280) {
      const preview = value.slice(0, 160).replace(/\s+/g, " ");
      return <details className="jt-long"><summary>{preview}…</summary><Markdown>{value}</Markdown></details>;
    }
    return <Markdown className="jt-str">{value}</Markdown>;
  }
  if (value === null) return <span className="jt-null">空值</span>;
  if (typeof value === "boolean") return <span className="jt-bool">{value ? "是" : "否"}</span>;
  if (typeof value === "number") return <span className="jt-num">{value}</span>;
  return <span>{String(value)}</span>;
}

function JsonNode({ value, depth, label }: { value: unknown; depth: number; label?: string }): React.ReactElement {
  const parsed = parseMaybeJson(value);
  const indent = { paddingLeft: `${12 + Math.min(depth, 8) * 14}px` };
  if (parsed === null || typeof parsed !== "object") {
    return <div className="jt-row" style={indent}>
      {label && <div className="jt-key" title={label}>{label}</div>}
      <div className="jt-value"><JsonScalar value={parsed} /></div>
    </div>;
  }
  const array = Array.isArray(parsed);
  const entries = array ? parsed.map((item, index) => [String(index), item] as const) : Object.entries(parsed);
  return <details className="jt-node" open={depth < 2}>
    <summary className="jt-toggle" style={indent}>
      <span className="jt-branch-label" title={label}>{label ?? (array ? "列表" : "字段")}</span>
      <span className="jt-meta">{array ? "列表" : "字段"} · {entries.length} 项</span>
    </summary>
    <div className="jt-children">
      {entries.map(([key, child]) => <JsonNode
        key={key}
        value={child}
        depth={depth + 1}
        label={array ? `第 ${Number(key) + 1} 项` : key}
      />)}
    </div>
  </details>;
}

export function JsonView({ value, className }: { value: unknown; className?: string }): React.ReactElement {
  return <div className={cn("json-tree", className)}><JsonNode value={value} depth={0} /></div>;
}

export type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export type BtnSize = "sm" | "md" | "lg";

const BTN_VARIANTS = {
  primary: "filled",
  secondary: "outlined",
  ghost: "text",
  danger: "danger",
} as const;

export function Btn({
  children,
  variant = "primary",
  size = "md",
  disabled,
  loading,
  className,
  type = "button",
  onClick,
}: {
  children: ReactNode;
  variant?: BtnVariant;
  size?: BtnSize;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  type?: "button" | "submit" | "reset";
  onClick?: (event: React.MouseEvent<HTMLElement>) => void;
}): React.ReactElement {
  return <MaterialButton
    type={type}
    onClick={onClick}
    disabled={disabled}
    loading={loading}
    variant={BTN_VARIANTS[variant]}
    className={cn(`ak-material-button--${size}`, className)}
  >
    {children}
  </MaterialButton>;
}

export type ChipTone = "neutral" | "success" | "warning" | "danger" | "muted" | "accent";

const CHIP_TONES: Record<ChipTone, string> = {
  neutral: "ak-chip--neutral",
  success: "ak-chip--success",
  warning: "ak-chip--warning",
  danger: "ak-chip--danger",
  muted: "ak-chip--muted",
  accent: "ak-chip--accent",
};

const CHIP_DOTS: Record<ChipTone, string> = {
  neutral: "bg-muted",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
  muted: "bg-subtle",
  accent: "bg-accent",
};

export function chipClass(tone: ChipTone = "neutral", className?: string): string {
  return cn(
    "ak-chip inline-flex items-center gap-1.5 px-2.5 py-1 font-sans text-[11px] tabular-nums",
    CHIP_TONES[tone],
    className,
  );
}

export function Chip({
  children,
  tone = "neutral",
  dot = false,
  className,
}: {
  children: ReactNode;
  tone?: ChipTone;
  dot?: boolean;
  className?: string;
}): React.ReactElement {
  return <span className={chipClass(tone, className)}>
    {dot && <span className={cn("h-1.5 w-1.5 rounded-full", CHIP_DOTS[tone])} />}
    {children}
  </span>;
}
