import { type ReactNode } from "react";
import { SunMoon } from "lucide-react";
import { cn } from "./cn";
import { renderMarkdown } from "../format";
import { cycleTheme, themes, useTheme } from "../../../theme/src/theme-runtime";

// ---------------------------------------------------------------------------
// Shared primitives for the dashboard and the /design storybook.
// Semantic theme colors · sharp corners (small radii) · hairline borders.
// This atomic layer is the single source of visual truth — the same vocabulary
// is exposed to runtime-injected plugin panels (see pluginRuntime.ts).
// ---------------------------------------------------------------------------

export function ShortcutKey({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded border border-border bg-surface-2 px-1 font-mono text-[10px] text-muted",
        className,
      )}
    >
      {children}
    </kbd>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="font-sans text-[11px] font-medium tracking-wide text-subtle">{children}</span>
  );
}

export function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <label className="mb-2 block font-sans text-[11px] font-medium tracking-wide text-muted">
      {children}
    </label>
  );
}

export function Tile({
  children,
  className,
  label,
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  label?: string;
  padded?: boolean;
}) {
  return (
    <div className={cn("relative rounded border border-border bg-surface", padded && "p-4", className)}>
      {label && (
        <div className="mb-4 flex items-center justify-between">
          <Label>{label}</Label>
        </div>
      )}
      {children}
    </div>
  );
}

export function Stack({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("ak-plugin-stack", className)}>{children}</div>;
}

export function Grid({
  children,
  columns = 2,
  className,
}: {
  children: ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}) {
  return <div className={cn("ak-plugin-grid", `ak-plugin-grid-${columns}`, className)}>{children}</div>;
}

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cn("ak-plugin-panel", className)}>{children}</section>;
}

export function Toolbar({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("ak-plugin-toolbar", className)}>{children}</div>;
}

export function Markdown({ children, className }: { children: unknown; className?: string }) {
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

function JsonScalar({ value }: { value: unknown }) {
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

function JsonNode({ value, depth, label }: { value: unknown; depth: number; label?: string }) {
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

export function JsonView({ value, className }: { value: unknown; className?: string }) {
  return <div className={cn("json-tree", className)}><JsonNode value={value} depth={0} /></div>;
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

export type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export type BtnSize = "sm" | "md" | "lg";

const BTN_SIZES: Record<BtnSize, string> = {
  sm: "h-7 px-2.5 text-[12px]",
  md: "h-8 px-3 text-[13px]",
  lg: "h-10 px-4 text-[14px]",
};

const BTN_VARIANTS: Record<BtnVariant, string> = {
  primary:
    "bg-accent text-accent-ink hover:brightness-110 active:brightness-95 shadow-[0_1px_0_0_rgba(255,255,255,0.12)_inset]",
  secondary: "bg-transparent text-fg border border-border hover:border-border-strong",
  ghost: "bg-transparent text-fg hover:bg-surface-2",
  danger: "bg-danger/20 text-danger hover:bg-danger/30 active:bg-danger/25",
};

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
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={cn(
        "inline-flex select-none items-center gap-2 rounded-md font-medium tracking-tight transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40",
        BTN_SIZES[size],
        BTN_VARIANTS[variant],
        className,
      )}
    >
      {loading && (
        <span className="inline-block h-3 w-3 animate-spin rounded-full border border-current border-t-transparent" />
      )}
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Chips
// ---------------------------------------------------------------------------

export type ChipTone = "neutral" | "success" | "warning" | "danger" | "muted" | "accent";

const CHIP_TONES: Record<ChipTone, string> = {
  neutral: "bg-surface-2 text-fg",
  success: "bg-success/15 text-success",
  warning: "bg-warning/15 text-warning",
  danger: "bg-danger/15 text-danger",
  muted: "bg-surface-2 text-muted",
  accent: "bg-accent-soft text-accent",
};

const CHIP_DOTS: Record<ChipTone, string> = {
  neutral: "bg-muted",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
  muted: "bg-subtle",
  accent: "bg-accent",
};

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
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 font-sans text-[11px] tabular-nums",
        CHIP_TONES[tone],
        className,
      )}
    >
      {dot && <span className={cn("h-1.5 w-1.5 rounded-full", CHIP_DOTS[tone])} />}
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  const { className, ...rest } = props;
  return (
    <input
      {...rest}
      className={cn(
        "h-9 w-full rounded-md border border-border bg-surface-2 px-3 text-[13px] text-fg placeholder:text-subtle focus:border-border-strong focus:outline-none",
        className,
      )}
    />
  );
}

export function SearchInput({
  value,
  onChange,
  placeholder = "搜索",
  shortcut,
  className,
}: {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  shortcut?: string;
  className?: string;
}) {
  return (
    <div className={cn("relative", className)}>
      <svg
        className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-subtle"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      <input
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        className={cn(
          "h-9 w-full rounded-md border border-border bg-surface-2 pl-8 text-[12.5px] text-fg placeholder:text-subtle focus:border-border-strong focus:outline-none",
          shortcut ? "pr-16" : "pr-3",
        )}
      />
      {shortcut && (
        <span className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm border border-border bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] text-muted">
          {shortcut}
        </span>
      )}
    </div>
  );
}

export function Select({
  value,
  onChange,
  options,
  className,
}: {
  value?: string;
  onChange?: (value: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <div className={cn("relative", className)}>
      <select
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className="h-9 w-full appearance-none rounded-md border border-border bg-surface-2 pl-3 pr-8 text-[13px] text-fg focus:border-border-strong focus:outline-none"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <svg
        className="pointer-events-none absolute right-2.5 top-1/2 h-3 w-3 -translate-y-1/2 text-subtle"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="m6 9 6 6 6-6" />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Brand mark + app shell chrome
// ---------------------------------------------------------------------------

export function BrandMark({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "grid h-8 w-8 place-items-center rounded-md border border-border bg-surface-2 font-mono text-[15px] font-semibold text-accent shadow-inset-hairline",
        className,
      )}
    >
      A
    </div>
  );
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

export function ThemeToggle() {
  const theme = useTheme();
  const options = themes();
  const currentIndex = options.findIndex((option) => option.id === theme.id);
  const nextTheme = options[(currentIndex + 1) % options.length];
  return (
    <button
      type="button"
      onClick={() => cycleTheme()}
      title={`当前主题：${theme.label}；切换到${nextTheme.label}`}
      aria-label={`切换主题，当前为${theme.label}，下一主题为${nextTheme.label}`}
      className="theme-cycle-button"
    >
      <SunMoon size={20} strokeWidth={2} aria-hidden="true" />
      <span>主题 · {theme.label}</span>
    </button>
  );
}
