import { useEffect, useState, type ReactNode } from "react";
import { cn } from "./cn";

// ---------------------------------------------------------------------------
// Shared primitives for the dashboard and the /design storybook.
// Dark canvas · cobalt accent · sharp corners (small radii) · hairline borders.
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
    <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-subtle">{children}</span>
  );
}

export function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <label className="mb-2 block font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
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
    <div className={cn("relative rounded-lg border border-border bg-surface", padded && "p-5", className)}>
      {label && (
        <div className="mb-4 flex items-center justify-between">
          <Label>{label}</Label>
        </div>
      )}
      {children}
    </div>
  );
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
    "bg-accent text-accent-ink hover:brightness-110 active:brightness-95 shadow-[0_1px_0_0_rgba(255,255,255,0.12)_inset,0_6px_14px_-6px_rgba(72,90,226,0.55)]",
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
        "inline-flex select-none items-center gap-2 rounded-md font-medium tracking-tight transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-40",
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
        "inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 font-mono text-[11px] tabular-nums",
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

export type Theme = "light" | "dark";

function readTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(readTheme()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return theme;
}

export function ThemeToggle() {
  const theme = useTheme();
  const next: Theme = theme === "dark" ? "light" : "dark";
  const apply = (t: Theme): void => {
    document.documentElement.setAttribute("data-theme", t);
    try {
      localStorage.setItem("theme", t);
    } catch {
      // ignore storage failures (private mode etc.)
    }
  };
  return (
    <button
      type="button"
      onClick={() => apply(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="grid h-7 w-7 place-items-center rounded-md border border-border text-muted transition-colors hover:text-fg"
    >
      {theme === "dark" ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
