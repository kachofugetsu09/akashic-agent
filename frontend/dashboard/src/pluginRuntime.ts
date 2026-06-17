import { api } from "./api";
import { encodePath, escapeHtml, formatSessionKeyForTable, renderMarkdown, shortTs, stripMarkdown } from "./format";
import type { DashboardGlobal, DashboardUi, PluginConfig, UiBtnSize, UiBtnVariant, UiTone } from "./types";

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text || (!text.startsWith("{") && !text.startsWith("[") && !text.startsWith("\""))) return value;
  try {
    return parseMaybeJson(JSON.parse(text));
  } catch {
    return value;
  }
}

function scalarNode(value: unknown): HTMLElement {
  const span = document.createElement("span");
  if (typeof value === "string") {
    span.className = "jt-str";
    span.textContent = JSON.stringify(value);
    return span;
  }
  if (typeof value === "number") {
    span.className = "jt-num";
    span.textContent = String(value);
    return span;
  }
  if (typeof value === "boolean") {
    span.className = "jt-bool";
    span.textContent = String(value);
    return span;
  }
  if (value === null) {
    span.className = "jt-null";
    span.textContent = "null";
    return span;
  }
  span.textContent = String(value);
  return span;
}

function makeNode(value: unknown, depth: number): HTMLElement {
  const parsed = parseMaybeJson(value);
  if (parsed === null || typeof parsed !== "object") {
    return scalarNode(parsed);
  }

  const isArray = Array.isArray(parsed);
  const entries = isArray ? parsed.map((item, index) => [String(index), item] as const) : Object.entries(parsed);
  const wrapper = document.createElement("div");
  wrapper.className = "jt-node";

  const details = document.createElement("details");
  details.open = depth < 3;
  const summary = document.createElement("summary");
  summary.className = "jt-toggle";
  summary.textContent = isArray ? `Array(${entries.length})` : `Object(${entries.length})`;
  details.appendChild(summary);

  const children = document.createElement("div");
  children.className = "jt-children";
  for (const [key, child] of entries) {
    const row = document.createElement("div");
    row.className = "jt-row";
    const keySpan = document.createElement("span");
    keySpan.className = "jt-key";
    keySpan.textContent = isArray ? `[${key}]` : key;
    const colon = document.createElement("span");
    colon.className = "jt-colon";
    colon.textContent = ": ";
    row.appendChild(keySpan);
    row.appendChild(colon);
    row.appendChild(makeNode(child, depth + 1));
    children.appendChild(row);
  }

  details.appendChild(children);
  wrapper.appendChild(details);
  return wrapper;
}

export function makeJsonViewer(data: unknown): HTMLElement {
  const root = document.createElement("div");
  root.className = "json-tree";
  root.appendChild(makeNode(data, 0));
  return root;
}

export function jvPlaceholder(data: unknown): string {
  return `<div data-jv="${escapeHtml(encodeURIComponent(JSON.stringify(data ?? null)))}"></div>`;
}

export function attachJsonViewers(container: ParentNode): void {
  container.querySelectorAll<HTMLElement>("[data-jv]").forEach((node) => {
    const encoded = node.getAttribute("data-jv");
    if (!encoded) return;
    try {
      const value = JSON.parse(decodeURIComponent(encoded));
      node.replaceWith(makeJsonViewer(value));
    } catch {
      node.replaceWith(makeJsonViewer(null));
    }
  });
}

// ---------------------------------------------------------------------------
// Shared visual vocabulary (industrial design system) handed to plugin panels.
// Class strings are full literals so Tailwind's content scanner keeps them.
// ---------------------------------------------------------------------------

// Tone -> background + text classes, mirroring design/ui.tsx Chip.
const UI_TONES: Record<UiTone, string> = {
  neutral: "bg-surface-2 text-fg",
  success: "bg-success/15 text-success",
  warning: "bg-warning/15 text-warning",
  danger: "bg-danger/15 text-danger",
  muted: "bg-surface-2 text-muted",
  accent: "bg-accent-soft text-accent",
};

const UI_TONE_DOTS: Record<UiTone, string> = {
  neutral: "bg-muted",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
  muted: "bg-subtle",
  accent: "bg-accent",
};

const UI_BTN_SIZES: Record<UiBtnSize, string> = {
  sm: "h-7 px-2.5 text-[12px]",
  md: "h-8 px-3 text-[13px]",
  lg: "h-10 px-4 text-[14px]",
};

const UI_BTN_VARIANTS: Record<UiBtnVariant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-110 active:brightness-95",
  secondary: "bg-transparent text-fg border border-border hover:border-border-strong",
  ghost: "bg-transparent text-fg hover:bg-surface-2",
  danger: "bg-danger/20 text-danger hover:bg-danger/30 active:bg-danger/25",
};

const UI_BADGE_BASE = "inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 font-mono text-[11px] tabular-nums";
const UI_BTN_BASE = "inline-flex select-none items-center gap-2 rounded-md font-medium tracking-tight transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-40";
const UI_INPUT = "h-9 w-full rounded-md border border-border bg-surface-2 px-3 text-[13px] text-fg placeholder:text-subtle focus:border-border-strong focus:outline-none";
const UI_TILE = "relative rounded-lg border border-border bg-surface p-5";
const UI_LABEL = "font-mono text-[10px] uppercase tracking-[0.2em] text-subtle";
const UI_MONO = "font-mono tabular-nums";

function badgeClass(tone: UiTone = "neutral"): string {
  return `${UI_BADGE_BASE} ${UI_TONES[tone]}`;
}

function btnClass(variant: UiBtnVariant = "primary", size: UiBtnSize = "md"): string {
  return `${UI_BTN_BASE} ${UI_BTN_SIZES[size]} ${UI_BTN_VARIANTS[variant]}`;
}

function createDashboardUi(): DashboardUi {
  const makeBadge = (text: string, opts?: { tone?: UiTone; dot?: boolean }): HTMLSpanElement => {
    const tone = opts?.tone ?? "neutral";
    const span = document.createElement("span");
    span.className = badgeClass(tone);
    if (opts?.dot) {
      const dot = document.createElement("span");
      dot.className = `h-1.5 w-1.5 rounded-full ${UI_TONE_DOTS[tone]}`;
      span.appendChild(dot);
    }
    span.appendChild(document.createTextNode(text));
    return span;
  };
  return {
    badge: makeBadge,
    chip: makeBadge,
    btn(text, opts) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = btnClass(opts?.variant ?? "primary", opts?.size ?? "md");
      button.textContent = text;
      if (opts?.onClick) button.addEventListener("click", opts.onClick);
      return button;
    },
    tile(opts) {
      const div = document.createElement("div");
      div.className = opts?.className ? `${UI_TILE} ${opts.className}` : UI_TILE;
      if (opts?.label) {
        const label = document.createElement("div");
        label.className = `mb-4 ${UI_LABEL}`;
        label.textContent = opts.label;
        div.appendChild(label);
      }
      return div;
    },
    label(text) {
      const span = document.createElement("span");
      span.className = UI_LABEL;
      span.textContent = text;
      return span;
    },
    cx: {
      badge: badgeClass,
      btn: btnClass,
      input: UI_INPUT,
      tile: UI_TILE,
      label: UI_LABEL,
      mono: UI_MONO,
    },
  };
}

export function installDashboardGlobals(onRegister: (plugin: PluginConfig) => void): DashboardGlobal {
  const dashboard: DashboardGlobal = {
    _plugins: [],
    _formatters: {
      text: (value) => String(value ?? ""),
      "mono-session": (value) => formatSessionKeyForTable(value),
      "mono-time": (value) => shortTs(value),
      "text-preview": (value) => stripMarkdown(value),
      metric: (value) => String(value ?? 0),
    },
    registerPlugin(config) {
      const exists = this._plugins.some((plugin) => plugin.id === config.id);
      if (exists) {
        return;
      }
      this._plugins.push(config);
      onRegister(config);
    },
    registerFormatter(name, fn) {
      this._formatters[name] = fn;
    },
    ui: createDashboardUi(),
  };

  const target = window as Window & {
    AkashicDashboard: DashboardGlobal;
    api: typeof api;
    escapeHtml: typeof escapeHtml;
    encodePath: typeof encodePath;
    renderMarkdown: typeof renderMarkdown;
    makeJsonViewer: typeof makeJsonViewer;
    jvPlaceholder: typeof jvPlaceholder;
    attachJsonViewers: typeof attachJsonViewers;
  };
  target.AkashicDashboard = dashboard;
  target.api = api;
  target.escapeHtml = escapeHtml;
  target.encodePath = encodePath;
  target.renderMarkdown = renderMarkdown;
  target.makeJsonViewer = makeJsonViewer;
  target.jvPlaceholder = jvPlaceholder;
  target.attachJsonViewers = attachJsonViewers;
  return dashboard;
}

export async function loadPluginAssets(): Promise<void> {
  const plugins = await api<{ id: string; panels: { name: string; js_version: string; has_css: boolean }[] }[]>("/api/dashboard/plugins").catch(() => []);
  for (const plugin of plugins) {
    for (const panel of plugin.panels ?? []) {
      const v = panel.js_version ? `?v=${encodeURIComponent(panel.js_version)}` : "";
      if (panel.has_css) injectStylesheet(`/plugins/${plugin.id}/${panel.name}.css${v}`);
      // ESM modules: bare react / @akashic/dashboard-ui specifiers resolve via
      // the host import map to shared singletons. The module registers itself
      // as a side-effect of import.
      await importPanel(`/plugins/${plugin.id}/${panel.name}.js${v}`);
    }
  }
}

async function importPanel(src: string): Promise<void> {
  try {
    await import(/* @vite-ignore */ src);
  } catch (error) {
    console.error(`[dashboard] failed to load plugin panel ${src}`, error);
  }
}

function injectStylesheet(href: string): void {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.appendChild(link);
}
