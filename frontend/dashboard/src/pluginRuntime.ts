import { api } from "./api";
import { encodePath, escapeHtml, formatSessionKeyForTable, renderMarkdown, shortTs, stripMarkdown } from "./format";
import type { DashboardGlobal, PluginConfig } from "./types";

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
  const plugins = await api<{ id: string; asset_version?: string }[]>("/api/dashboard/plugins").catch(() => []);
  for (const plugin of plugins) {
    const version = plugin.asset_version ? `?v=${encodeURIComponent(plugin.asset_version)}` : "";
    injectStylesheet(`/plugins/${plugin.id}/panel.css${version}`);
    await injectScript(`/plugins/${plugin.id}/panel.js${version}`);
  }
}

function injectScript(src: string): Promise<void> {
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => resolve();
    document.head.appendChild(script);
  });
}

function injectStylesheet(href: string): void {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.appendChild(link);
}
