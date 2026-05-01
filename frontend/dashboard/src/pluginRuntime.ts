import { api } from "./api";
import { encodePath, escapeHtml, formatSessionKeyForTable, renderMarkdown, shortTs, stripMarkdown } from "./format";
import type { DashboardGlobal, PluginConfig } from "./types";

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
  };
  target.AkashicDashboard = dashboard;
  target.api = api;
  target.escapeHtml = escapeHtml;
  target.encodePath = encodePath;
  target.renderMarkdown = renderMarkdown;
  return dashboard;
}

export async function loadPluginAssets(): Promise<void> {
  const plugins = await api<{ id: string }[]>("/api/dashboard/plugins").catch(() => []);
  for (const plugin of plugins) {
    injectStylesheet(`/plugins/${plugin.id}/panel.css`);
    await injectScript(`/plugins/${plugin.id}/panel.js`);
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
