// Type declarations for globals injected by the React dashboard bundle.
// Dashboard plugin interfaces are defined in frontend/dashboard/src/types.ts.

import type { DashboardColumn, DashboardGlobal, FetchPageOpts, FetchPageResult, PluginConfig } from "../frontend/dashboard/src/types";

export type Column = DashboardColumn;
export type { DashboardColumn, DashboardGlobal, FetchPageOpts, FetchPageResult, PluginConfig };

declare global {
  interface Window {
    AkashicDashboard: DashboardGlobal;
  }

  function api<T = unknown>(url: string, options?: RequestInit): Promise<T>;
  function escapeHtml(s: string): string;
  function encodePath(s: string): string;
  function renderMarkdown(s: string): string;
}
