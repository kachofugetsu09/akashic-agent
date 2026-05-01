// Type declarations for the AkashicDashboard global and helpers injected by app.js.
// Consumed by plugin TypeScript files compiled with esbuild.

export interface Column {
  /** Field name on each row item */
  key: string;
  /** Header label shown in table head */
  label: string;
  /** Fixed pixel width (exclusive with flex) */
  width?: number;
  /** Flexible column that fills remaining space (1fr) */
  flex?: boolean;
  /** Built-in or registered formatter name (default: "text") */
  fmt?: string;
  align?: "left" | "right";
  /** Extra CSS class applied to each body cell */
  cellClass?: string;
  /** When true, sets HTML title to the raw (pre-format) value */
  rawTitle?: boolean;
}

export interface FetchPageOpts {
  page: number;
  pageSize: number;
}

export interface FetchPageResult {
  items: Record<string, unknown>[];
  total: number;
}

export interface PluginConfig {
  id: string;
  label: string;
  viewLabel?: string;
  pageSize?: number;
  countTitle?: (n: number) => string;
  /** Property on each row item used as the unique row key */
  rowKey: string;
  columns: Column[];
  getCount(): Promise<number | null>;
  fetchPage(opts: FetchPageOpts): Promise<FetchPageResult>;
  /** If present, called on row click; result is passed to renderDetail */
  fetchDetail?: (item: Record<string, unknown>) => Promise<Record<string, unknown>>;
  rowClass?: (item: Record<string, unknown>) => string;
  emptyMessage?: string;
  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void;
  /** Extra named formatters: value → display string (will be HTML-escaped) */
  formatters?: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
}

declare global {
  interface Window {
    AkashicDashboard: {
      _plugins: PluginConfig[];
      _formatters: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
      registerPlugin(config: PluginConfig): void;
      registerFormatter(name: string, fn: (value: unknown, item: Record<string, unknown>) => string): void;
    };
  }

  // Globals injected by app.js
  function api(url: string): Promise<Record<string, unknown>>;
  function escapeHtml(s: string): string;
  function encodePath(s: string): string;
  function renderMarkdown(s: string): string;
}
