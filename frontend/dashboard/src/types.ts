export type SortOrder = "asc" | "desc";
export type BuiltinView = "sessions" | "proactive";
export type ViewMode = BuiltinView | `plugin:${string}`;

export interface PageResult<T> {
  items: T[];
  total: number;
  page?: number;
  page_size?: number;
}

export interface SessionRow {
  key: string;
  created_at: string;
  updated_at: string;
  last_consolidated: number;
  metadata: Record<string, unknown>;
  last_user_at: string | null;
  last_proactive_at: string | null;
  message_count: number;
}

export interface MessageRow {
  id: string;
  session_key: string;
  seq: number;
  role: string;
  content: string;
  tool_chain: unknown;
  extra: Record<string, unknown>;
  timestamp: string;
}

export interface ProactiveOverview {
  counts: Record<string, number>;
  result_counts: Record<string, number>;
  flow_counts: Record<string, number>;
  last_tick_at: string | null;
  last_send_at: string | null;
  last_skip_reason: string | null;
  recent_tick: ProactiveTick | null;
}

export interface ProactiveTick {
  tick_id: string;
  session_key: string;
  started_at: string;
  finished_at?: string | null;
  gate_exit?: string | null;
  terminal_action?: string | null;
  skip_reason?: string | null;
  steps_taken?: number;
  drift_entered?: boolean | number;
  final_message?: string | null;
  alert_count?: number;
  content_count?: number;
  context_count?: number;
  interesting_ids?: string[];
  discarded_ids?: string[];
  cited_ids?: string[];
}

export interface ProactiveStep {
  step_index: number;
  phase: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: unknown;
  tool_result_text: string;
  terminal_action_after: string;
  skip_reason_after: string;
  final_message_after: string;
  interesting_ids_after: string[];
  discarded_ids_after: string[];
  cited_ids_after: string[];
}

export interface DashboardColumn {
  key: string;
  label: string;
  width?: number;
  flex?: boolean;
  fmt?: string;
  align?: "left" | "right";
  cellClass?: string;
  rawTitle?: boolean;
  sortable?: boolean;
  renderCell?(value: unknown, item: Record<string, unknown>): string;
}

export interface FetchPageOpts {
  page: number;
  pageSize: number;
  filters?: Record<string, string>;
  sortBy?: string;
  sortOrder?: SortOrder;
}

export interface FetchPageResult {
  items: Record<string, unknown>[];
  total: number;
}

// Dispatch context passed to all plugin render slots
export interface PluginDispatch {
  readonly filters: Readonly<Record<string, string>>;
  setFilter(key: string, value: string): void;
  clearFilter(key: string): void;
  setFilters(next: Record<string, string>): void;
  clearFilters(keys: string[]): void;
  readonly sortBy: string;
  readonly sortOrder: SortOrder;
  setSort(key: string): void;
  refresh(): void;
  activate(): void;
}

export interface PluginBatchAction {
  label: string;
  className: string;
  run(ids: string[]): Promise<void>;
}

export type PluginLayout = "table" | "workbench";

export interface PluginConfig {
  id: string;
  label: string;
  viewLabel?: string;
  layout?: PluginLayout;
  pageSize?: number;
  countTitle?: (n: number) => string;
  rowKey: string;
  columns: DashboardColumn[];
  defaultSortBy?: string;
  defaultSortOrder?: SortOrder;
  getCount(): Promise<number | null>;
  fetchPage(opts: FetchPageOpts): Promise<FetchPageResult>;
  fetchDetail?: (item: Record<string, unknown>) => Promise<Record<string, unknown>>;
  rowClass?: (item: Record<string, unknown>) => string;
  emptyMessage?: string;
  renderMain?(container: HTMLElement, dispatch: PluginDispatch): void;
  renderDetail?(item: Record<string, unknown> | null, container: HTMLElement, dispatch?: PluginDispatch): void;
  // React-native detail panel (Option 2). Takes precedence over renderDetail;
  // composes directly into the host React tree via the shared React instance.
  Detail?: import("react").ComponentType<{ item: Record<string, unknown> | null; dispatch?: PluginDispatch }>;
  // React-native full-width workbench (layout: "workbench"). Takes precedence
  // over renderMain.
  Main?: import("react").ComponentType<{ dispatch: PluginDispatch }>;
  renderNavBody?(container: HTMLElement, dispatch: PluginDispatch): void;
  renderFilters?(container: HTMLElement, dispatch: PluginDispatch): void;
  renderTopbarAction?(container: HTMLElement, dispatch: PluginDispatch): void;
  batchActions?: PluginBatchAction[];
  formatters?: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
}

export interface PluginState {
  page: number;
  pageSize: number;
  total: number;
  items: Record<string, unknown>[];
  activeRowKey: string | null;
  activeDetail: Record<string, unknown> | null;
  filters: Record<string, string>;
  sortBy: string;
  sortOrder: SortOrder;
  selectedIds: Set<string>;
}

// Shared visual vocabulary exposed to runtime-injected plugin panels so they
// render in the same industrial design system as the host (no third-party UI
// library, no per-plugin class drift). Factories return DOM nodes; `cx` returns
// the matching class strings for plugins that build HTML strings.
export type UiTone = "neutral" | "success" | "warning" | "danger" | "muted" | "accent";
export type UiBtnVariant = "primary" | "secondary" | "ghost" | "danger";
export type UiBtnSize = "sm" | "md" | "lg";

export interface DashboardUi {
  badge(text: string, opts?: { tone?: UiTone; dot?: boolean }): HTMLSpanElement;
  chip(text: string, opts?: { tone?: UiTone; dot?: boolean }): HTMLSpanElement;
  btn(label: string, opts?: { variant?: UiBtnVariant; size?: UiBtnSize; onClick?: (e: MouseEvent) => void }): HTMLButtonElement;
  tile(opts?: { label?: string; className?: string }): HTMLDivElement;
  label(text: string): HTMLSpanElement;
  cx: {
    badge(tone?: UiTone): string;
    btn(variant?: UiBtnVariant, size?: UiBtnSize): string;
    input: string;
    tile: string;
    label: string;
    mono: string;
  };
}

export interface DashboardGlobal {
  _plugins: PluginConfig[];
  _formatters: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
  registerPlugin(config: PluginConfig): void;
  registerFormatter(name: string, fn: (value: unknown, item: Record<string, unknown>) => string): void;
  ui: DashboardUi;
}
