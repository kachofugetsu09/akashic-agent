export type SortOrder = "asc" | "desc";
export type BuiltinView = "sessions" | "compaction";
export type ViewMode = BuiltinView | `plugin:${string}`;

export interface PageResult<T> {
  items: T[];
  total: number;
  page?: number;
  page_size?: number;
}

export interface SessionCompactionBrief {
  generation: number;
  trigger: string;
  tokens_before: number;
  tokens_after: number;
  summary_preview: string;
  model: string | null;
  created_at: string | null;
}

export interface CompactionGeneration {
  generation: number;
  parent_generation: number;
  created_at: string;
  trigger: string;
  summary: string;
  source_from_seq: number;
  consolidated_through_seq: number;
  source_message_count: number;
  source_plan_digest: string;
  model: string;
  model_runtime_id: string;
  context_window: number;
  threshold_tokens: number;
  hard_input_tokens: number;
  keep_recent_tokens: number;
  tokens_before: number;
  tokens_after: number;
  summary_usage: Record<string, unknown>;
  invalidated_at: string | null;
  invalidated_reason: string | null;
}

export interface CompactionDetail {
  head: {
    parent_generation: number;
    next_generation: number;
  };
  active: CompactionGeneration | null;
  history: CompactionGeneration[];
}

export interface SessionRow {
  key: string;
  created_at: string;
  updated_at: string;
  last_consolidated: number;
  metadata: Record<string, unknown>;
  last_user_at: string | null;
  last_proactive_at: string | null;
  first_message_content: string | null;
  message_count: number;
  compaction?: SessionCompactionBrief | null;
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
  signal: AbortSignal;
}

export interface PluginReadOptions {
  signal: AbortSignal;
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
  closePane?(): void;
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
  applyStyle(host: HTMLElement): WebUiDisposer;
  viewLabel?: string;
  layout?: PluginLayout;
  pageSize?: number;
  countTitle?: (n: number) => string;
  rowKey: string;
  columns: DashboardColumn[];
  defaultSortBy?: string;
  defaultSortOrder?: SortOrder;
  getCount(opts: PluginReadOptions): Promise<number | null>;
  fetchPage(opts: FetchPageOpts): Promise<FetchPageResult>;
  fetchDetail?: (item: Record<string, unknown>, opts: PluginReadOptions) => Promise<Record<string, unknown>>;
  rowClass?: (item: Record<string, unknown>) => string;
  emptyMessage?: string;
  renderMain?(container: HTMLElement, dispatch: PluginDispatch): void | WebUiDisposer;
  renderDetail?(item: Record<string, unknown> | null, container: HTMLElement, dispatch?: PluginDispatch): void | WebUiDisposer;
  // React-native detail panel (Option 2). Takes precedence over renderDetail;
  // composes directly into the host React tree via the shared React instance.
  Detail?: import("react").ComponentType<{ item: Record<string, unknown> | null; dispatch?: PluginDispatch }>;
  // React-native full-width workbench (layout: "workbench"). Takes precedence
  // over renderMain.
  Main?: import("react").ComponentType<{ dispatch: PluginDispatch }>;
  renderNavBody?(container: HTMLElement, dispatch: PluginDispatch): void | WebUiDisposer;
  renderFilters?(container: HTMLElement, dispatch: PluginDispatch): void | WebUiDisposer;
  renderTopbarAction?(container: HTMLElement, dispatch: PluginDispatch): void | WebUiDisposer;
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
import type { WebUiDisposer } from "@akashic/web-ui-v1";
