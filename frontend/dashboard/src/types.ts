export type SortOrder = "asc" | "desc";
export type BuiltinView = "sessions" | "memory" | "proactive";
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
  ts: string;
}

export interface MemoryRow {
  id: string;
  memory_type: string;
  summary: string;
  source_ref: string;
  happened_at: string;
  status: string;
  created_at: string;
  updated_at: string;
  reinforcement: number;
  emotional_weight: number;
  has_embedding: boolean;
  scope_channel: string;
  scope_chat_id: string;
}

export interface MemoryDetail extends MemoryRow {
  content_hash?: string;
  extra_json: Record<string, unknown>;
  embedding_dim: number;
  embedding?: number[] | null;
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
  rowKey: string;
  columns: DashboardColumn[];
  getCount(): Promise<number | null>;
  fetchPage(opts: FetchPageOpts): Promise<FetchPageResult>;
  fetchDetail?: (item: Record<string, unknown>) => Promise<Record<string, unknown>>;
  rowClass?: (item: Record<string, unknown>) => string;
  emptyMessage?: string;
  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void;
  formatters?: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
}

export interface PluginState {
  page: number;
  pageSize: number;
  total: number;
  items: Record<string, unknown>[];
  activeRowKey: string | null;
  activeDetail: Record<string, unknown> | null;
}

export interface DashboardGlobal {
  _plugins: PluginConfig[];
  _formatters: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
  registerPlugin(config: PluginConfig): void;
  registerFormatter(name: string, fn: (value: unknown, item: Record<string, unknown>) => string): void;
}
