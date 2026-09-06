import type { WebUiDisposer } from "@akashic/web-ui-v1";
import type {
  FetchPageResult,
  SortOrder,
  WorkbenchBatchAction,
  WorkbenchColumn,
  WorkbenchDispatch,
  WorkbenchPanelEntry,
} from "@akashic/workbench-ui-v2";

export type {
  FetchPageResult,
  SortOrder,
  WorkbenchBatchAction as PluginBatchAction,
  WorkbenchColumn as DashboardColumn,
  WorkbenchDispatch as PluginDispatch,
};
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

export type PluginConfig = WorkbenchPanelEntry & {
  applyStyle(host: HTMLElement): WebUiDisposer;
};

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
