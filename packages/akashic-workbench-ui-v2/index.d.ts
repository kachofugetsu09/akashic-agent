import type { WebMountDefinition, WebUiDisposer } from "@akashic/web-ui-v1";
import type { ComponentType, ReactNode } from "react";

export type SortOrder = "asc" | "desc";
export type ChartTone = "accent" | "success" | "warning" | "danger" | "muted";

export interface WorkbenchUi {
  readonly Grid: ComponentType<{
    children: ReactNode;
    columns?: 2 | 3 | 4;
    className?: string;
  }>;
  readonly Chip: ComponentType<{
    children: ReactNode;
    tone?: ChartTone | "neutral";
    dot?: boolean;
    className?: string;
  }>;
  readonly MetricTile: ComponentType<{
    label: string;
    value: ReactNode;
    unit?: string;
    delta?: number | null;
    sub?: ReactNode;
    tone?: ChartTone;
    spark?: number[];
    className?: string;
  }>;
  readonly Sparkline: ComponentType<{
    data: number[];
    tone?: ChartTone;
    height?: number;
    className?: string;
  }>;
  readonly TrendChart: ComponentType<{
    data: { label: string; value: number }[];
    kind?: "area" | "bar";
    tone?: ChartTone;
    height?: number;
    valueFmt?: (value: number) => string;
    className?: string;
    empty?: ReactNode;
  }>;
}

export interface WorkbenchColumn {
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

export interface FetchPageOptions {
  page: number;
  pageSize: number;
  filters?: Record<string, string>;
  sortBy?: string;
  sortOrder?: SortOrder;
  signal: AbortSignal;
}

export interface WorkbenchReadOptions {
  signal: AbortSignal;
}

export interface FetchPageResult {
  items: Record<string, unknown>[];
  total: number;
}

export interface WorkbenchDispatch {
  readonly ui: WorkbenchUi;
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
  closePane(): void;
}

export interface WorkbenchBatchAction {
  label: string;
  className: string;
  run(ids: string[]): Promise<void>;
}

export interface WorkbenchPanelEntry {
  id: string;
  order?: number;
  children?: WebMountDefinition[];
  label: string;
  viewLabel?: string;
  layout?: "table" | "workbench";
  pageSize?: number;
  countTitle?(total: number): string;
  rowKey: string;
  columns: WorkbenchColumn[];
  defaultSortBy?: string;
  defaultSortOrder?: SortOrder;
  getCount(options: WorkbenchReadOptions): Promise<number | null>;
  fetchPage(options: FetchPageOptions): Promise<FetchPageResult>;
  fetchDetail?(item: Record<string, unknown>, options: WorkbenchReadOptions): Promise<Record<string, unknown>>;
  rowClass?(item: Record<string, unknown>): string;
  emptyMessage?: string;
  renderMain?(container: HTMLElement, dispatch: WorkbenchDispatch): void | WebUiDisposer;
  renderDetail?(item: Record<string, unknown> | null, container: HTMLElement, dispatch: WorkbenchDispatch): void | WebUiDisposer;
  renderNavBody?(container: HTMLElement, dispatch: WorkbenchDispatch): void | WebUiDisposer;
  renderFilters?(container: HTMLElement, dispatch: WorkbenchDispatch): void | WebUiDisposer;
  renderTopbarAction?(container: HTMLElement, dispatch: WorkbenchDispatch): void | WebUiDisposer;
  batchActions?: WorkbenchBatchAction[];
  formatters?: Record<string, (value: unknown, item: Record<string, unknown>) => string>;
}
