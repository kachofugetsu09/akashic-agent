import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChevronDown, ChevronLeft, ChevronRight, X } from "lucide-react";
import { MaterialIconButton } from "@akashic/web-ui-v1";
import type { WebEntryView, WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type { WorkbenchUi } from "@akashic/workbench-ui-v2";
import type { DashboardColumn, FetchPageResult, PluginConfig, PluginDispatch, PluginState, SortOrder } from "./types";
import { api, bindApiRequest } from "./api";
import { formatSessionKeyForTable, relativeTime, roleClass, shortTs, stripMarkdown } from "./format";
import { akashicBrandIcon } from "./brand";
import { PluginDetail, PluginMain, mountPluginDom } from "./PluginDetail";
import { Btn, Chip as WorkbenchChip, Grid, JsonView, Markdown } from "./ui";
import { MetricTile, Sparkline, TrendChart } from "./charts";
import "./style.css";

const WORKBENCH_UI = {
  Chip: WorkbenchChip,
  Grid,
  MetricTile,
  Sparkline,
  TrendChart,
} satisfies WorkbenchUi;

const WORKBENCH_FORMATTERS: Record<string, (value: unknown, item: Record<string, unknown>) => string> = {
  text: (value) => String(value ?? ""),
  "mono-session": (value) => formatSessionKeyForTable(value),
  "mono-time": (value) => shortTs(value),
  "text-preview": (value) => stripMarkdown(value),
  metric: (value) => String(value ?? 0),
};

function makeDispatch(
  plugin: PluginConfig,
  getState: () => PluginState | null,
  onSetState: (updater: (s: PluginState) => PluginState) => void,
  startRead: () => AbortController,
  onActivate?: () => void,
  onClosePane?: () => void,
  onError: (error: unknown) => void = (error) => console.error("[dashboard] plugin request failed", error),
): PluginDispatch {
  const report = (promise: Promise<void>): void => {
    void promise.catch(onError);
  };

  const fetchAndApply = async (
    nextFilters: Record<string, string>,
    nextSortBy: string,
    nextSortOrder: SortOrder,
  ): Promise<void> => {
    const state = getState();
    if (!state) return;
    const controller = startRead();
    let result: FetchPageResult;
    try {
      result = checkedPluginPage(plugin, await plugin.fetchPage({
      page: 1,
      pageSize: state.pageSize,
      filters: nextFilters,
      sortBy: nextSortBy,
      sortOrder: nextSortOrder,
      signal: controller.signal,
      }));
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      return;
    }
    if (controller.signal.aborted) return;
    onSetState((s) => ({
      ...s,
      page: 1,
      total: result.total,
      items: result.items,
      activeRowKey: null,
      activeDetail: null,
      selectedIds: new Set(),
      filters: nextFilters,
      sortBy: nextSortBy,
      sortOrder: nextSortOrder,
    }));
  };

  const updateFilters = (updater: (filters: Record<string, string>) => Record<string, string>): void => {
    const state = getState();
    if (!state) return;
    report(fetchAndApply(updater({ ...state.filters }), state.sortBy, state.sortOrder));
  };

  return {
    ui: WORKBENCH_UI,
    get filters() { return getState()?.filters ?? {}; },
    setFilter(key: string, value: string): void {
      updateFilters((filters) => ({ ...filters, [key]: value }));
    },
    clearFilter(key: string): void {
      updateFilters((filters) => {
        delete filters[key];
        return filters;
      });
    },
    setFilters(next: Record<string, string>): void {
      updateFilters((filters) => ({ ...filters, ...next }));
    },
    clearFilters(keys: string[]): void {
      updateFilters((filters) => {
        for (const key of keys) delete filters[key];
        return filters;
      });
    },
    get sortBy() { return getState()?.sortBy ?? ""; },
    get sortOrder() { return getState()?.sortOrder ?? "desc"; },
    setSort(key: string): void {
      const state = getState();
      if (!state) return;
      const nextOrder: SortOrder = state.sortBy === key && state.sortOrder === "desc" ? "asc" : "desc";
      report(fetchAndApply(state.filters, key, nextOrder));
    },
    refresh(): void {
      const state = getState();
      if (!state) return;
      report(fetchAndApply(state.filters, state.sortBy, state.sortOrder));
    },
    activate(): void {
      onActivate?.();
    },
    closePane(): void {
      onClosePane?.();
    },
  };
}

function checkedPluginPage(plugin: PluginConfig, result: FetchPageResult): FetchPageResult {
  if (
    !result
    || !Array.isArray(result.items)
    || typeof result.total !== "number"
    || !Number.isFinite(result.total)
    || result.total < 0
  ) {
    throw new Error(`插件 ${plugin.id} 返回了无效分页数据`);
  }
  return result;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function useLatestReader<T>(value: T): () => T {
  const ref = useRef(value);
  useLayoutEffect(() => {
    ref.current = value;
  }, [value]);
  return useCallback(() => ref.current, []);
}


interface SessionRow {
  key: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  head_seq: number;
  first_message_content: string;
  attributes: { visibility: "listed" | "internal"; learning: string };
}

interface MessageRow {
  id: string;
  session_id: string;
  seq: number;
  timestamp: string;
  author: string;
  source: string;
  body: { kind: string; parts?: { kind: string; value?: unknown }[] };
}

interface SessionPage {
  items: SessionRow[];
  total: number;
  next_cursor: [string, string] | null;
}

interface MessagePage {
  items: MessageRow[];
  through_seq: number;
  next_before_seq: number | null;
  has_more: boolean;
}

interface NavigationProps {
  currentPluginId: string | null;
  sessionsCount: number;
  plugins: PluginConfig[];
  counts: Record<string, number | null>;
  onSelect(pluginId: string | null): void;
}

function Brand(): React.ReactElement {
  return <div className="brand">
    <img className="brand-mark" src={akashicBrandIcon} alt="" />
    <div><div className="brand-title">Akashic</div><div className="brand-sub">Dashboard</div></div>
  </div>;
}

function ModuleSwitcher(props: NavigationProps): React.ReactElement {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const current = props.plugins.find((plugin) => plugin.id === props.currentPluginId) ?? null;
  const currentLabel = current?.label ?? "Sessions";
  const currentCount = current ? props.counts[current.id] ?? 0 : props.sessionsCount;

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent): void => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent): void => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const select = (pluginId: string | null): void => {
    props.onSelect(pluginId);
    setOpen(false);
    queueMicrotask(() => triggerRef.current?.focus());
  };

  return <div className="module-switcher" ref={rootRef}>
    <button ref={triggerRef} className="module-switcher-trigger" type="button" aria-expanded={open}
      aria-controls="workbench-module-options" onClick={() => setOpen((current) => !current)}>
      <span className="module-switcher-label">{currentLabel}</span>
      <span className="module-switcher-meta"><span className="module-switcher-count">{currentCount}</span>
        <ChevronDown className={open ? "open" : ""} size={16} aria-hidden="true" /></span>
    </button>
    <div id="workbench-module-options" className="module-switcher-options" hidden={!open}>
      <button className={`module-switcher-option ${props.currentPluginId === null ? "active" : ""}`} type="button"
        aria-current={props.currentPluginId === null ? "page" : undefined} onClick={() => select(null)}>
        <span>Sessions</span><span>{props.sessionsCount}</span>
      </button>
      {props.plugins.map((plugin) => <button key={plugin.id}
        className={`module-switcher-option ${props.currentPluginId === plugin.id ? "active" : ""}`} type="button"
        aria-current={props.currentPluginId === plugin.id ? "page" : undefined} onClick={() => select(plugin.id)}>
        <span>{plugin.label}</span><span>{props.counts[plugin.id] ?? 0}</span>
      </button>)}
    </div>
  </div>;
}

function SessionNavItem(props: { session: SessionRow; active: boolean; onSelect(): void }): React.ReactElement {
  const title = stripMarkdown(props.session.first_message_content).trim() || formatSessionKeyForTable(props.session.key);
  return <button className={`session-item ${props.active ? "active" : ""}`} type="button"
    aria-current={props.active ? "page" : undefined} title={`${title}\n${props.session.key}`} onClick={props.onSelect}>
    <div className="nav-item-row"><span className="nav-item-name">{title}</span>
      <span className="nav-item-count" title={`${props.session.message_count} 条消息`}>{props.session.message_count}</span></div>
    <div className="nav-item-desc"><span>{props.session.attributes.visibility === "internal" ? "内部" : props.session.key.split(":", 1)[0]}</span>
      <span aria-hidden="true">·</span><span>{relativeTime(props.session.updated_at)}</span></div>
  </button>;
}

function messageText(message: MessageRow): string {
  return message.body.parts
    ?.filter((part) => part.kind === "text" && typeof part.value === "string")
    .map((part) => String(part.value))
    .join("\n") ?? "";
}

function Messages(props: NavigationProps & { selected: string | null; select(key: string | null): void }): React.ReactElement {
  const [sessions, setSessions] = useState<SessionPage>({ items: [], total: 0, next_cursor: null });
  const [prefix, setPrefix] = useState("");
  const [visibility, setVisibility] = useState("");
  const [page, setPage] = useState<MessagePage | null>(null);
  const [activeMessage, setActiveMessage] = useState<MessageRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sessionRequest = useRef<AbortController | null>(null);
  const messageRequest = useRef<AbortController | null>(null);
  const report = useCallback((error: unknown): void => {
    if (!isAbortError(error)) setError(error instanceof Error ? error.message : String(error));
  }, []);

  const loadSessions = useCallback(async (cursor: [string, string] | null = null) => {
    sessionRequest.current?.abort();
    const controller = new AbortController();
    sessionRequest.current = controller;
    const query = new URLSearchParams({ prefix, limit: "50" });
    if (visibility) query.set("visibility", visibility);
    cursor?.forEach((value) => query.append("cursor", value));
    let result: SessionPage;
    try {
      result = await api<SessionPage>(`/api/dashboard/sessions?${query}`, { signal: controller.signal });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      return;
    }
    if (!controller.signal.aborted) {
      setSessions((previous) => ({ ...result, items: cursor ? [...previous.items, ...result.items] : result.items }));
      setError(null);
    }
  }, [prefix, visibility]);

  const loadMessages = useCallback(async (older: MessagePage | null = null) => {
    messageRequest.current?.abort();
    if (props.selected === null) {
      setPage(null);
      return;
    }
    const controller = new AbortController();
    messageRequest.current = controller;
    const query = new URLSearchParams({ limit: "50" });
    if (older !== null) {
      query.set("through_seq", String(older.through_seq));
      if (older.next_before_seq !== null) query.set("before_seq", String(older.next_before_seq));
    }
    let result: MessagePage;
    try {
      result = await api<MessagePage>(`/api/dashboard/sessions/${encodeURIComponent(props.selected)}/messages?${query}`, { signal: controller.signal });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      return;
    }
    if (!controller.signal.aborted) {
      setPage({ ...result, items: older ? [...result.items, ...older.items] : result.items });
      setError(null);
    }
  }, [props.selected]);

  useEffect(() => {
    void loadSessions().catch(report);
    return () => sessionRequest.current?.abort();
  }, [loadSessions, report]);
  useEffect(() => {
    setPage(null);
    setActiveMessage(null);
    void loadMessages().catch(report);
    return () => messageRequest.current?.abort();
  }, [loadMessages, report]);

  const messageColumns = "64px 86px minmax(220px, 1fr) 110px 92px 104px";
  return <div className="shell">
    <aside className="sessions-pane" aria-label="会话目录">
      <Brand />
      <ModuleSwitcher {...props} sessionsCount={sessions.total} />
      <div className="explorer-body">
        <div className="filters-stack session-filters">
          <label className="search search-small"><span aria-hidden="true">⌕</span>
            <input aria-label="会话前缀" value={prefix} placeholder="例如 akashic:" onChange={(event) => setPrefix(event.target.value)} />
          </label>
          <select aria-label="可见范围" value={visibility} onChange={(event) => setVisibility(event.target.value)}>
            <option value="">全部范围</option><option value="listed">普通</option><option value="internal">内部</option>
          </select>
        </div>
        <div className="session-list">
          <button className={`all-messages-row ${props.selected === null ? "active" : ""}`} type="button"
            onClick={() => props.select(null)}><span>全部会话</span><strong>{sessions.total}</strong></button>
          {sessions.items.map((session) => <SessionNavItem key={session.key} session={session}
            active={props.selected === session.key} onSelect={() => props.select(session.key)} />)}
          {sessions.next_cursor && <Btn size="sm" variant="ghost" onClick={() => void loadSessions(sessions.next_cursor).catch(report)}>更多会话</Btn>}
        </div>
      </div>
    </aside>
    <section className="content-shell">
      <header className="content-toolbar">
        <div className="content-filters"><div className="filter-row">
          {props.selected ? <div className="active-session-chip"><span>Session</span><code>{props.selected}</code>
            <button aria-label="清除 Session 筛选" type="button" onClick={() => props.select(null)}>×</button></div>
            : <span className="muted-text">从左侧选择一个 Session 查看原始消息</span>}
        </div></div>
        <div className="content-toolbar-actions"><Btn size="sm" variant="ghost" onClick={() => void loadSessions().catch(report)}>刷新会话</Btn>
          {props.selected && <Btn size="sm" variant="secondary" onClick={() => void loadMessages().catch(report)}>读取最新消息</Btn>}</div>
      </header>
      {error && <div role="alert" className="plugin-entry-error"><strong>请求失败</strong><span>{error}</span></div>}
      <main className="workspace">
        <section className="messages-pane">
          <div className="table-head" style={{ gridTemplateColumns: messageColumns }}>
            <div>Seq</div><div>Author</div><div>Content</div><div>Source</div><div>Kind</div><div>Timestamp</div>
          </div>
          <div className="table-body">
            {page?.has_more && <div className="pane-head"><Btn size="sm" variant="ghost" onClick={() => void loadMessages(page).catch(report)}>读取更早消息</Btn></div>}
            {page?.items.map((message) => <div className="table-row-wrap" key={message.id}>
              <button className={`table-row table-row ${activeMessage?.id === message.id ? "active" : ""}`} style={{ gridTemplateColumns: messageColumns }}
                type="button" aria-expanded={activeMessage?.id === message.id} onClick={() => setActiveMessage((current) => current?.id === message.id ? null : message)}>
                <span className="cell-seq mono">#{message.seq}</span>
                <span><span className={`role-pill ${roleClass(message.author)}`}>{message.author}</span></span>
                <span className="content-preview">{stripMarkdown(messageText(message))}</span>
                <span className="cell-source">{message.source}</span>
                <span className="cell-type">{message.body.kind}</span>
                <span className="cell-time mono">{shortTs(message.timestamp)}</span>
              </button>
            </div>)}
            {props.selected && page?.items.length === 0 && <div className="empty-state">此会话没有消息。</div>}
            {!props.selected && <div className="empty-state">选择 Session 后，这里会显示按原始顺序保存的 Message。</div>}
          </div>
          <footer className="table-foot"><div>{page ? `已读取 ${page.items.length} 条` : "消息只读视图"}</div>
            <div className="muted-text">编辑、撤销和删除尚未接入；历史摘要保持保留</div></footer>
        </section>
        <aside className={`detail-pane ${activeMessage ? "is-open" : ""}`} aria-label="详情">
          {activeMessage ? <MessageDetail message={activeMessage} onClose={() => setActiveMessage(null)} />
            : <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">点开一条消息后，这里会显示完整正文和原始字段。</div></div>}
        </aside>
      </main>
    </section>
  </div>;
}

function MessageDetail({ message, onClose }: { message: MessageRow; onClose(): void }): React.ReactElement {
  return <div className="detail-wrap">
    <div className="detail-toolbar"><div><div className="detail-title">消息详情</div>
      <div className="detail-subtext">{message.session_id} · #{message.seq}</div></div>
      <MaterialIconButton variant="standard" label="关闭详情" onClick={onClose}><X size={18} aria-hidden="true" /></MaterialIconButton></div>
    <div className="detail-grid">
      {detailRow("author", <span className={`role-pill ${roleClass(message.author)}`}>{message.author}</span>)}
      {detailRow("source", <code>{message.source}</code>)}
      {detailRow("time", <code>{message.timestamp}</code>)}
      {detailRow("id", <code>{message.id}</code>)}
    </div>
    <div className="detail-block"><div className="detail-label">Content</div><Markdown className="detail-content">{messageText(message)}</Markdown></div>
    <div className="detail-block"><div className="detail-label">Raw Message</div><JsonView value={message} /></div>
  </div>;
}

type SlotRenderer = (host: HTMLElement, dispatch: PluginDispatch) => void | WebUiDisposer;

function Slot({ plugin, render, dispatch, slot }: { plugin: PluginConfig; render: SlotRenderer; dispatch: PluginDispatch; slot: string }): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => ref.current ? plugin.applyStyle(ref.current) : undefined, [plugin]);
  const filters = JSON.stringify(dispatch.filters);
  useEffect(() => {
    if (!ref.current) return;
    const host = ref.current;
    return mountPluginDom(host, plugin.id, slot, () => render(host, dispatch));
  }, [plugin, render, dispatch, filters, dispatch.sortBy, dispatch.sortOrder, slot]);
  return <div ref={ref} />;
}

function Panel(props: { plugin: PluginConfig } & NavigationProps): React.ReactElement {
  const { plugin } = props;
  const [state, setState] = useState<PluginState>({ page: 1, pageSize: plugin.pageSize ?? 25,
    total: 0, items: [], activeRowKey: null, activeDetail: null, filters: {},
    sortBy: plugin.defaultSortBy ?? "", sortOrder: plugin.defaultSortOrder ?? "desc", selectedIds: new Set() });
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const readState = useLatestReader(state);
  const request = useRef<AbortController | null>(null);
  const detailRequest = useRef<AbortController | null>(null);
  const startRead = useCallback(() => {
    request.current?.abort();
    detailRequest.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    return controller;
  }, []);
  const report = useCallback((error: unknown) => {
    if (!isAbortError(error)) setError(error instanceof Error ? error.message : String(error));
  }, []);
  const closeDetail = useCallback(() => {
    detailRequest.current?.abort();
    setDetailLoading(false);
    setState((state) => ({ ...state, activeRowKey: null, activeDetail: null }));
  }, []);
  const applyPage = useCallback((update: (state: PluginState) => PluginState) => {
    setState(update);
    setError(null);
  }, []);
  const dispatch = useMemo(() => makeDispatch(plugin, readState, applyPage, startRead, undefined, closeDetail, report),
    [plugin, readState, applyPage, startRead, closeDetail, report]);
  const load = useCallback(async (page?: number) => {
    const current = readState();
    const controller = startRead();
    let result: FetchPageResult;
    try {
      result = checkedPluginPage(plugin, await plugin.fetchPage({ page: page ?? current.page, pageSize: current.pageSize,
        filters: current.filters, sortBy: current.sortBy, sortOrder: current.sortOrder, signal: controller.signal }));
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      return;
    }
    if (!controller.signal.aborted) {
      setState((state) => ({ ...state, ...result, page: page ?? current.page, activeRowKey: null, activeDetail: null, selectedIds: new Set() }));
      setDetailLoading(false);
      setError(null);
    }
  }, [plugin, readState, startRead]);

  useEffect(() => {
    void load().catch(report);
    const refresh = () => void load().catch(report);
    window.addEventListener("akashic-dashboard-refresh", refresh);
    return () => {
      request.current?.abort();
      detailRequest.current?.abort();
      window.removeEventListener("akashic-dashboard-refresh", refresh);
    };
  }, [load, report]);

  const open = async (item: Record<string, unknown>) => {
    const key = String(item[plugin.rowKey] ?? "");
    if (readState().activeRowKey === key) {
      closeDetail();
      return;
    }
    detailRequest.current?.abort();
    const controller = new AbortController();
    detailRequest.current = controller;
    setDetailLoading(true);
    setState((state) => ({ ...state, activeRowKey: key, activeDetail: null }));
    setError(null);
    try {
      const detail = plugin.fetchDetail ? await plugin.fetchDetail(item, { signal: controller.signal }) : item;
      if (!controller.signal.aborted) setState((state) => ({ ...state, activeDetail: detail }));
    } catch (error) {
      if (!controller.signal.aborted) {
        setState((state) => ({ ...state, activeRowKey: null, activeDetail: null }));
        report(error);
      }
    } finally {
      if (!controller.signal.aborted) setDetailLoading(false);
    }
  };

  const hasBatch = Boolean(plugin.batchActions?.length);
  const columns = `${hasBatch ? "32px " : ""}${gridTemplate(plugin.columns)}`;
  const pageCount = Math.max(1, Math.ceil(state.total / state.pageSize));
  const workbenchLayout = plugin.layout === "workbench" && plugin.renderMain;
  return <div className="shell">
    <aside className="sessions-pane">
      <Brand />
      <ModuleSwitcher {...props} sessionsCount={props.sessionsCount} />
      <div className="explorer-body">{plugin.renderNavBody
        ? <Slot plugin={plugin} render={plugin.renderNavBody} dispatch={dispatch} slot="navigation" />
        : <div className="detail-empty"><div className="detail-empty-title">{plugin.label}</div>
          <div className="detail-empty-text">此面板没有额外导航。</div></div>}</div>
    </aside>
    <section className="content-shell">
      <header className="content-toolbar">
        <div className="content-filters">{plugin.renderFilters
          ? <Slot plugin={plugin} render={plugin.renderFilters} dispatch={dispatch} slot="filters" />
          : <div className="filter-row"><strong>{plugin.viewLabel ?? plugin.label}</strong></div>}</div>
        <div className="content-toolbar-actions"><span className="muted-text">{plugin.countTitle?.(state.total) ?? `${state.total} 条`}</span>
          <Btn size="sm" variant="ghost" onClick={() => void load().catch(report)}>刷新</Btn>
          {plugin.renderTopbarAction && <Slot plugin={plugin} render={plugin.renderTopbarAction} dispatch={dispatch} slot="topbar action" />}</div>
      </header>
      {error && <div role="alert" className="plugin-entry-error"><strong>请求失败</strong><span>{error}</span></div>}
      <main className={`workspace ${workbenchLayout ? "plugin-workbench-mode" : ""}`}>
        {workbenchLayout ? <section className="plugin-workbench-pane"><PluginMain plugin={plugin} dispatch={dispatch} /></section> : <>
          <section className="messages-pane">
            {state.selectedIds.size > 0 && <div className="batch-bar"><span>已选 {state.selectedIds.size} 条</span>
              {plugin.batchActions?.map((action) => <Btn key={action.label} size="sm" variant="secondary" className={action.className}
                onClick={() => void action.run([...state.selectedIds]).then(() => load()).catch(report)}>{action.label}</Btn>)}
              <Btn size="sm" variant="ghost" onClick={() => setState((state) => ({ ...state, selectedIds: new Set() }))}>取消选择</Btn></div>}
            <div className="table-head" style={{ gridTemplateColumns: columns }}>
              {hasBatch && <div />}
              {plugin.columns.map((column) => column.sortable
                ? <SortHead key={column.key} label={column.label} active={state.sortBy === column.key} order={state.sortOrder}
                    onClick={() => dispatch.setSort(column.key)} />
                : <div key={column.key}>{column.label}</div>)}
            </div>
            <div className="table-body">{state.items.length ? state.items.map((item) => {
              const key = String(item[plugin.rowKey] ?? "");
              const selected = state.selectedIds.has(key);
              return <div className="table-row-wrap" key={key}>
                {hasBatch && <label className="checkbox-cell"><input type="checkbox" aria-label={`选择 ${key}`} checked={selected}
                  onChange={(event) => setState((state) => {
                    const selectedIds = new Set(state.selectedIds);
                    if (event.target.checked) selectedIds.add(key); else selectedIds.delete(key);
                    return { ...state, selectedIds };
                  })} /></label>}
                <button className={`table-row table-row ${state.activeRowKey === key ? "active" : ""} ${selected ? "selected" : ""} ${plugin.rowClass?.(item) ?? ""}`}
                  style={{ gridTemplateColumns: columns }} type="button" aria-expanded={state.activeRowKey === key}
                  onClick={() => void open(item).catch(report)}>
                  {hasBatch && <span aria-hidden="true" />}
                  {plugin.columns.map((column) => column.renderCell
                    ? <span key={column.key} className={columnCellClass(column)} title={column.rawTitle ? String(item[column.key] ?? "") : undefined}
                        dangerouslySetInnerHTML={{ __html: column.renderCell(item[column.key], item) }} />
                    : <span key={column.key} className={columnCellClass(column)} title={column.rawTitle ? String(item[column.key] ?? "") : undefined}>
                        {formatPluginCell(plugin, column, item)}</span>)}
                </button>
              </div>;
            }) : <div className="empty-state">{plugin.emptyMessage ?? "暂无记录。"}</div>}</div>
            <footer className="table-foot"><div>{plugin.countTitle?.(state.total) ?? `共 ${state.total} 条`}</div>
              <div className="pager"><MaterialIconButton variant="standard" label="上一页" disabled={state.page <= 1}
                onClick={() => void load(state.page - 1).catch(report)}><ChevronLeft size={18} aria-hidden="true" /></MaterialIconButton>
                <span>{state.page} / {pageCount}</span>
                <MaterialIconButton variant="standard" label="下一页" disabled={state.page >= pageCount}
                  onClick={() => void load(state.page + 1).catch(report)}><ChevronRight size={18} aria-hidden="true" /></MaterialIconButton></div></footer>
          </section>
          <aside className={`detail-pane ${state.activeRowKey ? "is-open" : ""}`} aria-label="详情">
            {state.activeRowKey && <button className="detail-close-btn" type="button" aria-label="关闭详情" onClick={closeDetail}>
              <X size={18} aria-hidden="true" />
            </button>}
            {detailLoading ? <DetailLoading /> : state.activeRowKey
              ? plugin.renderDetail ? <PluginDetail plugin={plugin} item={state.activeDetail} dispatch={dispatch} />
                : <div className="detail-wrap"><div className="detail-toolbar"><div className="detail-title">详情</div></div>
                    <JsonView value={state.activeDetail} /></div>
              : <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">点开一条记录后，这里会显示完整字段。</div></div>}
          </aside>
        </>}
      </main>
    </section>
  </div>;
}

function SortHead(props: { label: string; active: boolean; order: SortOrder; onClick(): void }): React.ReactElement {
  return <button className={`table-sort-btn ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}>
    <span>{props.label}</span><span className="table-sort-arrow">{props.active ? props.order === "asc" ? "↑" : "↓" : ""}</span>
  </button>;
}

function DetailLoading(): React.ReactElement {
  return <div className="detail-loading" role="status" aria-label="正在加载详情">
    {React.createElement("md-linear-progress", { className: "detail-loading-progress", indeterminate: true, "aria-label": "正在加载详情" })}
    <div className="detail-loading-line detail-loading-line-short" /><div className="detail-loading-line detail-loading-line-title" />
    <div className="detail-loading-block" /><div className="detail-loading-line" /><div className="detail-loading-line" />
  </div>;
}

function detailRow(label: string, value: React.ReactNode): React.ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{label}</div><div className="detail-row-val">{value}</div></div>;
}

function gridTemplate(columns: DashboardColumn[]): string {
  return columns.map((column) => column.flex ? "minmax(0, 1fr)"
    : column.width ? `minmax(0, ${column.width}px)` : "minmax(0, auto)").join(" ");
}

function formatPluginCell(plugin: PluginConfig, column: DashboardColumn, item: Record<string, unknown>): string {
  const value = item[column.key];
  const formatter = plugin.formatters?.[column.fmt ?? ""] ?? WORKBENCH_FORMATTERS[column.fmt ?? "text"];
  return formatter ? formatter(value, item) : String(value ?? "");
}

function columnCellClass(column: DashboardColumn): string {
  const classes = [column.cellClass ?? ""];
  if (!column.cellClass && column.fmt === "text-preview") classes.push("content-preview");
  if (!column.cellClass && (column.fmt === "mono-session" || column.fmt === "mono-time")) {
    classes.push(column.fmt === "mono-session" ? "mono cell-session" : "mono cell-time");
  }
  if (column.align === "right") classes.push("align-right");
  return classes.filter(Boolean).join(" ");
}

function DashboardWorkspace({ initialPlugins }: { initialPlugins: PluginConfig[] }): React.ReactElement {
  const [pluginId, setPluginId] = useState<string | null>(null);
  const [session, setSession] = useState<string | null>(null);
  const [sessionCount, setSessionCount] = useState(0);
  const [counts, setCounts] = useState<Record<string, number | null>>({});
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    for (const plugin of initialPlugins) {
      void plugin.getCount({ signal: controller.signal }).then((count) => {
        if (count !== null && (!Number.isFinite(count) || count < 0)) throw new Error(`${plugin.id} 返回无效计数`);
        if (!controller.signal.aborted) setCounts((counts) => ({ ...counts, [plugin.id]: count }));
      }).catch((error: unknown) => {
        if (!controller.signal.aborted && !isAbortError(error)) setError(error instanceof Error ? error.message : String(error));
      });
    }
    void api<SessionPage>("/api/dashboard/sessions?limit=1", { signal: controller.signal }).then((result) => {
      if (!controller.signal.aborted) setSessionCount(result.total);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted && !isAbortError(error)) setError(error instanceof Error ? error.message : String(error));
    });
    const jump = (event: Event) => {
      const key = (event as CustomEvent<unknown>).detail;
      if (typeof key !== "string" || !key) return;
      setSession(key);
      setPluginId(null);
    };
    window.addEventListener("akashic:goto-session", jump);
    return () => {
      controller.abort();
      window.removeEventListener("akashic:goto-session", jump);
    };
  }, [initialPlugins]);
  const plugins = initialPlugins.filter((plugin) => counts[plugin.id] !== null);
  const current = plugins.find((plugin) => plugin.id === pluginId) ?? null;
  const navigation = { currentPluginId: pluginId, sessionsCount: sessionCount, plugins, counts, onSelect: setPluginId };
  return <div className="workbench-root">
    {current ? <Panel key={current.id} plugin={current} {...navigation} />
      : <Messages {...navigation} selected={session} select={setSession} />}
    {error && <div className="workbench-modal-backdrop"><div className="workbench-modal" role="alert">
      <div className="workbench-modal-title">工作台加载失败</div><div className="workbench-modal-sub">{error}</div>
      <div className="workbench-modal-actions"><Btn onClick={() => setError(null)}>关闭</Btn></div>
    </div></div>}
  </div>;
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  const releaseApi = bindApiRequest(ctx.http.request);
  const releaseEntry = ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "workbench",
    label: "工作台",
    route: "workbench",
    order: 20,
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-gauge" aria-hidden="true"><path d="m12 14 4-4"></path><path d="M3.34 19a10 10 0 1 1 17.32 0"></path></svg>',
    children: [{ id: "workbench.panels.v2", cardinality: "list" }],
    render(host: HTMLElement, view: WebEntryView): WebUiDisposer {
      const panels = view.child("workbench.panels.v2");
      const plugins = panels.entries.map((entry) => ({
        ...checkPanelEntry(entry as Record<string, unknown>),
        applyStyle: (target: HTMLElement) => panels.style(entry.id, target),
      }));
      const root = createRoot(host);
      root.render(<DashboardWorkspace initialPlugins={plugins} />);
      return () => root.unmount();
    },
  }));
  return () => {
    releaseEntry();
    releaseApi();
  };
}

function checkPanelEntry(entry: Record<string, unknown>): PluginConfig {
  const plugin = entry as unknown as PluginConfig;
  if (
    typeof plugin.id !== "string"
    || typeof plugin.label !== "string"
    || typeof plugin.rowKey !== "string"
    || !Array.isArray(plugin.columns)
    || typeof plugin.getCount !== "function"
    || typeof plugin.fetchPage !== "function"
  ) {
    throw new Error(`工作台面板合同无效: ${String(entry.id ?? "unknown")}`);
  }
  return plugin;
}
