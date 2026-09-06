import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { WebEntryView, WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type { WorkbenchUi } from "@akashic/workbench-ui-v2";
import type { FetchPageResult, PluginConfig, PluginDispatch, PluginState, SortOrder } from "./types";
import { api, bindApiRequest } from "./api";
import { formatSessionKeyForTable, shortTs, stripMarkdown } from "./format";
import { akashicBrandIcon } from "./brand";
import { PluginDetail, PluginMain, mountPluginDom } from "./PluginDetail";
import { Chip as WorkbenchChip, Grid, JsonView } from "./ui";
import { MetricTile, Sparkline, TrendChart } from "./charts";
import "./style.css";
import "./messages.css";

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
interface SessionPage { items: SessionRow[]; total: number; next_cursor: [string, string] | null }
interface MessagePage { items: MessageRow[]; through_seq: number; next_before_seq: number | null; has_more: boolean }

function Messages({ selected, select }: { selected: string | null; select(key: string): void }): React.ReactElement {
  const [sessions, setSessions] = useState<SessionPage>({ items: [], total: 0, next_cursor: null });
  const [prefix, setPrefix] = useState("");
  const [visibility, setVisibility] = useState("");
  const [page, setPage] = useState<MessagePage | null>(null);
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
    if (selected === null) { setPage(null); return; }
    const controller = new AbortController();
    messageRequest.current = controller;
    const query = new URLSearchParams({ limit: "50" });
    if (older !== null) {
      query.set("through_seq", String(older.through_seq));
      if (older.next_before_seq !== null) query.set("before_seq", String(older.next_before_seq));
    }
    let result: MessagePage;
    try {
      result = await api<MessagePage>(`/api/dashboard/sessions/${encodeURIComponent(selected)}/messages?${query}`, { signal: controller.signal });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      return;
    }
    if (!controller.signal.aborted) {
      setPage({ ...result, items: older ? [...result.items, ...older.items] : result.items });
      setError(null);
    }
  }, [selected]);
  useEffect(() => { void loadSessions().catch(report); return () => sessionRequest.current?.abort(); }, [loadSessions, report]);
  useEffect(() => { setPage(null); void loadMessages().catch(report); return () => messageRequest.current?.abort(); }, [loadMessages, report]);
  return <div className="message-inspector">
    <aside className="message-sessions" aria-label="会话目录">
      <label>会话前缀<input value={prefix} placeholder="例如 akashic:" onChange={(event) => setPrefix(event.target.value)} /></label>
      <label>可见范围<select value={visibility} onChange={(event) => setVisibility(event.target.value)}>
        <option value="">全部会话</option><option value="listed">普通会话</option><option value="internal">内部会话</option>
      </select></label>
      <div className="message-toolbar"><span>{sessions.total} 个会话</span><button onClick={() => void loadSessions().catch(report)}>刷新</button></div>
      {sessions.items.map((session) => <button type="button" key={session.key} aria-current={selected === session.key ? "true" : undefined}
        className="message-session" onClick={() => select(session.key)}>
        <strong>{session.first_message_content || session.key}</strong>
        <small>{session.key}</small><small>{session.message_count} 条 · {shortTs(session.updated_at)} · {session.attributes.visibility === "internal" ? "内部" : "普通"}</small>
      </button>)}
      {sessions.next_cursor && <button onClick={() => void loadSessions(sessions.next_cursor).catch(report)}>更多会话</button>}
    </aside>
    <main className="message-records">
      <div className="message-toolbar"><h2>{selected ?? "选择会话"}</h2>
        {selected && <button onClick={() => void loadMessages().catch(report)}>读取最新消息</button>}</div>
      <p className="message-note">消息按原始顺序保存。编辑、撤销和删除尚未接入；历史摘要记录保持保留。</p>
      {error && <p role="alert" className="message-error">{error}</p>}
      {page?.has_more && <button onClick={() => void loadMessages(page).catch(report)}>读取更早消息</button>}
      {page?.items.map((message) => <article className="message-record" key={message.id}>
        <div className="message-toolbar"><strong>#{message.seq} · {message.author}</strong>
          <span>{message.source} · {message.body.kind} · {shortTs(message.timestamp)}</span></div>
        {message.body.parts?.filter((part) => part.kind === "text" && typeof part.value === "string")
          .map((part, index) => <p className="message-text" key={index}>{String(part.value)}</p>)}
        <details><summary>原始 Message · {message.id}</summary><JsonView value={message} /></details>
      </article>)}
      {selected && page?.items.length === 0 && <p>此会话没有消息。</p>}
    </main>
  </div>;
}

type SlotRenderer = (host: HTMLElement, dispatch: PluginDispatch) => void | WebUiDisposer;
function Slot({ plugin, render, dispatch }: { plugin: PluginConfig; render: SlotRenderer; dispatch: PluginDispatch }): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => ref.current ? plugin.applyStyle(ref.current) : undefined, [plugin]);
  const filters = JSON.stringify(dispatch.filters);
  useEffect(() => {
    if (ref.current) {
      const host = ref.current;
      return mountPluginDom(host, plugin.id, "slot", () => render(host, dispatch));
    }
  }, [plugin, render, dispatch, filters, dispatch.sortBy, dispatch.sortOrder]);
  return <div ref={ref} />;
}

function Panel({ plugin }: { plugin: PluginConfig }): React.ReactElement {
  const [state, setState] = useState<PluginState>({ page: 1, pageSize: plugin.pageSize ?? 25,
    total: 0, items: [], activeRowKey: null, activeDetail: null, filters: {},
    sortBy: plugin.defaultSortBy ?? "", sortOrder: plugin.defaultSortOrder ?? "desc", selectedIds: new Set() });
  const [error, setError] = useState<string | null>(null);
  const readState = useLatestReader(state);
  const request = useRef<AbortController | null>(null);
  const detailRequest = useRef<AbortController | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => ref.current ? plugin.applyStyle(ref.current) : undefined, [plugin]);
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
      detailRequest.current?.abort();
      setState((state) => ({ ...state, ...result, page: page ?? current.page, activeRowKey: null, activeDetail: null, selectedIds: new Set() }));
      setError(null);
    }
  }, [plugin, readState, startRead]);
  useEffect(() => {
    void load().catch(report);
    const refresh = () => void load().catch(report);
    window.addEventListener("akashic-dashboard-refresh", refresh);
    return () => { request.current?.abort(); detailRequest.current?.abort(); window.removeEventListener("akashic-dashboard-refresh", refresh); };
  }, [load, report]);
  const open = async (item: Record<string, unknown>) => {
    detailRequest.current?.abort();
    const controller = new AbortController();
    detailRequest.current = controller;
    const key = String(item[plugin.rowKey]);
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
    }
  };
  const columns = [
    ...(plugin.batchActions?.length ? ["40px"] : []),
    ...plugin.columns.map((column) => column.flex ? "minmax(0, 1fr)"
      : column.width ? `minmax(0, ${column.width}px)` : "minmax(0, auto)"),
  ].join(" ");
  return <div ref={ref} className="message-panel">
    <div className="message-toolbar"><h2>{plugin.viewLabel ?? plugin.label}</h2><span>{plugin.countTitle?.(state.total) ?? `${state.total} 条`}</span>
      <button onClick={() => void load().catch(report)}>刷新</button>
      {plugin.renderTopbarAction && <Slot plugin={plugin} render={plugin.renderTopbarAction} dispatch={dispatch} />}</div>
    {error && <p role="alert" className="message-error">{error}</p>}
    {plugin.renderNavBody && <Slot plugin={plugin} render={plugin.renderNavBody} dispatch={dispatch} />}
    {plugin.renderFilters && <Slot plugin={plugin} render={plugin.renderFilters} dispatch={dispatch} />}
    {plugin.layout === "workbench" && plugin.renderMain ? <PluginMain plugin={plugin} dispatch={dispatch} /> : <>
      <div className="message-toolbar">{plugin.batchActions?.map((action) => <button key={action.label} className={action.className}
        disabled={!state.selectedIds.size} onClick={() => {
          if (window.confirm(`${action.label}：已选择 ${state.selectedIds.size} 项，是否继续？`)) {
            void action.run([...state.selectedIds]).then(() => load()).catch(report);
          }
        }}>{action.label}</button>)}</div>
      <div className="message-table-scroll"><table role="table"><thead role="rowgroup"><tr role="row" style={{ gridTemplateColumns: columns }}>
        {plugin.batchActions?.length ? <th>选择</th> : null}
        {plugin.columns.map((column) => <th key={column.key} style={{ textAlign: column.align }}>
          {!column.sortable ? column.label : <button onClick={() => dispatch.setSort(column.key)}>{column.label}{state.sortBy === column.key ? state.sortOrder === "asc" ? " ↑" : " ↓" : ""}</button>}
        </th>)}
      </tr></thead><tbody>{state.items.map((item) => <tr role="row" key={String(item[plugin.rowKey])} style={{ gridTemplateColumns: columns }} className={plugin.rowClass?.(item)}>
        {plugin.batchActions?.length ? <td><input type="checkbox" aria-label={`选择 ${String(item[plugin.rowKey])}`} checked={state.selectedIds.has(String(item[plugin.rowKey]))}
          onChange={(event) => setState((state) => {
            const selectedIds = new Set(state.selectedIds);
            if (event.target.checked) selectedIds.add(String(item[plugin.rowKey])); else selectedIds.delete(String(item[plugin.rowKey]));
            return { ...state, selectedIds };
          })} /></td> : null}
        {plugin.columns.map((column, index) => {
          const value = item[column.key];
          const format = plugin.formatters?.[column.fmt ?? ""] ?? WORKBENCH_FORMATTERS[column.fmt ?? "text"];
          // v2 的自定义 renderer 自己转义 HTML；普通 formatter 仍是文本。
          const cell = column.renderCell
            ? <span dangerouslySetInnerHTML={{ __html: column.renderCell(value, item) }} />
            : format ? format(value, item) : String(value ?? "");
          return <td key={column.key} className={column.cellClass} style={{ textAlign: column.align }} title={column.rawTitle ? String(value) : undefined}>
            {index === 0 ? <button onClick={() => void open(item).catch(report)}>{cell}</button> : cell}</td>;
        })}
      </tr>)}</tbody></table></div>
      {!state.items.length && <p>{plugin.emptyMessage ?? "没有记录。"}</p>}
      <div className="message-toolbar"><button disabled={state.page <= 1} onClick={() => void load(state.page - 1).catch(report)}>上一页</button>
        <span>第 {state.page} 页</span><button disabled={state.page * state.pageSize >= state.total} onClick={() => void load(state.page + 1).catch(report)}>下一页</button></div>
    </>}
    {state.activeRowKey && <section className="message-panel-detail"><button onClick={closeDetail}>关闭详情</button>
      {state.activeDetail ? plugin.renderDetail ? <PluginDetail plugin={plugin} item={state.activeDetail} dispatch={dispatch} /> : <JsonView value={state.activeDetail} /> : <p>正在读取…</p>}
    </section>}
  </div>;
}

function DashboardWorkspace({ initialPlugins }: { initialPlugins: PluginConfig[] }): React.ReactElement {
  const [pluginId, setPluginId] = useState<string | null>(null);
  const [session, setSession] = useState<string | null>(null);
  const [counts, setCounts] = useState<Record<string, number | null>>({});
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    for (const plugin of initialPlugins) {
      void plugin.getCount({ signal: controller.signal }).then((count) => {
        if (count !== null && (!Number.isFinite(count) || count < 0)) throw new Error(`${plugin.id} 返回无效计数`);
        if (!controller.signal.aborted) setCounts((counts) => ({ ...counts, [plugin.id]: count }));
      }).catch((error: unknown) => { if (!controller.signal.aborted && !isAbortError(error)) setError(error instanceof Error ? error.message : String(error)); });
    }
    const jump = (event: Event) => {
      const key = (event as CustomEvent<unknown>).detail;
      if (typeof key !== "string" || !key) return;
      setSession(key); setPluginId(null);
    };
    window.addEventListener("akashic:goto-session", jump);
    return () => { controller.abort(); window.removeEventListener("akashic:goto-session", jump); };
  }, [initialPlugins]);
  const current = initialPlugins.find((plugin) => plugin.id === pluginId);
  return <div className="workbench-root message-workbench">
    <header className="message-workbench-nav"><img src={akashicBrandIcon} alt="" /><strong>工作台</strong>
      <button aria-current={pluginId === null ? "page" : undefined} onClick={() => setPluginId(null)}>消息</button>
      {initialPlugins.filter((plugin) => counts[plugin.id] !== null).map((plugin) => <button key={plugin.id}
        aria-current={pluginId === plugin.id ? "page" : undefined} onClick={() => setPluginId(plugin.id)}>{plugin.label}</button>)}
    </header>
    {error && <p role="alert" className="message-error">{error}</p>}
    {current ? <Panel key={current.id} plugin={current} /> : <Messages selected={session} select={setSession} />}
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
