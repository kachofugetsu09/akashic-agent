import React, { useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Dialog from "@radix-ui/react-dialog";
import { ChevronDown, ChevronLeft, ChevronRight, X } from "lucide-react";
import "./style.css";
import { akashicBrandIcon } from "./brand";
import { api, asPageResult, bindApiRequest, interactionDeleteRequirement, pageCount } from "./api";
import {
  encodePath,
  formatSessionKeyForTable,
  formatTokens,
  relativeTime,
  roleClass,
  shortTs,
  stripMarkdown,
} from "./format";
import { MaterialIconButton } from "@akashic/web-ui-v1";
import { mountPluginDom, PluginDetail, PluginMain } from "./PluginDetail";
import { Btn, Chip as WorkbenchChip, Grid, JsonView, Markdown } from "./ui";
import { MetricTile, Sparkline, TrendChart } from "./charts";
import type { WebEntryView, WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type { WorkbenchUi } from "@akashic/workbench-ui-v2";
import type {
  CompactionDetail,
  DashboardColumn,
  FetchPageResult,
  MessageRow,
  PluginBatchAction,
  PluginConfig,
  PluginDispatch,
  PluginState,
  SessionRow,
  SortOrder,
  ViewMode,
} from "./types";

const notificationIcon = akashicBrandIcon;
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

// Creates a PluginDispatch bound to the given plugin + latest state getter.
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
    const result = checkedPluginPage(plugin, await plugin.fetchPage({
      page: 1,
      pageSize: state.pageSize,
      filters: nextFilters,
      sortBy: nextSortBy,
      sortOrder: nextSortOrder,
      signal: controller.signal,
    }));
    if (controller.signal.aborted) return;
    onSetState((s) => ({
      ...s,
      page: 1,
      total: result.total,
      items: result.items,
      activeRowKey: null,
      activeDetail: null,
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

function DashboardWorkspace({ initialPlugins }: { initialPlugins: PluginConfig[] }): React.ReactElement {
  const [workbenchRoot, setWorkbenchRoot] = useState<HTMLDivElement | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("sessions");
  const plugins = initialPlugins;
  const [pluginState, setPluginState] = useState<Record<string, PluginState>>({});
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionChannel, setSessionChannel] = useState("");
  const [expandedSessionGroups, setExpandedSessionGroups] = useState<Record<string, boolean>>({
    scheduler: false,
    programmatic: false,
  });
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionRow | null>(null);
  const [compaction, setCompaction] = useState<CompactionDetail | null>(null);
  const [compactionPending, setCompactionPending] = useState(false);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [messageSearch, setMessageSearch] = useState("");
  const [messageRole, setMessageRole] = useState("");
  const [messagePage, setMessagePage] = useState(1);
  const [messageSortBy, setMessageSortBy] = useState("ts");
  const [messageSortOrder, setMessageSortOrder] = useState<SortOrder>("desc");
  const [totalMessages, setTotalMessages] = useState(0);
  const [activeMessage, setActiveMessage] = useState<MessageRow | null>(null);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set());
  const [hiddenPlugins, setHiddenPlugins] = useState<Record<string, boolean>>({});
  const [pendingPluginDetailKey, setPendingPluginDetailKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const compactionRequestRef = useRef<AbortController | null>(null);
  const pluginReadRequestsRef = useRef(new Map<string, AbortController>());
  const pluginDetailRequestRef = useRef<AbortController | null>(null);

  const messagePageSize = 25;
  const currentPluginId = viewMode.startsWith("plugin:") ? viewMode.slice(7) : "";
  const currentPlugin = plugins.find((plugin) => plugin.id === currentPluginId) ?? null;
  const currentPluginState = currentPluginId ? pluginState[currentPluginId] : null;
  const hasCurrentPluginState = Boolean(currentPluginState);
  const currentPluginLayout = currentPlugin?.layout ?? "table";
  const readPluginState = useLatestReader(pluginState);
  const setPluginStateFor = useCallback((pluginId: string, updater: (s: PluginState) => PluginState): void => {
    setPluginState((current) => {
      const state = current[pluginId];
      if (!state) return current;
      return { ...current, [pluginId]: updater(state) };
    });
  }, []);
  const activatePlugin = useCallback((pluginId: string): void => {
    setViewMode(`plugin:${pluginId}`);
  }, []);
  const closePlugin = useCallback((pluginId: string): void => {
    pluginDetailRequestRef.current?.abort();
    pluginDetailRequestRef.current = null;
    setPendingPluginDetailKey(null);
    setPluginState((current) => {
      const state = current[pluginId];
      if (!state) return current;
      return { ...current, [pluginId]: { ...state, activeRowKey: null, activeDetail: null } };
    });
  }, []);

  const startPluginRead = useCallback((pluginId: string): AbortController => {
    pluginReadRequestsRef.current.get(pluginId)?.abort();
    const controller = new AbortController();
    pluginReadRequestsRef.current.set(pluginId, controller);
    return controller;
  }, []);

  const channels = useMemo(() => Array.from(new Set(sessions.map((session) => session.key.split(":")[0]).filter(Boolean))), [sessions]);
  const ordinarySessions = useMemo(
    () => sessions.filter((session) => !isFoldedSession(session)),
    [sessions],
  );
  const schedulerSessions = useMemo(
    () => sessions.filter((session) => sessionChannelOf(session) === "scheduler"),
    [sessions],
  );
  const programmaticSessions = useMemo(
    () => sessions.filter((session) => sessionChannelOf(session) === "programmatic"),
    [sessions],
  );

  const reportError = useCallback((exc: unknown): void => {
    if (isAbortError(exc)) return;
    console.error("[dashboard] request failed", exc);
    setError(exc instanceof Error ? exc.message : String(exc));
  }, []);

  const run = useCallback(async (work: () => Promise<void>) => {
    try {
      setError(null);
      await work();
    } catch (exc) {
      reportError(exc);
    }
  }, [reportError]);

  const deleteSelectedMessages = useCallback(async (ids: string[]): Promise<boolean> => {
    const pending = new Set(ids);
    while (pending.size > 0) {
      try {
        await api("/api/dashboard/messages/batch-delete", {
          method: "POST",
          body: JSON.stringify({ ids: [...pending] }),
        });
        return true;
      } catch (exc) {
        const requirement = interactionDeleteRequirement(exc);
        if (!requirement) throw exc;
        if (!window.confirm("所选消息属于一次完整交互。继续会撤销这一轮的全部用户输入和最终回复，是否继续？")) {
          return false;
        }
        const deletion = await api<{ message_ids: string[] }>(
          `/api/dashboard/interactions/${encodePath(requirement.control_turn_id)}`,
          { method: "DELETE" },
        );
        for (const messageId of deletion.message_ids) pending.delete(messageId);
        if (pending.has(requirement.message_id)) {
          throw new Error("整轮撤销响应未包含触发删除的消息", { cause: exc });
        }
      }
    }
    return true;
  }, []);

  const loadSessions = useCallback(async () => {
    sessionsRequestRef.current?.abort();
    const controller = new AbortController();
    sessionsRequestRef.current = controller;
    const params = new URLSearchParams();
    if (sessionSearch) params.set("q", sessionSearch);
    if (sessionChannel) params.set("channel", sessionChannel);
    params.set("page_size", "200");
    try {
      const payload = asPageResult<SessionRow>(await api(`/api/dashboard/sessions?${params.toString()}`, { signal: controller.signal }));
      setSessions(payload.items);
      setActiveSession((current) => {
        if (!current) return null;
        return payload.items.find((session) => session.key === current.key) ?? null;
      });
    } finally {
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null;
    }
  }, [sessionChannel, sessionSearch]);

  const loadMessages = useCallback(async () => {
    messagesRequestRef.current?.abort();
    const controller = new AbortController();
    messagesRequestRef.current = controller;
    const params = new URLSearchParams();
    if (activeSessionKey) params.set("session_key", activeSessionKey);
    if (messageSearch) params.set("q", messageSearch);
    if (messageRole) params.set("role", messageRole);
    params.set("page", String(messagePage));
    params.set("page_size", String(messagePageSize));
    params.set("sort_by", messageSortBy);
    params.set("sort_order", messageSortOrder);
    try {
      const payload = asPageResult<MessageRow>(await api(`/api/dashboard/messages?${params.toString()}`, { signal: controller.signal }));
      setMessages(payload.items);
      setTotalMessages(payload.total);
      setActiveMessage((current) => current && payload.items.some((item) => item.id === current.id) ? current : null);
    } finally {
      if (messagesRequestRef.current === controller) messagesRequestRef.current = null;
    }
  }, [activeSessionKey, messagePage, messageRole, messageSearch, messageSortBy, messageSortOrder]);

  const loadCompaction = useCallback(async () => {
    compactionRequestRef.current?.abort();
    if (!activeSessionKey) {
      compactionRequestRef.current = null;
      setCompaction(null);
      return;
    }
    const controller = new AbortController();
    compactionRequestRef.current = controller;
    setCompactionPending(true);
    try {
      const payload = await api<CompactionDetail>(
        `/api/dashboard/sessions/${encodePath(activeSessionKey)}/compaction`,
        { signal: controller.signal },
      );
      if (controller.signal.aborted) return;
      setCompaction(payload);
    } finally {
      if (compactionRequestRef.current === controller) {
        compactionRequestRef.current = null;
        setCompactionPending(false);
      }
    }
  }, [activeSessionKey]);

  const loadPluginPanel = useCallback(async (pluginId: string) => {
    const plugin = plugins.find((item) => item.id === pluginId);
    const state = readPluginState()[pluginId];
    if (!plugin || !state) return;
    const controller = startPluginRead(pluginId);
    const result = checkedPluginPage(plugin, await plugin.fetchPage({
      page: state.page,
      pageSize: state.pageSize,
      filters: state.filters,
      sortBy: state.sortBy,
      sortOrder: state.sortOrder,
      signal: controller.signal,
    }));
    if (controller.signal.aborted) return;
    setPluginState((current) => ({
      ...current,
      [pluginId]: {
        ...current[pluginId],
        total: result.total,
        items: result.items,
        activeRowKey: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeRowKey
          : null,
        activeDetail: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeDetail
          : null,
      },
    }));
  }, [plugins, readPluginState, startPluginRead]);

  const refreshCurrentView = useCallback(async () => {
    await loadSessions();
    if (viewMode === "compaction") {
      await loadCompaction();
    } else if (viewMode.startsWith("plugin:")) {
      await loadPluginPanel(viewMode.slice(7));
    } else {
      await loadMessages();
    }
  }, [loadCompaction, loadMessages, loadPluginPanel, loadSessions, viewMode]);

  useEffect(() => {
    const refresh = (): void => {
      void run(refreshCurrentView);
    };
    window.addEventListener("akashic-dashboard-refresh", refresh);
    return () => window.removeEventListener("akashic-dashboard-refresh", refresh);
  }, [refreshCurrentView, run]);

  useEffect(() => () => {
    sessionsRequestRef.current?.abort();
    messagesRequestRef.current?.abort();
    compactionRequestRef.current?.abort();
    pluginDetailRequestRef.current?.abort();
    for (const controller of pluginReadRequestsRef.current.values()) controller.abort();
    pluginReadRequestsRef.current.clear();
  }, []);

  useEffect(() => {
    setPluginState(Object.fromEntries(initialPlugins.map((plugin) => [plugin.id, {
      page: 1,
      pageSize: plugin.pageSize || 25,
      total: 0,
      items: [],
      activeRowKey: null,
      activeDetail: null,
      filters: {},
      sortBy: plugin.defaultSortBy ?? "",
      sortOrder: plugin.defaultSortOrder ?? "desc",
      selectedIds: new Set<string>(),
    }])));
  }, [initialPlugins]);

  useEffect(() => {
    void run(loadSessions);
  }, [loadSessions, run]);

  useEffect(() => {
    for (const plugin of plugins) {
      void run(async () => {
        const controller = startPluginRead(plugin.id);
        const count = await plugin.getCount({ signal: controller.signal });
        if (controller.signal.aborted) return;
        if (count === null) {
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: true }));
        } else {
          if (!Number.isFinite(count) || count < 0) {
            throw new Error(`插件 ${plugin.id} 返回了无效计数`);
          }
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: false }));
          setPluginState((current) => ({
            ...current,
            [plugin.id]: { ...current[plugin.id], total: count },
          }));
        }
      });
    }
  }, [plugins, run, startPluginRead]);

  const focusView = useCallback((next: ViewMode): void => {
    setViewMode(next);
  }, []);

  const selectView = (next: ViewMode): void => {
    focusView(next);
  };

  const gotoSession = useEffectEvent((key: string): void => {
    setActiveSessionKey(key);
    setActiveSession(sessions.find((session) => session.key === key) ?? null);
    setActiveMessage(null);
    setMessagePage(1);
    const channel = sessionChannelOf({ key });
    if (channel === "scheduler" || channel === "programmatic") {
      setExpandedSessionGroups((current) => ({ ...current, [channel]: true }));
    }
    selectView("sessions");
  });

  // 插件面板（如 observe 错误排障台）通过 CustomEvent 请求跳到某个 session 的对话现场。
  useEffect(() => {
    const onGoto = (e: Event): void => {
      const key = (e as CustomEvent<string>).detail;
      if (!key) return;
      gotoSession(key);
    };
    window.addEventListener("akashic:goto-session", onGoto);
    return () => window.removeEventListener("akashic:goto-session", onGoto);
  }, []);

  const sort = (key: string): void => {
    const flip = (currentKey: string, currentOrder: SortOrder): SortOrder => currentKey === key && currentOrder === "desc" ? "asc" : "desc";
    setMessageSortOrder(flip(messageSortBy, messageSortOrder));
    setMessageSortBy(key);
    setMessagePage(1);
  };

  // 必要 effect：按当前视图加载对应面板数据（fetch 有副作用，React 官方认可 effect 做数据获取）
  useEffect(() => {
    if (viewMode === "sessions") void run(loadMessages);
  }, [loadMessages, run, viewMode]);

  useEffect(() => {
    if (viewMode === "compaction") void run(loadCompaction);
  }, [loadCompaction, run, viewMode]);

  useEffect(() => {
    if (viewMode.startsWith("plugin:")) void run(() => loadPluginPanel(viewMode.slice(7)));
  }, [loadPluginPanel, run, viewMode]);

  const currentPageCount = currentPluginState
    ? pageCount(currentPluginState.total, currentPluginState.pageSize)
    : pageCount(totalMessages, messagePageSize);

  const currentPage = currentPluginState?.page ?? messagePage;

  const changePage = (delta: number): void => {
    if (currentPage + delta < 1 || currentPage + delta > currentPageCount) return;
    if (currentPluginId) {
      void run(async () => {
        const plugin = plugins.find((item) => item.id === currentPluginId);
        const state = pluginState[currentPluginId];
        if (!plugin || !state) return;
        const nextPage = state.page + delta;
        const controller = startPluginRead(currentPluginId);
        const result = checkedPluginPage(plugin, await plugin.fetchPage({
          page: nextPage,
          pageSize: state.pageSize,
          filters: state.filters,
          sortBy: state.sortBy,
          sortOrder: state.sortOrder,
          signal: controller.signal,
        }));
        if (controller.signal.aborted) return;
        setPluginState((current) => ({
          ...current,
          [currentPluginId]: {
            ...current[currentPluginId],
            page: nextPage,
            total: result.total,
            items: result.items,
            activeRowKey: null,
            activeDetail: null,
          },
        }));
      });
    } else setMessagePage((page) => page + delta);
  };

  // Batch count: messages or plugin selectedIds
  const pluginBatchCount = currentPluginState?.selectedIds.size ?? 0;
  const batchCount = viewMode.startsWith("plugin:") ? pluginBatchCount : selectedMessageIds.size;

  // dispatch for current plugin (used in DetailPane and batch bar)
  // 插件 dispatch 是宿主持有的稳定能力；事件读取最新状态，DOM renderer 不重复初始化。
  const currentDispatch = useMemo(() => currentPlugin && hasCurrentPluginState
    ? makeDispatch(
        currentPlugin,
        () => readPluginState()[currentPlugin.id] ?? null,
        (updater) => setPluginStateFor(currentPlugin.id, updater),
        () => startPluginRead(currentPlugin.id),
        () => activatePlugin(currentPlugin.id),
        () => closePlugin(currentPlugin.id),
        reportError,
      )
    : undefined, [activatePlugin, closePlugin, currentPlugin, hasCurrentPluginState, readPluginState, reportError, setPluginStateFor, startPluginRead]);
  const isPluginWorkbench = Boolean(
    currentPlugin
      && currentPluginState
      && currentDispatch
      && currentPluginLayout === "workbench"
      && currentPlugin.renderMain,
  );
  const detailOpen = viewMode.startsWith("plugin:")
    ? Boolean(currentPluginState?.activeRowKey)
    : viewMode === "compaction"
      ? false
      : Boolean(activeMessage || activeSession);

  return (
    <div ref={setWorkbenchRoot} className="workbench-root">
    <div className="shell">
      <aside className="sessions-pane">
        <div className="brand">
          <img className="brand-mark" src={notificationIcon} alt="" />
          <div>
            <div className="brand-title">Akashic</div>
            <div className="brand-sub">Dashboard</div>
          </div>
        </div>
        <ModuleSwitcher
            viewMode={viewMode}
            sessionsCount={sessions.length}
            plugins={plugins.filter((plugin) => !hiddenPlugins[plugin.id])}
            pluginState={pluginState}
            onSelect={(next) => {
              if (next === "sessions") {
                setActiveSessionKey(null);
                setActiveSession(null);
                setActiveMessage(null);
                setMessagePage(1);
              }
              selectView(next);
            }}
        />

          <div className="explorer-body">
            {(viewMode === "sessions" || viewMode === "compaction") && (
              <>
                <div className="filters-stack session-filters">
                  <label className="search search-small">
                    <span aria-hidden="true">⌕</span>
                    <input aria-label="搜索会话" type="text" placeholder="搜索会话" value={sessionSearch} onChange={(event) => setSessionSearch(event.target.value.trim())} />
                  </label>
                  <select aria-label="会话来源" value={sessionChannel} onChange={(event) => {
                    const channel = event.target.value;
                    setSessionChannel(channel);
                    if (channel === "scheduler" || channel === "programmatic") {
                      setExpandedSessionGroups((current) => ({ ...current, [channel]: true }));
                    }
                  }}>
                    <option value="">全部来源</option>
                    {channels.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
                  </select>
                </div>
                <div className="session-list">
                  <button className={`all-messages-row ${!activeSessionKey ? "active" : ""}`} type="button" onClick={() => {
                    setActiveSessionKey(null);
                    setActiveSession(null);
                    setActiveMessage(null);
                    setMessagePage(1);
                    selectView(viewMode === "compaction" ? "compaction" : "sessions");
                  }}>
                    <span>全部会话</span><strong>{sessions.length}</strong>
                  </button>
                  {ordinarySessions.map((session) => (
                    <SessionNavItem
                      key={session.key}
                      session={session}
                      active={activeSessionKey === session.key}
                      onSelect={() => {
                        setActiveSessionKey(session.key);
                        setActiveSession(session);
                        setActiveMessage(null);
                        setMessagePage(1);
                        selectView(viewMode === "compaction" ? "compaction" : "sessions");
                      }}
                    />
                  ))}
                  <SessionGroup
                    id="scheduler-sessions"
                    label="定时任务"
                    sessions={schedulerSessions}
                    open={expandedSessionGroups.scheduler}
                    activeSessionKey={activeSessionKey}
                    onOpenChange={() => setExpandedSessionGroups((current) => ({ ...current, scheduler: !current.scheduler }))}
                    onSelect={(session) => {
                      setActiveSessionKey(session.key);
                      setActiveSession(session);
                      setActiveMessage(null);
                      setMessagePage(1);
                      selectView(viewMode === "compaction" ? "compaction" : "sessions");
                    }}
                  />
                  <SessionGroup
                    id="programmatic-sessions"
                    label="程序会话"
                    sessions={programmaticSessions}
                    open={expandedSessionGroups.programmatic}
                    activeSessionKey={activeSessionKey}
                    onOpenChange={() => setExpandedSessionGroups((current) => ({ ...current, programmatic: !current.programmatic }))}
                    onSelect={(session) => {
                      setActiveSessionKey(session.key);
                      setActiveSession(session);
                      setActiveMessage(null);
                      setMessagePage(1);
                      selectView(viewMode === "compaction" ? "compaction" : "sessions");
                    }}
                  />
                </div>
              </>
            )}

            {viewMode.startsWith("plugin:") && currentPlugin && currentPluginState && currentPlugin.renderNavBody && (
              <PluginSlot
                plugin={currentPlugin}
                pluginId={currentPlugin.id}
                render={currentPlugin.renderNavBody}
                slot="navigation"
                redrawOnTotal={currentPluginState.total}
                state={currentPluginState}
                onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
                startRead={() => startPluginRead(currentPlugin.id)}
                onActivate={() => focusView(`plugin:${currentPlugin.id}`)}
                onError={reportError}
              />
            )}
        </div>
      </aside>

      <section className="content-shell">
        <header className="content-toolbar">
          <ContentFilters
            viewMode={viewMode}
            messageSearch={messageSearch}
            setMessageSearch={(value) => { setMessageSearch(value); setMessagePage(1); }}
            messageRole={messageRole}
            setMessageRole={(value) => { setMessageRole(value); setMessagePage(1); }}
            activeSessionKey={activeSessionKey}
            clearSession={() => { setActiveSessionKey(null); setActiveSession(null); setActiveMessage(null); setMessagePage(1); }}
            currentPlugin={currentPlugin}
            currentPluginState={currentPluginState}
            onSetPluginState={currentPlugin ? (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })) : undefined}
            startPluginRead={startPluginRead}
            onError={reportError}
          />
          {viewMode.startsWith("plugin:") && currentPlugin?.renderTopbarAction && currentPluginState && currentDispatch && (
            <div className="content-toolbar-actions">
              <PluginSlot
                plugin={currentPlugin}
                pluginId={currentPlugin.id}
                render={currentPlugin.renderTopbarAction}
                slot="topbar action"
                state={currentPluginState}
                onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
                startRead={() => startPluginRead(currentPlugin.id)}
                onActivate={() => focusView(`plugin:${currentPlugin.id}`)}
                onError={reportError}
              />
            </div>
          )}
        </header>

        <main className={`workspace${isPluginWorkbench ? " plugin-workbench-mode" : ""}`}>

        {isPluginWorkbench && currentPlugin && currentDispatch ? (
          <section className="plugin-workbench-pane">
            <PluginMain plugin={currentPlugin} dispatch={currentDispatch} />
          </section>
        ) : (
          <>
            <section className="messages-pane">
              {viewMode === "compaction" ? (
                <CompactionView
                  compaction={compaction}
                  pending={compactionPending}
                  activeSessionKey={activeSessionKey}
                />
              ) : (
              <>
              {batchCount > 0 && (
                <div className="batch-bar">
                  <span>已选 {batchCount} 条</span>
                  {viewMode.startsWith("plugin:") && currentPlugin?.batchActions && currentPluginState
                    ? currentPlugin.batchActions.map((action: PluginBatchAction) => (
                        <button key={action.label} className={action.className} type="button" onClick={() => void run(async () => {
                          const ids = [...currentPluginState.selectedIds];
                          await action.run(ids);
                          setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                          await loadPluginPanel(currentPlugin.id);
                        })}>{action.label}</button>
                      ))
                    : <Btn size="sm" variant="danger" onClick={() => void run(async () => {
                        if (!await deleteSelectedMessages([...selectedMessageIds])) return;
                        setSelectedMessageIds(new Set());
                        await refreshCurrentView();
                      })}>批量删除</Btn>
                  }
                  <Btn size="sm" variant="ghost" onClick={() => {
                    if (viewMode.startsWith("plugin:") && currentPlugin) {
                      setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                    } else {
                      setSelectedMessageIds(new Set());
                    }
                  }}>取消选择</Btn>
                </div>
              )}
              <TableHead viewMode={viewMode} plugin={currentPlugin} pluginState={currentPluginState} messageSortBy={messageSortBy} messageSortOrder={messageSortOrder} onSort={sort} onPluginSort={currentDispatch ? (key) => currentDispatch.setSort(key) : undefined} />
              <div className="table-body">
                <Rows
                  viewMode={viewMode}
                  messages={messages}
                  plugin={currentPlugin}
                  pluginState={currentPluginState}
                  selectedMessageIds={selectedMessageIds}
                  activeMessage={activeMessage}
                  onSelectMessage={(msg) => setActiveMessage((current) => current?.id === msg.id ? null : msg)}
                  onSelectPluginRow={(row) => {
                    if (!currentPlugin) return;
                    const key = String(row[currentPlugin.rowKey] ?? "");
                    pluginDetailRequestRef.current?.abort();
                    const closing = currentPluginState?.activeRowKey === key;
                    setPendingPluginDetailKey(closing ? null : `${currentPlugin.id}:${key}`);
                    setPluginState((c) => {
                      const ps = c[currentPlugin.id];
                      if (!ps) return c;
                      return { ...c, [currentPlugin.id]: { ...ps, activeRowKey: closing ? null : key, activeDetail: null } };
                    });
                    if (closing) {
                      pluginDetailRequestRef.current = null;
                      return;
                    }
                    const controller = new AbortController();
                    pluginDetailRequestRef.current = controller;
                    void (async () => {
                      try {
                        const detail = currentPlugin.fetchDetail
                          ? await currentPlugin.fetchDetail(row, { signal: controller.signal })
                          : row;
                        if (controller.signal.aborted || pluginDetailRequestRef.current !== controller) return;
                        setPluginState((current) => {
                          const state = current[currentPlugin.id];
                          if (!state || state.activeRowKey !== key) return current;
                          return { ...current, [currentPlugin.id]: { ...state, activeDetail: detail } };
                        });
                      } catch (exc) {
                        if (pluginDetailRequestRef.current === controller) reportError(exc);
                      } finally {
                        if (pluginDetailRequestRef.current === controller) {
                          pluginDetailRequestRef.current = null;
                          setPendingPluginDetailKey(null);
                        }
                      }
                    })();
                  }}
                  onTogglePluginRow={(id) => {
                    if (!currentPlugin) return;
                    setPluginState((c) => {
                      const ps = c[currentPlugin.id];
                      if (!ps) return c;
                      const next = new Set(ps.selectedIds);
                      if (next.has(id)) next.delete(id);
                      else next.add(id);
                      return { ...c, [currentPlugin.id]: { ...ps, selectedIds: next } };
                    });
                  }}
                  setSelectedMessageIds={setSelectedMessageIds}
                />
              </div>
               <footer className="table-foot">
                 <div>{tableMeta(totalMessages, currentPlugin, currentPluginState)}</div>
                 <div className="pager">
                   <MaterialIconButton variant="standard" label="上一页" disabled={currentPage <= 1} onClick={() => changePage(-1)}><ChevronLeft size={18} aria-hidden="true" /></MaterialIconButton>
                   <span>{currentPage} / {currentPageCount}</span>
                   <MaterialIconButton variant="standard" label="下一页" disabled={currentPage >= currentPageCount} onClick={() => changePage(1)}><ChevronRight size={18} aria-hidden="true" /></MaterialIconButton>
                 </div>
               </footer>
              </>
              )}
            </section>

            <aside className={`detail-pane${detailOpen ? " is-open" : ""}`} aria-label="详情">
              <DetailPane
                viewMode={viewMode}
                activeSession={activeSession}
                activeMessage={activeMessage}
                plugin={currentPlugin}
                pluginState={currentPluginState}
                loading={Boolean(currentPlugin && currentPluginState?.activeRowKey && pendingPluginDetailKey === `${currentPlugin.id}:${currentPluginState.activeRowKey}`)}
                dispatch={currentDispatch}
                onClose={() => {
                  setActiveSession(null);
                  setActiveMessage(null);
                  if (currentPlugin) {
                    pluginDetailRequestRef.current?.abort();
                    pluginDetailRequestRef.current = null;
                    setPendingPluginDetailKey(null);
                    setPluginState(c => {
                      const ps = c[currentPlugin.id];
                      if (!ps) return c;
                      return { ...c, [currentPlugin.id]: { ...ps, activeRowKey: null, activeDetail: null } };
                    });
                  }
                }}
              />
            </aside>
          </>
        )}
        </main>
      </section>
      <Dialog.Root open={Boolean(error)} onOpenChange={(open) => { if (!open) setError(null); }}>
        <Dialog.Portal container={workbenchRoot}>
          <Dialog.Overlay className="workbench-modal-backdrop" />
          <Dialog.Content className="workbench-modal" aria-describedby="dashboard-error-description">
            <Dialog.Title className="workbench-modal-title">请求失败</Dialog.Title>
            <Dialog.Description id="dashboard-error-description" className="workbench-modal-sub">{error}</Dialog.Description>
            <div className="workbench-modal-actions">
              <Btn onClick={() => setError(null)}>关闭</Btn>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
    </div>
  );
}

function SessionGroup(props: {
  id: string;
  label: string;
  sessions: SessionRow[];
  open: boolean;
  activeSessionKey: string | null;
  onOpenChange(): void;
  onSelect(session: SessionRow): void;
}): React.ReactElement | null {
  if (!props.sessions.length) return null;
  return (
    <div className={`nav-group session-group ${props.open ? "open" : ""}`}>
      <button
        className="nav-group-toggle"
        type="button"
        aria-expanded={props.open}
        aria-controls={props.id}
        onClick={props.onOpenChange}
      >
        <span className="nav-group-caret" aria-hidden="true">›</span>
        <span className="nav-group-label">{props.label}</span>
        <span className="nav-group-count">{props.sessions.length}</span>
      </button>
      <div
        id={props.id}
        className={`nav-group-body ${props.open ? "open" : ""}`}
        hidden={!props.open}
      >
        <div className="nav-group-body-inner">
          {props.sessions.map((session) => (
            <SessionNavItem
              key={session.key}
              session={session}
              active={props.activeSessionKey === session.key}
              nested
              onSelect={() => props.onSelect(session)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ModuleSwitcher(props: {
  viewMode: ViewMode;
  sessionsCount: number;
  plugins: PluginConfig[];
  pluginState: Record<string, PluginState>;
  onSelect(next: ViewMode): void;
}): React.ReactElement {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const currentPlugin = props.viewMode.startsWith("plugin:")
    ? props.plugins.find((plugin) => `plugin:${plugin.id}` === props.viewMode) ?? null
    : null;
  const currentLabel = props.viewMode === "sessions" ? "Sessions" : currentPlugin?.label ?? "Explorer";
  const currentCount = props.viewMode === "sessions"
    ? props.sessionsCount
    : currentPlugin ? props.pluginState[currentPlugin.id]?.total ?? 0 : 0;

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

  const select = (next: ViewMode): void => {
    props.onSelect(next);
    setOpen(false);
    queueMicrotask(() => triggerRef.current?.focus());
  };

  return (
    <div className="module-switcher" ref={rootRef}>
      <button
        ref={triggerRef}
        className="module-switcher-trigger"
        type="button"
        aria-expanded={open}
        aria-controls="dashboard-module-options"
        onClick={() => setOpen((current) => !current)}
      >
        <span className="module-switcher-label">{currentLabel}</span>
        <span className="module-switcher-meta">
          <span className="module-switcher-count">{currentCount}</span>
          <ChevronDown className={open ? "open" : ""} size={16} aria-hidden="true" />
        </span>
      </button>
      <div id="dashboard-module-options" className="module-switcher-options" hidden={!open}>
        <button
          className={`module-switcher-option ${props.viewMode === "sessions" ? "active" : ""}`}
          type="button"
          aria-current={props.viewMode === "sessions" ? "page" : undefined}
          onClick={() => select("sessions")}
        >
          <span>Sessions</span>
          <span>{props.sessionsCount}</span>
        </button>
        <button
          className={`module-switcher-option ${props.viewMode === "compaction" ? "active" : ""}`}
          type="button"
          aria-current={props.viewMode === "compaction" ? "page" : undefined}
          onClick={() => select("compaction")}
        >
          <span>Compaction</span>
          <span aria-hidden="true" />
        </button>
        {props.plugins.map((plugin) => {
          const mode = `plugin:${plugin.id}` as ViewMode;
          return (
            <button
              key={plugin.id}
              className={`module-switcher-option ${props.viewMode === mode ? "active" : ""}`}
              type="button"
              aria-current={props.viewMode === mode ? "page" : undefined}
              onClick={() => select(mode)}
            >
              <span>{plugin.label}</span>
              <span>{props.pluginState[plugin.id]?.total ?? 0}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SessionNavItem(props: {
  session: SessionRow;
  active: boolean;
  nested?: boolean;
  onSelect(): void;
}): React.ReactElement {
  const title = sessionNavTitle(props.session);
  const channel = sessionChannelOf(props.session);
  return (
    <button
      className={`session-item ${props.nested ? "nested" : ""} ${props.active ? "active" : ""}`}
      type="button"
      aria-current={props.active ? "page" : undefined}
      title={`${title}\n${props.session.key}`}
      onClick={props.onSelect}
    >
      <div className="nav-item-row">
        <span className="nav-item-name">{title}</span>
        <span className="nav-item-count" title={`${props.session.message_count} 条消息`}>
          {props.session.message_count}
        </span>
      </div>
      <div className="nav-item-desc">
        <span>{sessionChannelLabel(channel)}</span>
        <span aria-hidden="true">·</span>
        <span>{relativeTime(props.session.updated_at)}</span>
      </div>
    </button>
  );
}

function isFoldedSession(session: Pick<SessionRow, "key">): boolean {
  const channel = sessionChannelOf(session);
  return channel === "scheduler" || channel === "programmatic";
}

function sessionChannelOf(session: Pick<SessionRow, "key">): string {
  return session.key.split(":", 1)[0] || "unknown";
}

function sessionNavTitle(session: SessionRow): string {
  const firstMessage = stripMarkdown(session.first_message_content).trim();
  return firstMessage || formatSessionKeyForTable(session.key);
}

function sessionChannelLabel(channel: string): string {
  const labels: Record<string, string> = {
    cli: "CLI",
    cross_mem: "Cross Memory",
    dashboard: "Dashboard",
    feishu: "飞书",
    mobile: "Mobile",
    programmatic: "Programmatic",
    qq: "QQ",
    qqbot: "QQ Bot",
    scheduler: "定时任务",
    telegram: "Telegram",
    web: "Web",
  };
  return labels[channel] || channel;
}

type PluginSlotRenderer = NonNullable<PluginConfig["renderNavBody"]>;

function PluginSlot(props: {
  plugin: PluginConfig;
  pluginId: string;
  render: PluginSlotRenderer;
  slot: "navigation" | "filters" | "topbar action";
  redrawOnTotal?: number;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  startRead(): AbortController;
  onActivate(): void;
  onError(error: unknown): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const setState = useEffectEvent((updater: (s: PluginState) => PluginState) => props.onSetState(updater));
  const activate = useEffectEvent(() => props.onActivate());
  const report = useEffectEvent((error: unknown) => props.onError(error));
  const startRead = useEffectEvent(() => props.startRead());
  const filtersKey = JSON.stringify(props.state.filters);

  // 子插件拥有节点内容；React 只拥有挂载、清理和样式生命周期。
  useEffect(() => {
    if (ref.current) {
      const host = ref.current;
      const dispatch = makeDispatch(props.plugin, getState, setState, startRead, activate, undefined, report);
      return mountPluginDom(host, props.plugin.id, props.slot, () => props.render(host, dispatch));
    }
  }, [filtersKey, props.plugin, props.pluginId, props.redrawOnTotal, props.render, props.slot, props.state.sortBy, props.state.sortOrder]);

  useLayoutEffect(() => ref.current ? props.plugin.applyStyle(ref.current) : undefined, [props.plugin]);

  return <div ref={ref} />;
}

function ContentFilters(props: {
  viewMode: ViewMode;
  messageSearch: string;
  setMessageSearch(value: string): void;
  messageRole: string;
  setMessageRole(value: string): void;
  activeSessionKey: string | null;
  clearSession(): void;
  currentPlugin: PluginConfig | null;
  currentPluginState: PluginState | null;
  onSetPluginState?: (updater: (s: PluginState) => PluginState) => void;
  startPluginRead(pluginId: string): AbortController;
  onError(error: unknown): void;
}): React.ReactElement {
  return (
    <div className="content-filters">
      {props.viewMode.startsWith("plugin:") ? (
          props.currentPlugin?.renderFilters && props.currentPluginState && props.onSetPluginState
            ? <PluginSlot
                plugin={props.currentPlugin}
                pluginId={props.currentPlugin.id}
                render={props.currentPlugin.renderFilters}
                slot="filters"
                state={props.currentPluginState}
                onSetState={props.onSetPluginState}
                startRead={() => props.startPluginRead(props.currentPlugin!.id)}
                onActivate={() => {}}
                onError={props.onError}
              />
            : null
        ) : (
          <div className="filter-row">
            <label className="search"><span aria-hidden="true">⌕</span><input aria-label="搜索消息内容" type="text" placeholder="搜索消息内容" value={props.messageSearch} onChange={(event) => props.setMessageSearch(event.target.value.trim())} /></label>
            <select aria-label="消息角色" value={props.messageRole} onChange={(event) => props.setMessageRole(event.target.value)}>
              <option value="">全部 role</option><option value="user">user</option><option value="assistant">assistant</option><option value="system">system</option><option value="tool">tool</option>
            </select>
            {props.activeSessionKey && <Chip label="session" value={props.activeSessionKey} onClear={props.clearSession} />}
          </div>
        )
      }
    </div>
  );
}

function Chip(props: { label: string; value: string; onClear(): void }): React.ReactElement {
  return <div className="active-session-chip"><span>{props.label}</span><code>{props.value}</code><button aria-label={`清除 ${props.label} 筛选`} type="button" onClick={props.onClear}>×</button></div>;
}

function TableHead(props: {
  viewMode: ViewMode;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  messageSortBy: string;
  messageSortOrder: SortOrder;
  onSort(key: string): void;
  onPluginSort?: (key: string) => void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    const sortBy = props.pluginState?.sortBy ?? "";
    const sortOrder = props.pluginState?.sortOrder ?? "desc";
    return (
      <div className="table-head" style={{ gridTemplateColumns: grid }}>
        {hasBatch && <div />}
        {props.plugin.columns.map((col) => col.sortable && props.onPluginSort
          ? <SortHead key={col.key} label={col.label} active={sortBy === col.key} order={sortOrder} onClick={() => props.onPluginSort!(col.key)} />
          : <div key={col.key}>{col.label}</div>
        )}
      </div>
    );
  }
  return <div className="table-head mode-messages">
    <div />
    <SortHead label="Session Key" active={props.messageSortBy === "session_key"} order={props.messageSortOrder} onClick={() => props.onSort("session_key")} />
    <SortHead label="Seq" active={props.messageSortBy === "seq"} order={props.messageSortOrder} onClick={() => props.onSort("seq")} />
    <div>Content</div>
    <SortHead label="Timestamp" active={props.messageSortBy === "ts"} order={props.messageSortOrder} onClick={() => props.onSort("ts")} />
    <SortHead label="Role" active={props.messageSortBy === "role"} order={props.messageSortOrder} onClick={() => props.onSort("role")} />
    <div />
  </div>;
}

function SortHead(props: { label: string; active: boolean; order: SortOrder; onClick(): void }): React.ReactElement {
  return <button className={`table-sort-btn ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}><span>{props.label}</span><span className="table-sort-arrow">{props.active ? props.order === "asc" ? "↑" : "↓" : ""}</span></button>;
}

function Rows(props: {
  viewMode: ViewMode;
  messages: MessageRow[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  selectedMessageIds: Set<string>;
  activeMessage: MessageRow | null;
  onSelectMessage(item: MessageRow): void;
  onSelectPluginRow(row: Record<string, unknown>): void;
  onTogglePluginRow(id: string): void;
  setSelectedMessageIds(value: Set<string>): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin && props.pluginState) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    return <>{props.pluginState.items.length ? props.pluginState.items.map((item) => {
      const key = String(item[props.plugin!.rowKey] ?? "");
      const isSelected = props.pluginState!.selectedIds.has(key);
      return <div key={key} className="table-row-wrap">
        {hasBatch && (
          <label className="checkbox-cell">
            <input aria-label={`选择 ${props.plugin!.label} 记录 ${key}`} type="checkbox" checked={isSelected} onChange={() => props.onTogglePluginRow(key)} />
          </label>
        )}
        <button className={`table-row ${props.pluginState!.activeRowKey === key ? "active" : ""} ${isSelected ? "selected" : ""} ${props.plugin!.rowClass?.(item) ?? ""}`} style={{ gridTemplateColumns: grid }} type="button" aria-expanded={props.pluginState!.activeRowKey === key} onClick={() => props.onSelectPluginRow(item)}>
          {hasBatch && <span aria-hidden="true" />}
          {props.plugin!.columns.map((col) => {
            const cellClass = columnCellClass(col);
            if (col.renderCell) {
              return <span key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined} dangerouslySetInnerHTML={{ __html: col.renderCell(item[col.key], item) }} />;
            }
            return <span key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined}>{formatPluginCell(props.plugin!, col, item)}</span>;
          })}
        </button>
      </div>;
    }) : <div className="empty-state">{props.plugin.emptyMessage || "暂无记录。"}</div>}</>;
  }
  return <>{props.messages.map((item) => <div key={item.id} className="table-row-wrap">
    <label className="checkbox-cell"><input aria-label={`选择消息 ${item.seq}`} type="checkbox" checked={props.selectedMessageIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMessageIds, props.setSelectedMessageIds)} /></label>
    <button className={`table-row mode-messages ${props.activeMessage?.id === item.id ? "active" : ""} ${props.selectedMessageIds.has(item.id) ? "selected" : ""}`} type="button" aria-expanded={props.activeMessage?.id === item.id} onClick={() => props.onSelectMessage(item)}>
      <span aria-hidden="true" />
      <span className="cell-session mono" title={item.session_key}>{formatSessionKeyForTable(item.session_key)}</span>
      <span className="cell-seq mono">#{item.seq}</span>
      <span className="content-preview">{stripMarkdown(item.content)}</span>
      <span className="cell-time mono">{shortTs(item.timestamp)}</span>
      <span><span className={`role-pill ${roleClass(item.role)}`}>{item.role}</span></span>
      <span aria-hidden="true" />
    </button>
  </div>)}</>;
}

function DetailPane(props: {
  viewMode: ViewMode;
  activeSession: SessionRow | null;
  activeMessage: MessageRow | null;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  loading: boolean;
  dispatch?: PluginDispatch;
  onClose: () => void;
}): React.ReactElement {
  if (props.loading) return <DetailLoading />;
  if (props.viewMode.startsWith("plugin:") && props.plugin && props.dispatch) {
    return <PluginDetail plugin={props.plugin} item={props.pluginState?.activeDetail ?? null} dispatch={props.dispatch} />;
  }
  if (props.activeMessage) {
    const message = props.activeMessage;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">消息详情</div><div className="detail-subtext">{message.session_key} · #{message.seq}</div></div><MaterialIconButton variant="standard" label="关闭详情" onClick={props.onClose}><X size={18} aria-hidden="true" /></MaterialIconButton></div>
      <div className="detail-grid">
        {detailRow("role", <span className={`role-pill ${roleClass(message.role)}`}>{message.role}</span>)}
        {detailRow("time", <code>{message.timestamp}</code>)}
        {detailRow("id", <code>{message.id}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Content</div><Markdown className="detail-content">{message.content}</Markdown></div>
      <div className="detail-block"><div className="detail-label">Extra</div><JsonTreeBlock data={message.extra} /></div>
      <div className="detail-block"><div className="detail-label">Tool Chain</div><JsonTreeBlock data={message.tool_chain} /></div>
    </div>;
  }
  if (props.activeSession) {
    const session = props.activeSession;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Session 详情</div><div className="detail-subtext">{session.key}</div></div><MaterialIconButton variant="standard" label="关闭详情" onClick={props.onClose}><X size={18} aria-hidden="true" /></MaterialIconButton></div>
      <div className="detail-grid">
        {detailRow("messages", <code>{session.message_count}</code>)}
        {detailRow("updated", <code>{session.updated_at}</code>)}
        {detailRow("last_consolidated", <code>{session.last_consolidated}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Metadata</div><JsonTreeBlock data={session.metadata} /></div>
    </div>;
  }
  return <EmptyDetail text="点开消息、session 或 memory 后，这里会显示完整内容、字段和 JSON 信息。" />;
}

function DetailLoading(): React.ReactElement {
  return <div className="detail-loading" role="status" aria-label="正在加载详情">
    {React.createElement("md-linear-progress", { className: "detail-loading-progress", indeterminate: true, "aria-label": "正在加载详情" })}
    <div className="detail-loading-line detail-loading-line-short" />
    <div className="detail-loading-line detail-loading-line-title" />
    <div className="detail-loading-block" />
    <div className="detail-loading-line" />
    <div className="detail-loading-line" />
  </div>;
}

function EmptyDetail(props: { text: string }): React.ReactElement {
  return <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">{props.text}</div></div>;
}

function detailRow(label: string, value: React.ReactNode): React.ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{label}</div><div className="detail-row-val">{value}</div></div>;
}

function JsonTreeBlock(props: { data: unknown }): React.ReactElement {
  return <JsonView value={props.data} />;
}

function toggleSet(id: string, checked: boolean, source: Set<string>, update: (value: Set<string>) => void): void {
  const next = new Set(source);
  if (checked) next.add(id);
  else next.delete(id);
  update(next);
}

function gridTemplate(columns: DashboardColumn[]): string {
  return columns
    .map((col) => {
      if (col.flex) return "minmax(0, 1fr)";
      if (col.width) return `minmax(0, ${col.width}px)`;
      return "minmax(0, auto)";
    })
    .join(" ");
}

function formatPluginCell(plugin: PluginConfig, column: DashboardColumn, item: Record<string, unknown>): string {
  const value = item[column.key];
  const formatter = plugin.formatters?.[column.fmt || ""] ?? WORKBENCH_FORMATTERS[column.fmt || "text"];
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

function tableMeta(totalMessages: number, plugin: PluginConfig | null, pluginState: PluginState | null): string {
  if (plugin && pluginState) return plugin.countTitle ? plugin.countTitle(pluginState.total) : `共 ${pluginState.total} 条`;
  return `共 ${totalMessages} 条`;
}

function triggerLabel(trigger: string): string {
  if (trigger === "context_overflow") return "overflow";
  return trigger || "unknown";
}

function CompactionView(props: {
  compaction: CompactionDetail | null;
  pending: boolean;
  activeSessionKey: string | null;
}): React.ReactElement {
  if (!props.activeSessionKey) {
    return <EmptyDetail text="从左侧选择一个 session，查看其上下文压缩状态。" />;
  }
  if (props.pending && !props.compaction) {
    return <DetailLoading />;
  }
  if (!props.compaction) {
    return <EmptyDetail text="加载失败，请重试。" />;
  }
  const { head, active, history } = props.compaction;
  return (
    <div className="compaction-view-scroll">
    <div className="detail-wrap">
      <div className="detail-toolbar">
        <div>
          <div className="detail-title">Compaction</div>
          <div className="detail-subtext">
            {formatSessionKeyForTable(props.activeSessionKey)}
            {" · "}
            {active ? `generation ${active.generation} · 下一代 ${head.next_generation}` : "尚未压缩"}
          </div>
        </div>
      </div>

      {active ? (
        <>
          <div className="detail-grid">
            {detailRow("generation", <code>{active.generation}</code>)}
            {detailRow("source", <code>{active.source_from_seq} → {active.consolidated_through_seq}</code>)}
            {detailRow("messages", <code>{active.source_message_count}</code>)}
            {detailRow("tokens", <code>{formatTokens(active.tokens_before)} → {formatTokens(active.tokens_after)}</code>)}
            {detailRow("threshold", <code>soft {formatTokens(active.threshold_tokens)} · hard {formatTokens(active.hard_input_tokens)} · tail {formatTokens(active.keep_recent_tokens)}</code>)}
            {detailRow("model", <code>{active.model}</code>)}
            {detailRow("window", <code>{formatTokens(active.context_window)}</code>)}
            {detailRow("trigger", <span className="status-pill">{triggerLabel(active.trigger)}</span>)}
            {detailRow("created", <code>{active.created_at}</code>)}
          </div>
          <div className="detail-block">
            <div className="detail-label">当前摘要</div>
            <Markdown className="detail-content">{active.summary}</Markdown>
          </div>
          <div className="detail-block">
            <div className="detail-label">Summary Usage</div>
            <JsonTreeBlock data={active.summary_usage} />
          </div>
          <div className="detail-block">
            <div className="detail-label">Source Plan Digest</div>
            <div className="detail-content mono compaction-digest">{active.source_plan_digest}</div>
          </div>
        </>
      ) : (
        <EmptyDetail text="该 session 尚未发生压缩——模型上下文达到 74% 水位后自动生成摘要。" />
      )}

      {history.length > 0 && (
        <div className="detail-block">
          <div className="detail-label">历史 generations</div>
          {history.map((item) => (
            <div key={item.generation} className="compaction-history-row">
              <code>gen {item.generation}</code>
              <span className="muted-text">{shortTs(item.created_at)}</span>
              <code>{formatTokens(item.tokens_before)} → {formatTokens(item.tokens_after)}</code>
              <span className="status-pill">{triggerLabel(item.trigger)}</span>
              {item.invalidated_at ? (
                <span className="type-pill compaction-invalidated" title={item.invalidated_reason ?? undefined}>已失效</span>
              ) : (
                <span className="type-pill compaction-valid">有效</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
    </div>
  );
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
