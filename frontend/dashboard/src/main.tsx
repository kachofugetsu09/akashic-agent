import React, { useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Dialog from "@radix-ui/react-dialog";
import "@material/web/progress/linear-progress.js";
import { BookOpenText, Bot, ChevronDown, ChevronLeft, ChevronRight, Gauge, SlidersHorizontal, X } from "lucide-react";
import "./styles.css";
import { api, asPageResult, interactionDeleteRequirement, pageCount } from "./api";
import {
  encodePath,
  formatSessionKeyForTable,
  formatTokens,
  proactiveFlowLabel,
  proactiveResultLabel,
  proactiveSectionLabel,
  proactiveTickPreview,
  relativeTime,
  roleClass,
  shortTs,
  stripMarkdown,
} from "./format";
import { installDashboardGlobals, loadPluginAssets } from "./pluginRuntime";
import { exposeRuntime } from "./design/runtime";
import { Btn, JsonView, Markdown, ThemeToggle } from "./design/ui";
import { PluginDetail, PluginMain } from "./PluginDetail";
import { initializeTheme, startCrossPortThemeSync, useTheme } from "../../theme/src/theme-runtime";
import { MaterialIconButton } from "../../theme/src/material-react";
import type {
  CompactionDetail,
  DashboardColumn,
  MessageRow,
  PageResult,
  PluginBatchAction,
  PluginConfig,
  PluginDispatch,
  PluginState,
  ProactiveOverview,
  ProactiveStep,
  ProactiveTick,
  SessionRow,
  SortOrder,
  ViewMode,
} from "./types";

const pluginPreset = document.createElement("link");
pluginPreset.rel = "stylesheet";
pluginPreset.href = "/dashboard/assets/sdk/preset.css";
document.head.appendChild(pluginPreset);
initializeTheme();
startCrossPortThemeSync();

const notificationIcon = "data:image/png;base64,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";

// Creates a PluginDispatch bound to the given plugin + latest state getter.
function makeDispatch(
  plugin: PluginConfig,
  getState: () => PluginState | null,
  onSetState: (updater: (s: PluginState) => PluginState) => void,
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
    const result = checkedPluginPage(plugin, await plugin.fetchPage({ page: 1, pageSize: state.pageSize, filters: nextFilters, sortBy: nextSortBy, sortOrder: nextSortOrder }));
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
  return error instanceof DOMException && error.name === "AbortError";
}

function useLatestReader<T>(value: T): () => T {
  const ref = useRef(value);
  useLayoutEffect(() => {
    ref.current = value;
  }, [value]);
  return useCallback(() => ref.current, []);
}

type ShellView = "chat" | "dashboard" | "runtime" | "models";
type ShellStatus = "needs_setup" | "starting" | "ready";

interface ShellState {
  status: ShellStatus;
  chatReady: boolean;
}

function initialShellView(): ShellView {
  const value = window.location.hash.slice(1);
  return value === "dashboard" || value === "runtime" || value === "models" ? value : "chat";
}

function App(): React.ReactElement {
  const theme = useTheme();
  const [shellView, setShellView] = useState<ShellView>(initialShellView);
  const [shellStatus, setShellStatus] = useState<ShellStatus>("starting");
  const serviceOrigin = window.location.origin;
  const chatFrameRef = useRef<HTMLIFrameElement>(null);
  const runtimeFrameRef = useRef<HTMLIFrameElement>(null);
  const settingsFrameRef = useRef<HTMLIFrameElement>(null);

  const openView = useCallback((next: ShellView): void => {
    setShellView(next);
    const base = `${window.location.pathname}${window.location.search}`;
    window.history.replaceState(null, "", next === "chat" ? base : `${base}#${next}`);
  }, []);

  const syncFrameTheme = useCallback((frame: HTMLIFrameElement | null): void => {
    frame?.contentWindow?.postMessage(
      { type: "akashic.theme", themeId: theme.id },
      serviceOrigin,
    );
  }, [serviceOrigin, theme.id]);

  useEffect(() => {
    syncFrameTheme(chatFrameRef.current);
    syncFrameTheme(runtimeFrameRef.current);
    syncFrameTheme(settingsFrameRef.current);
  }, [syncFrameTheme]);

  useEffect(() => {
    const handleSettingsApplied = (event: MessageEvent<unknown>): void => {
      const payload = event.data;
      if (
        event.origin !== serviceOrigin
        || event.source !== settingsFrameRef.current?.contentWindow
        || typeof payload !== "object"
        || payload === null
        || !("type" in payload)
        || payload.type !== "akashic.settings.applied"
      ) return;
      chatFrameRef.current?.contentWindow?.postMessage(
        { type: "akashic.models.changed" },
        serviceOrigin,
      );
      setShellStatus("starting");
      openView("chat");
    };
    window.addEventListener("message", handleSettingsApplied);
    return () => window.removeEventListener("message", handleSettingsApplied);
  }, [openView, serviceOrigin]);

  useEffect(() => {
    let active = true;
    const refresh = async (): Promise<void> => {
      try {
        const state = await api<ShellState>("/api/shell/state");
        if (active) setShellStatus(state.chatReady ? "ready" : state.status);
      } catch (error) {
        console.error("[dashboard] shell readiness failed", error);
        if (active) setShellStatus("starting");
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 1_500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  // 必要 effect：setup 未完成时引导到模型页（跨状态导航 + history 副作用），不可改为渲染期计算
  useEffect(() => {
    if (shellStatus === "needs_setup" && shellView !== "models") openView("models");
  }, [openView, shellStatus, shellView]);

  return (
    <div className="unified-shell">
      <aside className="primary-rail" aria-label="Akashic 主导航">
        <div className="primary-rail-brand" title="Akashic">
          <img src={notificationIcon} alt="" />
        </div>
        <nav className="primary-rail-nav" aria-label="主要功能">
          <PrimaryRailButton label="聊天" active={shellView === "chat"} onClick={() => openView(shellStatus === "needs_setup" ? "models" : "chat")}>
            <Bot aria-hidden="true" />
          </PrimaryRailButton>
          <PrimaryRailButton label="工作台" active={shellView === "dashboard"} onClick={() => openView("dashboard")}>
            <Gauge aria-hidden="true" />
          </PrimaryRailButton>
          <PrimaryRailButton label="知识与运行" active={shellView === "runtime"} onClick={() => openView("runtime")}>
            <BookOpenText aria-hidden="true" />
          </PrimaryRailButton>
          <PrimaryRailButton label="模型" active={shellView === "models"} onClick={() => openView("models")}>
            <SlidersHorizontal aria-hidden="true" />
          </PrimaryRailButton>
        </nav>
        <div className="primary-rail-footer"><ThemeToggle /></div>
      </aside>

      <div className="shell-view-stack">
        <section className={`shell-view dashboard-shell-view ${shellView === "dashboard" ? "is-active" : ""}`} aria-hidden={shellView !== "dashboard"}>
          {shellStatus === "ready" ? <DashboardWorkspace /> : <RuntimeUnavailable status={shellStatus} />}
        </section>
        <section className={`shell-view ${shellView === "chat" ? "is-active" : ""}`} aria-hidden={shellView !== "chat"}>
          <iframe ref={chatFrameRef} title="Akashic 聊天" src="/chat?embedded=1" onLoad={() => syncFrameTheme(chatFrameRef.current)} />
        </section>
        <section className={`shell-view ${shellView === "runtime" ? "is-active" : ""}`} aria-hidden={shellView !== "runtime"}>
          {shellStatus === "ready"
            ? <iframe ref={runtimeFrameRef} title="知识与运行" src="/chat?embedded=1&surface=runtime" onLoad={() => syncFrameTheme(runtimeFrameRef.current)} />
            : <RuntimeUnavailable status={shellStatus} />}
        </section>
        <section className={`shell-view ${shellView === "models" ? "is-active" : ""}`} aria-hidden={shellView !== "models"}>
          <iframe ref={settingsFrameRef} title="模型配置" src="/settings?embedded=1" onLoad={() => syncFrameTheme(settingsFrameRef.current)} />
        </section>
      </div>
    </div>
  );
}

function RuntimeUnavailable({ status }: { status: Exclude<ShellStatus, "ready"> }): React.ReactElement {
  return (
    <div className="runtime-unavailable" role="status">
      <span>{status === "needs_setup" ? "首次使用" : "运行时启动中"}</span>
      <strong>{status === "needs_setup" ? "连接模型后显示这里" : "正在恢复工作区"}</strong>
      <p>{status === "needs_setup" ? "前往“模型”完成登录或添加 API Key。" : "聊天入口保持可用，准备完成后会自动恢复。"}</p>
    </div>
  );
}

function PrimaryRailButton(props: {
  label: string;
  active: boolean;
  onClick(): void;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <button type="button" className={`primary-rail-button ${props.active ? "is-active" : ""}`} aria-label={props.label} title={props.label} aria-current={props.active ? "page" : undefined} onClick={props.onClick}>
      {props.children}
      <span>{props.label}</span>
    </button>
  );
}

function DashboardWorkspace(): React.ReactElement {
  const [viewMode, setViewMode] = useState<ViewMode>("sessions");
  const [plugins, setPlugins] = useState<PluginConfig[]>([]);
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
  const [proactiveOverview, setProactiveOverview] = useState<ProactiveOverview | null>(null);
  const [proactiveSection, setProactiveSection] = useState("all");
  const [proactiveItems, setProactiveItems] = useState<ProactiveTick[]>([]);
  const [proactivePage, setProactivePage] = useState(1);
  const [proactiveSortBy, setProactiveSortBy] = useState("started_at");
  const [proactiveSortOrder, setProactiveSortOrder] = useState<SortOrder>("desc");
  const [proactiveTotal, setProactiveTotal] = useState(0);
  const [proactiveSessionFilter, setProactiveSessionFilter] = useState("");
  const [activeProactiveKey, setActiveProactiveKey] = useState<string | null>(null);
  const [activeProactiveDetail, setActiveProactiveDetail] = useState<ProactiveTick | null>(null);
  const [activeProactiveSteps, setActiveProactiveSteps] = useState<ProactiveStep[]>([]);
  const [proactiveDetailPending, setProactiveDetailPending] = useState(false);
  const [hiddenPlugins, setHiddenPlugins] = useState<Record<string, boolean>>({});
  const [pendingPluginDetailKey, setPendingPluginDetailKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const proactiveRequestRef = useRef<AbortController | null>(null);
  const proactiveDetailRequestRef = useRef(0);
  const pluginDetailRequestRef = useRef(0);

  const messagePageSize = 25;
  const proactivePageSize = 25;
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
    setPendingPluginDetailKey(null);
    setPluginState((current) => {
      const state = current[pluginId];
      if (!state) return current;
      return { ...current, [pluginId]: { ...state, activeRowKey: null, activeDetail: null } };
    });
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

  const loadProactiveOverview = useCallback(async () => {
    setProactiveOverview(await api<ProactiveOverview>("/api/dashboard/proactive/overview"));
  }, []);

  const loadCompaction = useCallback(async () => {
    if (!activeSessionKey) {
      setCompaction(null);
      return;
    }
    setCompactionPending(true);
    try {
      const payload = await api<CompactionDetail>(
        `/api/dashboard/sessions/${encodePath(activeSessionKey)}/compaction`,
      );
      setCompaction(payload);
    } finally {
      setCompactionPending(false);
    }
  }, [activeSessionKey]);

  const loadProactivePanel = useCallback(async () => {
    proactiveRequestRef.current?.abort();
    const controller = new AbortController();
    proactiveRequestRef.current = controller;
    const params = new URLSearchParams();
    params.set("page", String(proactivePage));
    params.set("page_size", String(proactivePageSize));
    params.set("sort_by", proactiveSortBy);
    params.set("sort_order", proactiveSortOrder);
    if (proactiveSessionFilter) params.set("session_key", proactiveSessionFilter);
    if (proactiveSection === "reply" || proactiveSection === "skip") params.set("terminal_action", proactiveSection);
    if (proactiveSection === "drift" || proactiveSection === "proactive") params.set("flow", proactiveSection);
    if (["busy", "cooldown", "presence"].includes(proactiveSection)) params.set("gate_exit", proactiveSection);
    try {
      const payload = asPageResult<ProactiveTick>(await api(`/api/dashboard/proactive/tick_logs?${params.toString()}`, { signal: controller.signal }));
      setProactiveItems(payload.items);
      setProactiveTotal(payload.total);
      setActiveProactiveKey((current) => current && payload.items.some((item) => item.tick_id === current) ? current : null);
    } finally {
      if (proactiveRequestRef.current === controller) proactiveRequestRef.current = null;
    }
  }, [proactivePage, proactiveSection, proactiveSessionFilter, proactiveSortBy, proactiveSortOrder]);

  const loadPluginPanel = useCallback(async (pluginId: string) => {
    const plugin = plugins.find((item) => item.id === pluginId);
    const state = readPluginState()[pluginId];
    if (!plugin || !state) return;
    const result = checkedPluginPage(plugin, await plugin.fetchPage({ page: state.page, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder }));
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
  }, [plugins, readPluginState]);

  const refreshCurrentView = useCallback(async () => {
    await loadSessions();
    if (viewMode === "proactive") {
      await loadProactiveOverview();
      await loadProactivePanel();
    } else if (viewMode === "compaction") {
      await loadCompaction();
    } else if (viewMode.startsWith("plugin:")) {
      await loadPluginPanel(viewMode.slice(7));
    } else {
      await loadMessages();
    }
  }, [loadCompaction, loadMessages, loadPluginPanel, loadProactiveOverview, loadProactivePanel, loadSessions, viewMode]);

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
    proactiveRequestRef.current?.abort();
  }, []);

  useEffect(() => {
    installDashboardGlobals((plugin) => {
      setPlugins((current) => current.some((item) => item.id === plugin.id) ? current : [...current, plugin]);
      setPluginState((current) => current[plugin.id] ? current : {
        ...current,
        [plugin.id]: {
          page: 1,
          pageSize: plugin.pageSize || 25,
          total: 0,
          items: [],
          activeRowKey: null,
          activeDetail: null,
          filters: {},
          sortBy: plugin.defaultSortBy ?? "",
          sortOrder: plugin.defaultSortOrder ?? "desc",
          selectedIds: new Set(),
        },
      });
    });
    exposeRuntime();
    void run(loadPluginAssets);
  }, [run]);

  useEffect(() => {
    void run(loadSessions);
  }, [loadSessions, run]);

  useEffect(() => {
    for (const plugin of plugins) {
      void run(async () => {
        const count = await plugin.getCount();
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
  }, [plugins, run]);

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

  const sort = (scope: "messages" | "proactive", key: string): void => {
    const flip = (currentKey: string, currentOrder: SortOrder): SortOrder => currentKey === key && currentOrder === "desc" ? "asc" : "desc";
    if (scope === "messages") {
      setMessageSortOrder(flip(messageSortBy, messageSortOrder));
      setMessageSortBy(key);
      setMessagePage(1);
    } else {
      setProactiveSortOrder(flip(proactiveSortBy, proactiveSortOrder));
      setProactiveSortBy(key);
      setProactivePage(1);
    }
  };

  // 必要 effect：按当前视图加载对应面板数据（fetch 有副作用，React 官方认可 effect 做数据获取）
  useEffect(() => {
    if (viewMode === "sessions") void run(loadMessages);
  }, [loadMessages, run, viewMode]);

  useEffect(() => {
    if (viewMode === "proactive") void run(loadProactivePanel);
  }, [loadProactivePanel, run, viewMode]);

  useEffect(() => {
    if (viewMode === "compaction") void run(loadCompaction);
  }, [loadCompaction, run, viewMode]);

  useEffect(() => {
    if (viewMode.startsWith("plugin:")) void run(() => loadPluginPanel(viewMode.slice(7)));
  }, [loadPluginPanel, run, viewMode]);

  const currentPageCount = currentPluginState
    ? pageCount(currentPluginState.total, currentPluginState.pageSize)
    : viewMode === "proactive"
      ? pageCount(proactiveTotal, proactivePageSize)
      : pageCount(totalMessages, messagePageSize);

  const currentPage = currentPluginState?.page ?? (viewMode === "proactive" ? proactivePage : messagePage);

  const changePage = (delta: number): void => {
    if (currentPage + delta < 1 || currentPage + delta > currentPageCount) return;
    if (currentPluginId) {
      void run(async () => {
        const plugin = plugins.find((item) => item.id === currentPluginId);
        const state = pluginState[currentPluginId];
        if (!plugin || !state) return;
        const nextPage = state.page + delta;
        const result = checkedPluginPage(plugin, await plugin.fetchPage({ page: nextPage, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder }));
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
    } else if (viewMode === "proactive") setProactivePage((page) => page + delta);
    else setMessagePage((page) => page + delta);
  };

  // Batch count: messages or plugin selectedIds
  const pluginBatchCount = currentPluginState?.selectedIds.size ?? 0;
  const batchCount = viewMode.startsWith("plugin:") ? pluginBatchCount : selectedMessageIds.size;

  // dispatch for current plugin (used in DetailPane and batch bar)
  // 插件 dispatch 是宿主持有的稳定能力；事件读取最新状态，legacy DOM 不重复初始化。
  const currentDispatch = useMemo(() => currentPlugin && hasCurrentPluginState
    ? makeDispatch(
        currentPlugin,
        () => readPluginState()[currentPlugin.id] ?? null,
        (updater) => setPluginStateFor(currentPlugin.id, updater),
        () => activatePlugin(currentPlugin.id),
        () => closePlugin(currentPlugin.id),
        reportError,
      )
    : undefined, [activatePlugin, closePlugin, currentPlugin, hasCurrentPluginState, readPluginState, reportError, setPluginStateFor]);
  const isPluginWorkbench = Boolean(
    currentPlugin
      && currentPluginState
      && currentDispatch
      && currentPluginLayout === "workbench"
      && (currentPlugin.renderMain || currentPlugin.Main),
  );
  const detailOpen = viewMode.startsWith("plugin:")
    ? Boolean(currentPluginState?.activeRowKey)
    : viewMode === "proactive"
      ? Boolean(activeProactiveKey)
      : viewMode === "compaction"
        ? false
        : Boolean(activeMessage || activeSession);

  return (
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
                    <span>⌕</span>
                    <input type="text" placeholder="搜索会话" value={sessionSearch} onChange={(event) => setSessionSearch(event.target.value.trim())} />
                  </label>
                  <select value={sessionChannel} onChange={(event) => {
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

            {viewMode === "proactive" && (
              <div className="proactive-quick-list">
                <button className={`all-messages-row ${proactiveSection === "all" ? "active" : ""}`} type="button" onClick={() => { setProactiveSection("all"); setProactivePage(1); selectView("proactive"); }}>
                  <span>{proactiveSectionLabel("all")}</span><strong>{proactiveSectionCount("all", proactiveOverview)}</strong>
                </button>
                {["drift", "proactive", "reply", "skip", "busy", "cooldown", "presence"].map((section) => (
                  <button key={section} className={`proactive-quick-item ${proactiveSection === section ? "active" : ""}`} type="button" onClick={() => { setProactiveSection(section); setProactivePage(1); selectView("proactive"); }}>
                    <div className="nav-item-row">
                      <span className="nav-item-name">{proactiveSectionLabel(section)}</span>
                      <span className="nav-item-count">{proactiveSectionCount(section, proactiveOverview)}</span>
                    </div>
                  </button>
                ))}
              </div>
            )}

            {viewMode.startsWith("plugin:") && currentPlugin && currentPluginState && currentPlugin.renderNavBody && (
              <PluginNavBody
                plugin={currentPlugin}
                pluginId={currentPlugin.id}
                state={currentPluginState}
                onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
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
            proactiveSection={proactiveSection}
            proactiveSessionFilter={proactiveSessionFilter}
            clearProactiveSession={() => { setProactiveSessionFilter(""); setProactivePage(1); }}
            currentPlugin={currentPlugin}
            currentPluginState={currentPluginState}
            onSetPluginState={currentPlugin ? (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })) : undefined}
            onError={reportError}
          />
          {viewMode.startsWith("plugin:") && currentPlugin?.renderTopbarAction && currentPluginState && currentDispatch && (
            <div className="content-toolbar-actions">
              <PluginTopbarAction
                plugin={currentPlugin}
                pluginId={currentPlugin.id}
                state={currentPluginState}
                onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
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
              <TableHead viewMode={viewMode} plugin={currentPlugin} pluginState={currentPluginState} messageSortBy={messageSortBy} messageSortOrder={messageSortOrder} proactiveSortBy={proactiveSortBy} proactiveSortOrder={proactiveSortOrder} onSort={sort} onPluginSort={currentDispatch ? (key) => currentDispatch.setSort(key) : undefined} />
              <div className="table-body">
                <Rows
                  viewMode={viewMode}
                  messages={messages}
                  proactiveItems={proactiveItems}
                  plugin={currentPlugin}
                  pluginState={currentPluginState}
                  selectedMessageIds={selectedMessageIds}
                  activeMessage={activeMessage}
                  activeProactiveKey={activeProactiveKey}
                  onSelectMessage={(msg) => setActiveMessage((current) => current?.id === msg.id ? null : msg)}
                  onSelectProactive={(item) => void run(async () => {
                    const closing = activeProactiveKey === item.tick_id;
                    const requestId = ++proactiveDetailRequestRef.current;
                    setActiveProactiveKey(closing ? null : item.tick_id);
                    setActiveProactiveDetail(null);
                    setActiveProactiveSteps([]);
                    setProactiveDetailPending(!closing);
                    if (closing) return;
                    try {
                      const [detail, stepsPayload] = await Promise.all([
                        api<ProactiveTick>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}`),
                        api<PageResult<ProactiveStep>>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}/steps`),
                      ]);
                      if (proactiveDetailRequestRef.current !== requestId) return;
                      const steps = asPageResult<ProactiveStep>(stepsPayload);
                      setActiveProactiveDetail(detail);
                      setActiveProactiveSteps(steps.items);
                    } finally {
                      if (proactiveDetailRequestRef.current === requestId) setProactiveDetailPending(false);
                    }
                  })}
                  onSelectPluginRow={(row) => {
                    if (!currentPlugin) return;
                    const key = String(row[currentPlugin.rowKey] ?? "");
                    const requestId = ++pluginDetailRequestRef.current;
                    const closing = currentPluginState?.activeRowKey === key;
                    setPendingPluginDetailKey(closing ? null : `${currentPlugin.id}:${key}`);
                    setPluginState((c) => {
                      const ps = c[currentPlugin.id];
                      if (!ps) return c;
                      return { ...c, [currentPlugin.id]: { ...ps, activeRowKey: closing ? null : key, activeDetail: null } };
                    });
                    if (closing) return;
                    void (async () => {
                      try {
                        const detail = currentPlugin.fetchDetail ? await currentPlugin.fetchDetail(row) : row;
                        if (pluginDetailRequestRef.current !== requestId) return;
                        setPluginState((current) => {
                          const state = current[currentPlugin.id];
                          if (!state || state.activeRowKey !== key) return current;
                          return { ...current, [currentPlugin.id]: { ...state, activeDetail: detail } };
                        });
                      } catch (exc) {
                        if (pluginDetailRequestRef.current === requestId) reportError(exc);
                      } finally {
                        if (pluginDetailRequestRef.current === requestId) setPendingPluginDetailKey(null);
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
                 <div>{tableMeta(viewMode, totalMessages, proactiveTotal, currentPlugin, currentPluginState, proactiveSessionFilter)}</div>
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
                activeProactiveDetail={activeProactiveDetail}
                activeProactiveSteps={activeProactiveSteps}
                plugin={currentPlugin}
                pluginState={currentPluginState}
                loading={viewMode === "proactive"
                  ? proactiveDetailPending
                  : Boolean(currentPlugin && currentPluginState?.activeRowKey && pendingPluginDetailKey === `${currentPlugin.id}:${currentPluginState.activeRowKey}`)}
                dispatch={currentDispatch}
                setProactiveSessionFilter={(key) => { setProactiveSessionFilter(key); setProactivePage(1); selectView("proactive"); }}
                onClose={() => {
                  setActiveSession(null);
                  setActiveMessage(null);
                  proactiveDetailRequestRef.current += 1;
                  setActiveProactiveKey(null);
                  setActiveProactiveDetail(null);
                  setActiveProactiveSteps([]);
                  setProactiveDetailPending(false);
                  if (currentPlugin) {
                    pluginDetailRequestRef.current += 1;
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
        <Dialog.Portal>
          <Dialog.Overlay className="modal-backdrop" />
          <Dialog.Content className="modal" aria-describedby="dashboard-error-description">
            <Dialog.Title className="modal-title">请求失败</Dialog.Title>
            <Dialog.Description id="dashboard-error-description" className="modal-sub">{error}</Dialog.Description>
            <div className="modal-actions">
              <Btn onClick={() => setError(null)}>关闭</Btn>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
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

function PluginNavBody(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
  onError(error: unknown): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const setState = useEffectEvent((updater: (s: PluginState) => PluginState) => props.onSetState(updater));
  const activate = useEffectEvent(() => props.onActivate());
  const report = useEffectEvent((error: unknown) => props.onError(error));
  const filtersKey = JSON.stringify(props.state.filters);

  // 必要 effect：legacy 插件 DOM render 契约（renderNavBody 直接操作 ref 节点），不可改为渲染期计算
  useEffect(() => {
    if (ref.current && props.plugin.renderNavBody) {
      const dispatch = makeDispatch(props.plugin, getState, setState, activate, undefined, report);
      props.plugin.renderNavBody(ref.current, dispatch);
    }
  }, [filtersKey, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder, props.state.total]);

  return <div ref={ref} />;
}

function PluginFilters(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
  onError(error: unknown): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const setState = useEffectEvent((updater: (s: PluginState) => PluginState) => props.onSetState(updater));
  const activate = useEffectEvent(() => props.onActivate());
  const report = useEffectEvent((error: unknown) => props.onError(error));
  const filtersKey = JSON.stringify(props.state.filters);

  // 必要 effect：legacy 插件 DOM render 契约（renderFilters 直接操作 ref 节点）
  useEffect(() => {
    if (ref.current && props.plugin.renderFilters) {
      const dispatch = makeDispatch(props.plugin, getState, setState, activate, undefined, report);
      props.plugin.renderFilters(ref.current, dispatch);
    }
  }, [filtersKey, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

  return <div ref={ref} />;
}

function PluginTopbarAction(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
  onError(error: unknown): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const setState = useEffectEvent((updater: (s: PluginState) => PluginState) => props.onSetState(updater));
  const activate = useEffectEvent(() => props.onActivate());
  const report = useEffectEvent((error: unknown) => props.onError(error));
  const filtersKey = JSON.stringify(props.state.filters);

  // 必要 effect：legacy 插件 DOM render 契约（renderTopbarAction 直接操作 ref 节点）
  useEffect(() => {
    if (ref.current && props.plugin.renderTopbarAction) {
      const dispatch = makeDispatch(props.plugin, getState, setState, activate, undefined, report);
      props.plugin.renderTopbarAction(ref.current, dispatch);
    }
  }, [filtersKey, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

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
  proactiveSection: string;
  proactiveSessionFilter: string;
  clearProactiveSession(): void;
  currentPlugin: PluginConfig | null;
  currentPluginState: PluginState | null;
  onSetPluginState?: (updater: (s: PluginState) => PluginState) => void;
  onError(error: unknown): void;
}): React.ReactElement {
  return (
    <div className="content-filters">
      {props.viewMode.startsWith("plugin:") ? (
          props.currentPlugin?.renderFilters && props.currentPluginState && props.onSetPluginState
            ? <PluginFilters
                plugin={props.currentPlugin}
                pluginId={props.currentPlugin.id}
                state={props.currentPluginState}
                onSetState={props.onSetPluginState}
                onActivate={() => {}}
                onError={props.onError}
              />
            : null
        ) : props.viewMode === "proactive" ? (
          <div className="filter-row">
            <div className="active-session-chip"><span>result</span><code>{proactiveSectionLabel(props.proactiveSection)}</code></div>
            {props.proactiveSessionFilter && <Chip label="session" value={props.proactiveSessionFilter} onClear={props.clearProactiveSession} />}
          </div>
        ) : (
          <div className="filter-row">
            <label className="search"><span>⌕</span><input type="text" placeholder="搜索消息内容" value={props.messageSearch} onChange={(event) => props.setMessageSearch(event.target.value.trim())} /></label>
            <select value={props.messageRole} onChange={(event) => props.setMessageRole(event.target.value)}>
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
  return <div className="active-session-chip"><span>{props.label}</span><code>{props.value}</code><button type="button" onClick={props.onClear}>×</button></div>;
}

function TableHead(props: {
  viewMode: ViewMode;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  messageSortBy: string;
  messageSortOrder: SortOrder;
  proactiveSortBy: string;
  proactiveSortOrder: SortOrder;
  onSort(scope: "messages" | "proactive", key: string): void;
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
  if (props.viewMode === "proactive") {
    return <div className="table-head mode-proactive-ticks">
      <SortHead label="Session" active={props.proactiveSortBy === "session_key"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "session_key")} />
      <SortHead label="Started" active={props.proactiveSortBy === "started_at"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "started_at")} />
      <SortHead label="Result" active={props.proactiveSortBy === "terminal_action"} order={props.proactiveSortOrder} onClick={() => props.onSort("proactive", "terminal_action")} />
      <div>Summary</div><div />
    </div>;
  }
  return <div className="table-head mode-messages">
    <div />
    <SortHead label="Session Key" active={props.messageSortBy === "session_key"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "session_key")} />
    <SortHead label="Seq" active={props.messageSortBy === "seq"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "seq")} />
    <div>Content</div>
    <SortHead label="Timestamp" active={props.messageSortBy === "ts"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "ts")} />
    <SortHead label="Role" active={props.messageSortBy === "role"} order={props.messageSortOrder} onClick={() => props.onSort("messages", "role")} />
    <div />
  </div>;
}

function SortHead(props: { label: string; active: boolean; order: SortOrder; onClick(): void }): React.ReactElement {
  return <button className={`table-sort-btn ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}><span>{props.label}</span><span className="table-sort-arrow">{props.active ? props.order === "asc" ? "↑" : "↓" : ""}</span></button>;
}

function Rows(props: {
  viewMode: ViewMode;
  messages: MessageRow[];
  proactiveItems: ProactiveTick[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  selectedMessageIds: Set<string>;
  activeMessage: MessageRow | null;
  activeProactiveKey: string | null;
  onSelectMessage(item: MessageRow): void;
  onSelectProactive(item: ProactiveTick): void;
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
      return <div key={key} className={`table-row ${props.pluginState!.activeRowKey === key ? "active" : ""} ${isSelected ? "selected" : ""} ${props.plugin!.rowClass?.(item) ?? ""}`} style={{ gridTemplateColumns: grid }} role="button" tabIndex={0} aria-expanded={props.pluginState!.activeRowKey === key} onClick={() => props.onSelectPluginRow(item)} onKeyDown={(event) => activateRowFromKeyboard(event, () => props.onSelectPluginRow(item))}>
        {hasBatch && (
          <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}>
            <input type="checkbox" checked={isSelected} onChange={() => props.onTogglePluginRow(key)} />
          </label>
        )}
        {props.plugin!.columns.map((col) => {
          const cellClass = columnCellClass(col);
          if (col.renderCell) {
            return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined} dangerouslySetInnerHTML={{ __html: col.renderCell(item[col.key], item) }} />;
          }
          return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined}>{formatPluginCell(props.plugin!, col, item)}</div>;
        })}
      </div>;
    }) : <div className="empty-state">{props.plugin.emptyMessage || "暂无记录。"}</div>}</>;
  }
  if (props.viewMode === "proactive") {
    return <>{props.proactiveItems.map((item) => <div key={item.tick_id} className={`table-row mode-proactive-ticks ${props.activeProactiveKey === item.tick_id ? "active" : ""}`} role="button" tabIndex={0} aria-expanded={props.activeProactiveKey === item.tick_id} onClick={() => props.onSelectProactive(item)} onKeyDown={(event) => activateRowFromKeyboard(event, () => props.onSelectProactive(item))}>
      <div className="cell-session mono">{formatSessionKeyForTable(item.session_key)}</div>
      <div className="cell-time">{shortTs(item.started_at)}</div>
      <div className="proactive-status-cell"><span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span><span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span></div>
      <div className="content-preview">{proactiveTickPreview(item)}</div>
      <div />
    </div>)}</>;
  }
  return <>{props.messages.map((item) => <div key={item.id} className={`table-row mode-messages ${props.activeMessage?.id === item.id ? "active" : ""} ${props.selectedMessageIds.has(item.id) ? "selected" : ""}`} role="button" tabIndex={0} aria-expanded={props.activeMessage?.id === item.id} onClick={() => props.onSelectMessage(item)} onKeyDown={(event) => activateRowFromKeyboard(event, () => props.onSelectMessage(item))}>
    <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={props.selectedMessageIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMessageIds, props.setSelectedMessageIds)} /></label>
    <div className="cell-session mono" title={item.session_key}>{formatSessionKeyForTable(item.session_key)}</div>
    <div className="cell-seq mono">#{item.seq}</div>
    <div className="content-preview">{stripMarkdown(item.content)}</div>
    <div className="cell-time mono">{shortTs(item.timestamp)}</div>
    <div><span className={`role-pill ${roleClass(item.role)}`}>{item.role}</span></div>
    <div />
  </div>)}</>;
}

function DetailPane(props: {
  viewMode: ViewMode;
  activeSession: SessionRow | null;
  activeMessage: MessageRow | null;
  activeProactiveDetail: ProactiveTick | null;
  activeProactiveSteps: ProactiveStep[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  loading: boolean;
  dispatch?: PluginDispatch;
  setProactiveSessionFilter(key: string): void;
  onClose: () => void;
}): React.ReactElement {
  if (props.loading) return <DetailLoading />;
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    return <PluginDetail plugin={props.plugin} item={props.pluginState?.activeDetail ?? null} dispatch={props.dispatch} />;
  }
  if (props.viewMode === "proactive") {
    const item = props.activeProactiveDetail;
    if (!item) return <EmptyDetail text="点开 tick 后，这里会显示 proactive 执行详情和工具链。" />;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Tick 详情</div><div className="detail-subtext">{item.tick_id}</div></div><MaterialIconButton variant="standard" label="关闭详情" onClick={props.onClose}><X size={18} aria-hidden="true" /></MaterialIconButton></div>
      <Btn size="sm" variant="ghost" onClick={() => props.setProactiveSessionFilter(item.session_key)}>只看这个 session</Btn>
      <div className="detail-grid">
        {detailRow("session", <code>{item.session_key}</code>)}
        {detailRow("started", <code>{item.started_at}</code>)}
        {detailRow("result", <span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span>)}
        {detailRow("flow", <span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span>)}
      </div>
      {item.final_message && <div className="detail-block"><div className="detail-label">Final Message</div><Markdown className="detail-content">{item.final_message}</Markdown></div>}
      <div className="detail-block"><div className="detail-label">Steps</div>{props.activeProactiveSteps.length ? props.activeProactiveSteps.map((step) => <div key={`${step.phase}-${step.step_index}`} className="tool-step"><div className="tool-step-head"><div className="tool-step-title"><span className="status-pill">step {step.step_index}</span><span className="type-pill">{step.tool_name}</span></div></div><JsonTreeBlock data={step.tool_args} /><div className="detail-content tool-result">{step.tool_result_text}</div></div>) : <div className="muted-text">没有记录到工具调用。</div>}</div>
    </div>;
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

function activateRowFromKeyboard(event: React.KeyboardEvent<HTMLDivElement>, activate: () => void): void {
  if (event.target !== event.currentTarget || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  activate();
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
  const formatter = plugin.formatters?.[column.fmt || ""] ?? (window as Window & { AkashicDashboard?: { _formatters: Record<string, (value: unknown, item?: Record<string, unknown>) => string> } }).AkashicDashboard?._formatters[column.fmt || "text"];
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

function tableMeta(viewMode: ViewMode, totalMessages: number, proactiveTotal: number, plugin: PluginConfig | null, pluginState: PluginState | null, proactiveSessionFilter: string): string {
  if (plugin && pluginState) return plugin.countTitle ? plugin.countTitle(pluginState.total) : `共 ${pluginState.total} 条`;
  if (viewMode === "proactive") return proactiveSessionFilter ? `共 ${proactiveTotal} 条 tick · session: ${proactiveSessionFilter}` : `共 ${proactiveTotal} 条 tick`;
  return `共 ${totalMessages} 条`;
}


function proactiveSectionCount(section: string, overview: ProactiveOverview | null): number {
  if (!overview) return 0;
  if (section === "all") return overview.counts.tick_logs ?? 0;
  if (section === "drift" || section === "proactive") return overview.flow_counts[section] ?? 0;
  return overview.result_counts[section] ?? 0;
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);

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
