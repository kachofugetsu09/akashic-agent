import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, asPageResult, pageCount } from "./api";
import {
  encodePath,
  formatNumber,
  formatSessionKeyForTable,
  jsonText,
  memoryTypeClass,
  proactiveFlowLabel,
  proactiveResultLabel,
  proactiveSectionLabel,
  proactiveTickPreview,
  relativeTime,
  renderMarkdown,
  roleClass,
  shortTs,
  stripMarkdown,
} from "./format";
import { installDashboardGlobals, loadPluginAssets } from "./pluginRuntime";
import { PluginDetail } from "./PluginDetail";
import type {
  DashboardColumn,
  MemoryDetail,
  MemoryRow,
  MessageRow,
  PageResult,
  PluginConfig,
  PluginState,
  ProactiveOverview,
  ProactiveStep,
  ProactiveTick,
  SessionRow,
  SortOrder,
  ViewMode,
} from "./types";

type NavOpen = Record<string, boolean>;
type ModalState = { title: string; body: React.ReactNode } | null;
type MemoryScope = { channel: string; chatId: string } | null;

function App(): React.ReactElement {
  const [viewMode, setViewMode] = useState<ViewMode>("sessions");
  const [navOpen, setNavOpen] = useState<NavOpen>({ sessions: true, memory: false, proactive: false });
  const [plugins, setPlugins] = useState<PluginConfig[]>([]);
  const [pluginState, setPluginState] = useState<Record<string, PluginState>>({});
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionChannel, setSessionChannel] = useState("");
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionRow | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [messageSearch, setMessageSearch] = useState("");
  const [messageRole, setMessageRole] = useState("");
  const [messagePage, setMessagePage] = useState(1);
  const [messageSortBy, setMessageSortBy] = useState("ts");
  const [messageSortOrder, setMessageSortOrder] = useState<SortOrder>("desc");
  const [totalMessages, setTotalMessages] = useState(0);
  const [activeMessage, setActiveMessage] = useState<MessageRow | null>(null);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set());
  const [memories, setMemories] = useState<MemoryRow[]>([]);
  const [memoryTypeCounts, setMemoryTypeCounts] = useState<{ memory_type: string; total: number }[]>([]);
  const [memorySearch, setMemorySearch] = useState("");
  const [memoryType, setMemoryType] = useState("");
  const [memoryStatus, setMemoryStatus] = useState("");
  const [memoryScope, setMemoryScope] = useState<MemoryScope>(null);
  const [memoryPage, setMemoryPage] = useState(1);
  const [memorySortBy, setMemorySortBy] = useState("created_at");
  const [memorySortOrder, setMemorySortOrder] = useState<SortOrder>("desc");
  const [totalMemories, setTotalMemories] = useState(0);
  const [activeMemoryId, setActiveMemoryId] = useState<string | null>(null);
  const [activeMemoryDetail, setActiveMemoryDetail] = useState<MemoryDetail | null>(null);
  const [activeMemorySimilar, setActiveMemorySimilar] = useState<MemoryRow[]>([]);
  const [selectedMemoryIds, setSelectedMemoryIds] = useState<Set<string>>(new Set());
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
  const [modal, setModal] = useState<ModalState>(null);
  const [error, setError] = useState<string | null>(null);

  const messagePageSize = 25;
  const memoryPageSize = 25;
  const proactivePageSize = 25;
  const currentPluginId = viewMode.startsWith("plugin:") ? viewMode.slice(7) : "";
  const currentPlugin = plugins.find((plugin) => plugin.id === currentPluginId) ?? null;
  const currentPluginState = currentPluginId ? pluginState[currentPluginId] : null;

  const channels = useMemo(() => Array.from(new Set(sessions.map((session) => session.key.split(":")[0]).filter(Boolean))), [sessions]);

  const run = useCallback(async (work: () => Promise<void>) => {
    try {
      setError(null);
      await work();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  const loadSessions = useCallback(async () => {
    const params = new URLSearchParams();
    if (sessionSearch) params.set("q", sessionSearch);
    if (sessionChannel) params.set("channel", sessionChannel);
    params.set("page_size", "200");
    const payload = asPageResult(await api<PageResult<SessionRow>>(`/api/dashboard/sessions?${params.toString()}`));
    setSessions(payload.items);
    setActiveSession((current) => {
      if (!activeSessionKey) return current;
      return payload.items.find((session) => session.key === activeSessionKey) ?? null;
    });
  }, [activeSessionKey, sessionChannel, sessionSearch]);

  const loadMessages = useCallback(async () => {
    const params = new URLSearchParams();
    if (activeSessionKey) params.set("session_key", activeSessionKey);
    if (messageSearch) params.set("q", messageSearch);
    if (messageRole) params.set("role", messageRole);
    params.set("page", String(messagePage));
    params.set("page_size", String(messagePageSize));
    params.set("sort_by", messageSortBy);
    params.set("sort_order", messageSortOrder);
    const payload = asPageResult(await api<PageResult<MessageRow>>(`/api/dashboard/messages?${params.toString()}`));
    setMessages(payload.items);
    setTotalMessages(payload.total);
    setActiveMessage((current) => current && payload.items.some((item) => item.id === current.id) ? current : null);
  }, [activeSessionKey, messagePage, messageRole, messageSearch, messageSortBy, messageSortOrder]);

  const loadMemorySidebar = useCallback(async () => {
    const memoryTypes = ["procedure", "preference", "event", "profile"];
    const result: { memory_type: string; total: number }[] = [];
    for (const itemType of memoryTypes) {
      const params = memoryParams({
        search: memorySearch,
        type: itemType,
        status: memoryStatus,
        scope: memoryScope,
        page: 1,
        pageSize: 1,
        sortBy: "updated_at",
        sortOrder: "desc",
      });
      const payload = asPageResult(await api<PageResult<MemoryRow>>(`/api/dashboard/memories?${params.toString()}`));
      if (payload.total > 0) result.push({ memory_type: itemType, total: payload.total });
    }
    setMemoryTypeCounts(result);
    setTotalMemories(result.reduce((sum, item) => sum + item.total, 0));
  }, [memoryScope, memorySearch, memoryStatus]);

  const loadMemories = useCallback(async () => {
    const params = memoryParams({
      search: memorySearch,
      type: memoryType,
      status: memoryStatus,
      scope: memoryScope,
      page: memoryPage,
      pageSize: memoryPageSize,
      sortBy: memorySortBy,
      sortOrder: memorySortOrder,
    });
    const payload = asPageResult(await api<PageResult<MemoryRow>>(`/api/dashboard/memories?${params.toString()}`));
    setMemories(payload.items);
    setTotalMemories(payload.total);
    setActiveMemoryId((current) => current && payload.items.some((item) => item.id === current) ? current : null);
  }, [memoryPage, memoryScope, memorySearch, memorySortBy, memorySortOrder, memoryStatus, memoryType]);

  const loadMemoryDetail = useCallback(async (memoryId: string) => {
    const [detail, similar] = await Promise.all([
      api<MemoryDetail>(`/api/dashboard/memories/${encodePath(memoryId)}`),
      api<PageResult<MemoryRow>>(`/api/dashboard/memories/${encodePath(memoryId)}/similar?top_k=6`).catch(() => ({ items: [], total: 0 })),
    ]);
    setActiveMemoryDetail(detail);
    setActiveMemorySimilar(similar.items ?? []);
  }, []);

  const loadProactiveOverview = useCallback(async () => {
    setProactiveOverview(await api<ProactiveOverview>("/api/dashboard/proactive/overview"));
  }, []);

  const loadProactivePanel = useCallback(async () => {
    const params = new URLSearchParams();
    params.set("page", String(proactivePage));
    params.set("page_size", String(proactivePageSize));
    params.set("sort_by", proactiveSortBy);
    params.set("sort_order", proactiveSortOrder);
    if (proactiveSessionFilter) params.set("session_key", proactiveSessionFilter);
    if (proactiveSection === "reply" || proactiveSection === "skip") params.set("terminal_action", proactiveSection);
    if (proactiveSection === "drift" || proactiveSection === "proactive") params.set("flow", proactiveSection);
    if (["busy", "cooldown", "presence"].includes(proactiveSection)) params.set("gate_exit", proactiveSection);
    const payload = asPageResult(await api<PageResult<ProactiveTick>>(`/api/dashboard/proactive/tick_logs?${params.toString()}`));
    setProactiveItems(payload.items);
    setProactiveTotal(payload.total);
    setActiveProactiveKey((current) => current && payload.items.some((item) => item.tick_id === current) ? current : null);
  }, [proactivePage, proactiveSection, proactiveSessionFilter, proactiveSortBy, proactiveSortOrder]);

  const loadPluginPanel = useCallback(async (pluginId: string) => {
    const plugin = plugins.find((item) => item.id === pluginId);
    const state = pluginState[pluginId];
    if (!plugin || !state) return;
    const result = await plugin.fetchPage({ page: state.page, pageSize: state.pageSize });
    setPluginState((current) => ({
      ...current,
      [pluginId]: {
        ...current[pluginId],
        total: result.total || 0,
        items: result.items || [],
        activeRowKey: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeRowKey
          : null,
        activeDetail: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeDetail
          : null,
      },
    }));
  }, [pluginState, plugins]);

  const refreshCurrentView = useCallback(async () => {
    await loadSessions();
    if (viewMode === "memory") {
      await loadMemories();
      await loadMemorySidebar();
    } else if (viewMode === "proactive") {
      await loadProactiveOverview();
      await loadProactivePanel();
    } else if (viewMode.startsWith("plugin:")) {
      await loadPluginPanel(viewMode.slice(7));
    } else {
      await loadMessages();
    }
  }, [loadMemories, loadMemorySidebar, loadMessages, loadPluginPanel, loadProactiveOverview, loadProactivePanel, loadSessions, viewMode]);

  useEffect(() => {
    installDashboardGlobals((plugin) => {
      setPlugins((current) => current.some((item) => item.id === plugin.id) ? current : [...current, plugin]);
      setPluginState((current) => current[plugin.id] ? current : {
        ...current,
        [plugin.id]: { page: 1, pageSize: plugin.pageSize || 25, total: 0, items: [], activeRowKey: null, activeDetail: null },
      });
    });
    void loadPluginAssets();
  }, []);

  useEffect(() => {
    void run(async () => {
      await loadSessions();
      await loadMessages();
      await loadMemorySidebar();
      await loadProactiveOverview();
    });
  }, [loadMemorySidebar, loadMessages, loadProactiveOverview, loadSessions, run]);

  useEffect(() => {
    for (const plugin of plugins) {
      void run(async () => {
        const count = await plugin.getCount();
        setPluginState((current) => ({
          ...current,
          [plugin.id]: { ...current[plugin.id], total: typeof count === "number" ? count : current[plugin.id]?.total ?? 0 },
        }));
      });
    }
  }, [plugins, run]);

  const selectView = (next: ViewMode): void => {
    setViewMode(next);
    setNavOpen((current) => ({ ...current, [next]: true }));
    void run(async () => {
      if (next === "sessions") await loadMessages();
      else if (next === "memory") {
        await loadMemories();
        await loadMemorySidebar();
      } else if (next === "proactive") {
        await loadProactiveOverview();
        await loadProactivePanel();
      } else await loadPluginPanel(next.slice(7));
    });
  };

  const toggleNav = (kind: ViewMode): void => {
    if (viewMode !== kind) {
      selectView(kind);
      return;
    }
    setNavOpen((current) => ({ ...current, [kind]: !current[kind] }));
  };

  const sort = (scope: "messages" | "memory" | "proactive", key: string): void => {
    const flip = (currentKey: string, currentOrder: SortOrder): SortOrder => currentKey === key && currentOrder === "desc" ? "asc" : "desc";
    if (scope === "messages") {
      setMessageSortOrder(flip(messageSortBy, messageSortOrder));
      setMessageSortBy(key);
      setMessagePage(1);
    } else if (scope === "memory") {
      setMemorySortOrder(flip(memorySortBy, memorySortOrder));
      setMemorySortBy(key);
      setMemoryPage(1);
    } else {
      setProactiveSortOrder(flip(proactiveSortBy, proactiveSortOrder));
      setProactiveSortBy(key);
      setProactivePage(1);
    }
  };

  useEffect(() => {
    if (viewMode === "sessions") void run(loadMessages);
  }, [loadMessages, run, viewMode]);

  useEffect(() => {
    if (viewMode === "memory") void run(async () => {
      await loadMemories();
      await loadMemorySidebar();
    });
  }, [loadMemories, loadMemorySidebar, run, viewMode]);

  useEffect(() => {
    if (viewMode === "proactive") void run(loadProactivePanel);
  }, [loadProactivePanel, run, viewMode]);

  const currentPageCount = currentPluginState
    ? pageCount(currentPluginState.total, currentPluginState.pageSize)
    : viewMode === "memory"
      ? pageCount(totalMemories, memoryPageSize)
      : viewMode === "proactive"
        ? pageCount(proactiveTotal, proactivePageSize)
        : pageCount(totalMessages, messagePageSize);

  const currentPage = currentPluginState?.page ?? (viewMode === "memory" ? memoryPage : viewMode === "proactive" ? proactivePage : messagePage);

  const changePage = (delta: number): void => {
    if (currentPage + delta < 1 || currentPage + delta > currentPageCount) return;
    if (currentPluginId) {
      void run(async () => {
        const plugin = plugins.find((item) => item.id === currentPluginId);
        const state = pluginState[currentPluginId];
        if (!plugin || !state) return;
        const nextPage = state.page + delta;
        const result = await plugin.fetchPage({ page: nextPage, pageSize: state.pageSize });
        setPluginState((current) => ({
          ...current,
          [currentPluginId]: {
            ...current[currentPluginId],
            page: nextPage,
            total: result.total || 0,
            items: result.items || [],
            activeRowKey: null,
            activeDetail: null,
          },
        }));
      });
    } else if (viewMode === "memory") setMemoryPage((page) => page + delta);
    else if (viewMode === "proactive") setProactivePage((page) => page + delta);
    else setMessagePage((page) => page + delta);
  };

  const batchCount = viewMode === "memory" ? selectedMemoryIds.size : selectedMessageIds.size;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">A</div>
          <div>
            <div className="brand-title">Akashic Dashboard</div>
            <div className="brand-sub">Session / Memory Explorer</div>
          </div>
        </div>
        <TopbarFilters
          viewMode={viewMode}
          messageSearch={messageSearch}
          setMessageSearch={(value) => { setMessageSearch(value); setMessagePage(1); }}
          messageRole={messageRole}
          setMessageRole={(value) => { setMessageRole(value); setMessagePage(1); }}
          activeSessionKey={activeSessionKey}
          clearSession={() => { setActiveSessionKey(null); setActiveSession(null); setActiveMessage(null); setMessagePage(1); }}
          memorySearch={memorySearch}
          setMemorySearch={(value) => { setMemorySearch(value); setMemoryPage(1); }}
          memoryType={memoryType}
          setMemoryType={(value) => { setMemoryType(value); setMemoryPage(1); }}
          memoryStatus={memoryStatus}
          setMemoryStatus={(value) => { setMemoryStatus(value); setMemoryPage(1); }}
          memoryScope={memoryScope}
          clearMemoryScope={() => { setMemoryScope(null); setMemoryPage(1); }}
          proactiveSection={proactiveSection}
          proactiveSessionFilter={proactiveSessionFilter}
          clearProactiveSession={() => { setProactiveSessionFilter(""); setProactivePage(1); }}
        />
        <div className="topbar-view">
          <button className="ghost cache-summary-button" type="button" onClick={() => void run(async () => setModal(await cacheSummaryModal()))}>KV Cache</button>
          <div className="view-chip"><span>{viewLabel(viewMode, currentPlugin)}</span></div>
        </div>
      </header>

      <main className="workspace">
        <aside className="sessions-pane">
          <div className="pane-head">
            <div className="pane-kicker">Explorer</div>
            <div className="pane-title">{viewMode === "memory" ? `${totalMemories} 条记忆` : `${sessions.length} 个会话`}</div>
          </div>
          <div className="filters-stack">
            <label className="search search-small">
              <span>⌕</span>
              <input type="text" placeholder="过滤 session" value={sessionSearch} onChange={(event) => setSessionSearch(event.target.value.trim())} />
            </label>
            <select value={sessionChannel} onChange={(event) => setSessionChannel(event.target.value)}>
              <option value="">全部 channel</option>
              {channels.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
            </select>
          </div>
          <nav className="explorer-nav">
            <NavGroup label="Sessions" count={totalMessages || totalSessionMessages(sessions)} active={viewMode === "sessions"} open={!!navOpen.sessions} onToggle={() => toggleNav("sessions")}>
              <button className={`all-messages-row ${viewMode === "sessions" && !activeSessionKey ? "active" : ""}`} type="button" onClick={() => {
                setActiveSessionKey(null);
                setActiveSession(null);
                setActiveMessage(null);
                setMessagePage(1);
                selectView("sessions");
              }}>
                <span>全部消息</span><strong>{sessions.length}</strong>
              </button>
              <div className="session-list">
                {sessions.map((session) => (
                  <button key={session.key} className={`session-item ${activeSessionKey === session.key ? "active" : ""}`} type="button" onClick={() => {
                    setActiveSessionKey(session.key);
                    setActiveSession(session);
                    setActiveMessage(null);
                    setMessagePage(1);
                    selectView("sessions");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-type-dot memory-type-profile" />
                      <span className="nav-item-name mono">{formatSessionKeyForTable(session.key)}</span>
                      <span className="nav-item-count">{session.message_count}</span>
                    </div>
                    <div className="nav-item-desc">{relativeTime(session.updated_at)}</div>
                  </button>
                ))}
              </div>
            </NavGroup>
            <NavGroup label="Memory" count={totalMemories} active={viewMode === "memory"} open={!!navOpen.memory} onToggle={() => toggleNav("memory")}>
              <button className={`all-messages-row ${viewMode === "memory" && !memoryType ? "active" : ""}`} type="button" onClick={() => {
                setMemoryType("");
                setActiveMemoryId(null);
                setActiveMemoryDetail(null);
                setSelectedMemoryIds(new Set());
                setMemoryPage(1);
                selectView("memory");
              }}>
                <span>全部记忆</span><strong>{totalMemories}</strong>
              </button>
              <div className="memory-quick-list">
                {memoryTypeCounts.map((item) => (
                  <button key={item.memory_type} className={`memory-quick-item ${memoryType === item.memory_type ? "active" : ""}`} type="button" onClick={() => {
                    setMemoryType(item.memory_type);
                    setMemoryPage(1);
                    selectView("memory");
                  }}>
                    <div className="nav-item-row">
                      <span className={`nav-type-dot ${memoryTypeClass(item.memory_type)}`} />
                      <span className="nav-item-name">{item.memory_type}</span>
                      <span className="nav-item-count">{item.total}</span>
                    </div>
                  </button>
                ))}
              </div>
            </NavGroup>
            <NavGroup label="Proactive" count={proactiveOverview?.counts.tick_logs ?? proactiveTotal} active={viewMode === "proactive"} open={!!navOpen.proactive} onToggle={() => toggleNav("proactive")}>
              {["all", "drift", "proactive", "reply", "skip", "busy", "cooldown", "presence"].map((section) => (
                <button key={section} className={`proactive-quick-item ${proactiveSection === section ? "active" : ""}`} type="button" onClick={() => {
                  setProactiveSection(section);
                  setProactivePage(1);
                  selectView("proactive");
                }}>
                  <div className="nav-item-row">
                    <span className="nav-item-name">{proactiveSectionLabel(section)}</span>
                    <span className="nav-item-count">{proactiveSectionCount(section, proactiveOverview)}</span>
                  </div>
                </button>
              ))}
            </NavGroup>
            {plugins.map((plugin) => (
              <NavGroup key={plugin.id} label={plugin.label} count={pluginState[plugin.id]?.total ?? 0} active={viewMode === `plugin:${plugin.id}`} open={!!navOpen[`plugin:${plugin.id}`]} onToggle={() => toggleNav(`plugin:${plugin.id}`)}>
                <button className={`all-messages-row ${viewMode === `plugin:${plugin.id}` ? "active" : ""}`} type="button" onClick={() => selectView(`plugin:${plugin.id}`)}>
                  <span>{plugin.label}</span><strong>{pluginState[plugin.id]?.total ?? 0}</strong>
                </button>
              </NavGroup>
            ))}
          </nav>
        </aside>

        <section className="messages-pane">
          {batchCount > 0 && (
            <div className="batch-bar">
              <span>已选 {batchCount} 条</span>
              <button className="danger-ghost" type="button" onClick={() => void run(async () => {
                if (viewMode === "memory") {
                  await api("/api/dashboard/memories/batch-delete", { method: "POST", body: JSON.stringify({ ids: [...selectedMemoryIds] }) });
                  setSelectedMemoryIds(new Set());
                } else {
                  await api("/api/dashboard/messages/batch-delete", { method: "POST", body: JSON.stringify({ ids: [...selectedMessageIds] }) });
                  setSelectedMessageIds(new Set());
                }
                await refreshCurrentView();
              })}>批量删除</button>
              <button className="ghost" type="button" onClick={() => viewMode === "memory" ? setSelectedMemoryIds(new Set()) : setSelectedMessageIds(new Set())}>取消选择</button>
            </div>
          )}
          <TableHead viewMode={viewMode} plugin={currentPlugin} messageSortBy={messageSortBy} messageSortOrder={messageSortOrder} memorySortBy={memorySortBy} memorySortOrder={memorySortOrder} proactiveSortBy={proactiveSortBy} proactiveSortOrder={proactiveSortOrder} onSort={sort} />
          <div className="table-body">
            <Rows
              viewMode={viewMode}
              messages={messages}
              memories={memories}
              proactiveItems={proactiveItems}
              plugin={currentPlugin}
              pluginState={currentPluginState}
              selectedMessageIds={selectedMessageIds}
              selectedMemoryIds={selectedMemoryIds}
              activeMessage={activeMessage}
              activeMemoryId={activeMemoryId}
              activeProactiveKey={activeProactiveKey}
              onSelectMessage={setActiveMessage}
              onSelectMemory={(item) => void run(async () => {
                setActiveMemoryId(item.id);
                setSelectedMemoryIds(new Set());
                await loadMemoryDetail(item.id);
              })}
              onSelectProactive={(item) => void run(async () => {
                setActiveProactiveKey(item.tick_id);
                const [detail, steps] = await Promise.all([
                  api<ProactiveTick>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}`),
                  api<PageResult<ProactiveStep>>(`/api/dashboard/proactive/tick_logs/${encodePath(item.tick_id)}/steps`),
                ]);
                setActiveProactiveDetail(detail);
                setActiveProactiveSteps(steps.items ?? []);
              })}
              onSelectPluginRow={(row) => {
                if (!currentPlugin || !currentPluginState) return;
                const key = String(row[currentPlugin.rowKey] ?? "");
                void run(async () => {
                  const detail = currentPlugin.fetchDetail ? await currentPlugin.fetchDetail(row) : row;
                  setPluginState((current) => ({ ...current, [currentPlugin.id]: { ...current[currentPlugin.id], activeRowKey: key, activeDetail: detail } }));
                });
              }}
              setSelectedMessageIds={setSelectedMessageIds}
              setSelectedMemoryIds={setSelectedMemoryIds}
            />
          </div>
          <footer className="table-foot">
            <div>{tableMeta(viewMode, totalMessages, totalMemories, proactiveTotal, currentPlugin, currentPluginState, memoryScope, proactiveSessionFilter)}</div>
            <div className="pager">
              <button className="ghost" type="button" disabled={currentPage <= 1} onClick={() => changePage(-1)}>‹</button>
              <span>{currentPage} / {currentPageCount}</span>
              <button className="ghost" type="button" disabled={currentPage >= currentPageCount} onClick={() => changePage(1)}>›</button>
            </div>
          </footer>
        </section>

        <aside className="detail-pane">
          <DetailPane
            viewMode={viewMode}
            activeSession={activeSession}
            activeMessage={activeMessage}
            activeMemoryDetail={activeMemoryDetail}
            activeMemorySimilar={activeMemorySimilar}
            activeProactiveDetail={activeProactiveDetail}
            activeProactiveSteps={activeProactiveSteps}
            plugin={currentPlugin}
            pluginState={currentPluginState}
            setMemoryScope={(scope) => { setMemoryScope(scope); setMemoryPage(1); selectView("memory"); }}
            setProactiveSessionFilter={(key) => { setProactiveSessionFilter(key); setProactivePage(1); selectView("proactive"); }}
          />
        </aside>
      </main>
      {error && <div className="modal-backdrop" onClick={() => setError(null)}><div className="modal"><div className="modal-title">请求失败</div><p>{error}</p><div className="modal-actions"><button className="primary" type="button" onClick={() => setError(null)}>关闭</button></div></div></div>}
      {modal && <div className="modal-backdrop" onClick={() => setModal(null)}><div className="modal" onClick={(event) => event.stopPropagation()}><div className="modal-title">{modal.title}</div>{modal.body}<div className="modal-actions"><button className="primary" type="button" onClick={() => setModal(null)}>关闭</button></div></div></div>}
    </div>
  );
}

function TopbarFilters(props: {
  viewMode: ViewMode;
  messageSearch: string;
  setMessageSearch(value: string): void;
  messageRole: string;
  setMessageRole(value: string): void;
  activeSessionKey: string | null;
  clearSession(): void;
  memorySearch: string;
  setMemorySearch(value: string): void;
  memoryType: string;
  setMemoryType(value: string): void;
  memoryStatus: string;
  setMemoryStatus(value: string): void;
  memoryScope: MemoryScope;
  clearMemoryScope(): void;
  proactiveSection: string;
  proactiveSessionFilter: string;
  clearProactiveSession(): void;
}): React.ReactElement {
  return (
    <div className="topbar-filters">
      {props.viewMode === "memory" ? (
        <div className="filter-row">
          <label className="search"><span>⌕</span><input type="text" placeholder="搜索 memory / source_ref" value={props.memorySearch} onChange={(event) => props.setMemorySearch(event.target.value.trim())} /></label>
          <select value={props.memoryType} onChange={(event) => props.setMemoryType(event.target.value)}>
            <option value="">全部 type</option><option value="procedure">procedure</option><option value="preference">preference</option><option value="event">event</option><option value="profile">profile</option>
          </select>
          <select value={props.memoryStatus} onChange={(event) => props.setMemoryStatus(event.target.value)}>
            <option value="">全部 status</option><option value="active">active</option><option value="superseded">superseded</option>
          </select>
          {props.memoryScope && <Chip label="scope" value={`${props.memoryScope.channel}:${props.memoryScope.chatId}`} onClear={props.clearMemoryScope} />}
        </div>
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
      )}
    </div>
  );
}

function Chip(props: { label: string; value: string; onClear(): void }): React.ReactElement {
  return <div className="active-session-chip"><span>{props.label}</span><code>{props.value}</code><button type="button" onClick={props.onClear}>×</button></div>;
}

function NavGroup(props: { label: string; count: number; active: boolean; open: boolean; onToggle(): void; children: React.ReactNode }): React.ReactElement {
  return (
    <section className={`nav-group ${props.active ? "active" : ""}`}>
      <button className="nav-group-toggle" type="button" onClick={props.onToggle}>
        <span className="nav-group-caret">{props.open ? "▾" : "▸"}</span>
        <span className="nav-group-label">{props.label}</span>
        <span className="nav-group-count">{props.count}</span>
      </button>
      <div className={`nav-group-body ${props.open ? "" : "hidden"}`}>{props.children}</div>
    </section>
  );
}

function TableHead(props: {
  viewMode: ViewMode;
  plugin: PluginConfig | null;
  messageSortBy: string;
  messageSortOrder: SortOrder;
  memorySortBy: string;
  memorySortOrder: SortOrder;
  proactiveSortBy: string;
  proactiveSortOrder: SortOrder;
  onSort(scope: "messages" | "memory" | "proactive", key: string): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    return <div className="table-head" style={{ gridTemplateColumns: gridTemplate(props.plugin.columns) }}>{props.plugin.columns.map((col) => <div key={col.key}>{col.label}</div>)}</div>;
  }
  if (props.viewMode === "memory") {
    return <div className="table-head mode-memory">
      <div />
      <SortHead label="Type" active={props.memorySortBy === "memory_type"} order={props.memorySortOrder} onClick={() => props.onSort("memory", "memory_type")} />
      <div>Summary</div>
      <SortHead label="Uses" active={props.memorySortBy === "reinforcement"} order={props.memorySortOrder} onClick={() => props.onSort("memory", "reinforcement")} />
      <SortHead label="Weight" active={props.memorySortBy === "emotional_weight"} order={props.memorySortOrder} onClick={() => props.onSort("memory", "emotional_weight")} />
      <div>Source</div>
      <SortHead label="Created" active={props.memorySortBy === "created_at"} order={props.memorySortOrder} onClick={() => props.onSort("memory", "created_at")} />
      <SortHead label="Updated" active={props.memorySortBy === "updated_at"} order={props.memorySortOrder} onClick={() => props.onSort("memory", "updated_at")} />
      <div>Status</div><div />
    </div>;
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
  memories: MemoryRow[];
  proactiveItems: ProactiveTick[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  selectedMessageIds: Set<string>;
  selectedMemoryIds: Set<string>;
  activeMessage: MessageRow | null;
  activeMemoryId: string | null;
  activeProactiveKey: string | null;
  onSelectMessage(item: MessageRow): void;
  onSelectMemory(item: MemoryRow): void;
  onSelectProactive(item: ProactiveTick): void;
  onSelectPluginRow(row: Record<string, unknown>): void;
  setSelectedMessageIds(value: Set<string>): void;
  setSelectedMemoryIds(value: Set<string>): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin && props.pluginState) {
    const grid = gridTemplate(props.plugin.columns);
    return <>{props.pluginState.items.length ? props.pluginState.items.map((item) => {
      const key = String(item[props.plugin!.rowKey] ?? "");
      return <div key={key} className={`table-row ${props.pluginState!.activeRowKey === key ? "active" : ""} ${props.plugin!.rowClass?.(item) ?? ""}`} style={{ gridTemplateColumns: grid }} onClick={() => props.onSelectPluginRow(item)}>
        {props.plugin!.columns.map((col) => <div key={col.key} className={col.cellClass ?? ""} title={col.rawTitle ? String(item[col.key] ?? "") : undefined}>{formatPluginCell(props.plugin!, col, item)}</div>)}
      </div>;
    }) : <div className="empty-state">{props.plugin.emptyMessage || "暂无记录。"}</div>}</>;
  }
  if (props.viewMode === "memory") {
    return <>{props.memories.map((item) => <div key={item.id} className={`table-row mode-memory ${props.activeMemoryId === item.id ? "active" : ""} ${props.selectedMemoryIds.has(item.id) ? "selected" : ""}`} onClick={() => props.onSelectMemory(item)}>
      <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={props.selectedMemoryIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMemoryIds, props.setSelectedMemoryIds)} /></label>
      <div className="cell-type"><span className={`type-pill ${memoryTypeClass(item.memory_type)}`}>{item.memory_type}</span></div>
      <div className="content-preview">{item.summary}</div>
      <div className="cell-metric">{item.reinforcement}</div>
      <div className="cell-metric">{item.emotional_weight}</div>
      <div className="cell-source">{item.source_ref || "-"}</div>
      <div className="cell-time">{shortTs(item.created_at)}</div>
      <div className="cell-time">{shortTs(item.updated_at)}</div>
      <div className="cell-status"><span className={`status-pill memory-status-${item.status}`}>{item.status}</span></div>
      <div />
    </div>)}</>;
  }
  if (props.viewMode === "proactive") {
    return <>{props.proactiveItems.map((item) => <div key={item.tick_id} className={`table-row mode-proactive-ticks ${props.activeProactiveKey === item.tick_id ? "active" : ""}`} onClick={() => props.onSelectProactive(item)}>
      <div className="cell-session mono">{formatSessionKeyForTable(item.session_key)}</div>
      <div className="cell-time">{shortTs(item.started_at)}</div>
      <div className="proactive-status-cell"><span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span><span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span></div>
      <div className="content-preview">{proactiveTickPreview(item)}</div>
      <div />
    </div>)}</>;
  }
  return <>{props.messages.map((item) => <div key={item.id} className={`table-row mode-messages ${props.activeMessage?.id === item.id ? "active" : ""} ${props.selectedMessageIds.has(item.id) ? "selected" : ""}`} onClick={() => props.onSelectMessage(item)}>
    <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={props.selectedMessageIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMessageIds, props.setSelectedMessageIds)} /></label>
    <div className="cell-session mono" title={item.session_key}>{formatSessionKeyForTable(item.session_key)}</div>
    <div className="cell-seq mono">#{item.seq}</div>
    <div className="content-preview">{stripMarkdown(item.content)}</div>
    <div className="cell-time mono">{shortTs(item.ts)}</div>
    <div><span className={`role-pill ${roleClass(item.role)}`}>{item.role}</span></div>
    <div />
  </div>)}</>;
}

function DetailPane(props: {
  viewMode: ViewMode;
  activeSession: SessionRow | null;
  activeMessage: MessageRow | null;
  activeMemoryDetail: MemoryDetail | null;
  activeMemorySimilar: MemoryRow[];
  activeProactiveDetail: ProactiveTick | null;
  activeProactiveSteps: ProactiveStep[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  setMemoryScope(scope: MemoryScope): void;
  setProactiveSessionFilter(key: string): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    return <PluginDetail plugin={props.plugin} item={props.pluginState?.activeDetail ?? null} />;
  }
  if (props.viewMode === "memory") {
    const item = props.activeMemoryDetail;
    if (!item) return <EmptyDetail text="点开 memory 后，这里会显示完整字段、JSON 和相似记忆。" />;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">记忆详情</div><div className="detail-subtext">{item.id}</div></div></div>
      <div className="detail-block"><div className="detail-label">Summary</div><div className="detail-content">{item.summary}</div></div>
      <div className="detail-grid">
        {detailRow("type", <span className={`type-pill ${memoryTypeClass(item.memory_type)}`}>{item.memory_type}</span>)}
        {detailRow("status", <span className={`status-pill memory-status-${item.status}`}>{item.status}</span>)}
        {detailRow("source_ref", <code>{item.source_ref || "-"}</code>)}
        {detailRow("embedding", <code>{item.has_embedding ? `${item.embedding_dim} dims` : "none"}</code>)}
      </div>
      {Boolean(item.extra_json.scope_channel || item.extra_json.scope_chat_id) && <button className="ghost" type="button" onClick={() => props.setMemoryScope({ channel: String(item.extra_json.scope_channel ?? ""), chatId: String(item.extra_json.scope_chat_id ?? "") })}>查看同 scope 记忆</button>}
      <div className="detail-block"><div className="detail-label">Extra JSON</div><pre className="json-tree">{jsonText(item.extra_json)}</pre></div>
      <div className="detail-block"><div className="detail-label">Similar</div><div className="detail-similar-list">{props.activeMemorySimilar.length ? props.activeMemorySimilar.map((similar) => <div key={similar.id} className="detail-callout"><code>{similar.id}</code><div>{similar.summary}</div></div>) : <div className="muted-text">没有相似记忆。</div>}</div></div>
    </div>;
  }
  if (props.viewMode === "proactive") {
    const item = props.activeProactiveDetail;
    if (!item) return <EmptyDetail text="点开 tick 后，这里会显示 proactive 执行详情和工具链。" />;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Tick 详情</div><div className="detail-subtext">{item.tick_id}</div></div></div>
      <button className="ghost" type="button" onClick={() => props.setProactiveSessionFilter(item.session_key)}>只看这个 session</button>
      <div className="detail-grid">
        {detailRow("session", <code>{item.session_key}</code>)}
        {detailRow("started", <code>{item.started_at}</code>)}
        {detailRow("result", <span className={`status-pill proactive-result-${proactiveResultLabel(item)}`}>{proactiveResultLabel(item)}</span>)}
        {detailRow("flow", <span className={`type-pill proactive-flow-${proactiveFlowLabel(item).toLowerCase()}`}>{proactiveFlowLabel(item)}</span>)}
      </div>
      {item.final_message && <div className="detail-block"><div className="detail-label">Final Message</div><div className="detail-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(item.final_message) }} /></div>}
      <div className="detail-block"><div className="detail-label">Steps</div>{props.activeProactiveSteps.length ? props.activeProactiveSteps.map((step) => <div key={`${step.phase}-${step.step_index}`} className="tool-step"><div className="tool-step-head"><div className="tool-step-title"><span className="status-pill">step {step.step_index}</span><span className="type-pill">{step.tool_name}</span></div></div><pre className="json-tree">{jsonText(step.tool_args)}</pre><div className="detail-content tool-result">{step.tool_result_text}</div></div>) : <div className="muted-text">没有记录到工具调用。</div>}</div>
    </div>;
  }
  if (props.activeMessage) {
    const message = props.activeMessage;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">消息详情</div><div className="detail-subtext">{message.session_key} · #{message.seq}</div></div></div>
      <div className="detail-grid">
        {detailRow("role", <span className={`role-pill ${roleClass(message.role)}`}>{message.role}</span>)}
        {detailRow("time", <code>{message.ts}</code>)}
        {detailRow("id", <code>{message.id}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Content</div><div className="detail-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} /></div>
      <div className="detail-block"><div className="detail-label">Extra</div><pre className="json-tree">{jsonText(message.extra)}</pre></div>
      <div className="detail-block"><div className="detail-label">Tool Chain</div><pre className="json-tree">{jsonText(message.tool_chain)}</pre></div>
    </div>;
  }
  if (props.activeSession) {
    const session = props.activeSession;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Session 详情</div><div className="detail-subtext">{session.key}</div></div></div>
      <div className="detail-grid">
        {detailRow("messages", <code>{session.message_count}</code>)}
        {detailRow("updated", <code>{session.updated_at}</code>)}
        {detailRow("last_consolidated", <code>{session.last_consolidated}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Metadata</div><pre className="json-tree">{jsonText(session.metadata)}</pre></div>
    </div>;
  }
  return <EmptyDetail text="点开消息、session 或 memory 后，这里会显示完整内容、字段和 JSON 信息。" />;
}

function EmptyDetail(props: { text: string }): React.ReactElement {
  return <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">{props.text}</div></div>;
}

function detailRow(label: string, value: React.ReactNode): React.ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{label}</div><div className="detail-row-val">{value}</div></div>;
}

function memoryParams(args: {
  search: string;
  type: string;
  status: string;
  scope: MemoryScope;
  page: number;
  pageSize: number;
  sortBy: string;
  sortOrder: SortOrder;
}): URLSearchParams {
  const params = new URLSearchParams();
  if (args.search) params.set("q", args.search);
  if (args.type) params.set("memory_type", args.type);
  if (args.status) params.set("status", args.status);
  if (args.scope?.channel) params.set("scope_channel", args.scope.channel);
  if (args.scope?.chatId) params.set("scope_chat_id", args.scope.chatId);
  params.set("page", String(args.page));
  params.set("page_size", String(args.pageSize));
  params.set("sort_by", args.sortBy);
  params.set("sort_order", args.sortOrder);
  return params;
}

function toggleSet(id: string, checked: boolean, source: Set<string>, update: (value: Set<string>) => void): void {
  const next = new Set(source);
  if (checked) next.add(id);
  else next.delete(id);
  update(next);
}

function gridTemplate(columns: DashboardColumn[]): string {
  return columns.map((col) => col.flex ? "1fr" : col.width ? `${col.width}px` : "auto").join(" ");
}

function formatPluginCell(plugin: PluginConfig, column: DashboardColumn, item: Record<string, unknown>): string {
  const value = item[column.key];
  const formatter = plugin.formatters?.[column.fmt || ""] ?? (window as Window & { AkashicDashboard?: { _formatters: Record<string, (value: unknown, item?: Record<string, unknown>) => string> } }).AkashicDashboard?._formatters[column.fmt || "text"];
  return formatter ? formatter(value, item) : String(value ?? "");
}

function tableMeta(viewMode: ViewMode, totalMessages: number, totalMemories: number, proactiveTotal: number, plugin: PluginConfig | null, pluginState: PluginState | null, memoryScope: MemoryScope, proactiveSessionFilter: string): string {
  if (plugin && pluginState) return plugin.countTitle ? plugin.countTitle(pluginState.total) : `共 ${pluginState.total} 条`;
  if (viewMode === "memory") return memoryScope ? `共 ${totalMemories} 条记忆 · scope: ${memoryScope.channel}:${memoryScope.chatId}` : `共 ${totalMemories} 条记忆`;
  if (viewMode === "proactive") return proactiveSessionFilter ? `共 ${proactiveTotal} 条 tick · session: ${proactiveSessionFilter}` : `共 ${proactiveTotal} 条 tick`;
  return `共 ${totalMessages} 条`;
}

function totalSessionMessages(sessions: SessionRow[]): number {
  return sessions.reduce((sum, session) => sum + (session.message_count || 0), 0);
}

function proactiveSectionCount(section: string, overview: ProactiveOverview | null): number {
  if (!overview) return 0;
  if (section === "all") return overview.counts.tick_logs ?? 0;
  if (section === "drift" || section === "proactive") return overview.flow_counts[section] ?? 0;
  return overview.result_counts[section] ?? 0;
}

function viewLabel(viewMode: ViewMode, plugin: PluginConfig | null): string {
  if (plugin) return plugin.viewLabel || plugin.label;
  if (viewMode === "memory") return "memory";
  if (viewMode === "proactive") return "proactive";
  return "messages";
}

async function cacheSummaryModal(): Promise<ModalState> {
  const summary = await api<Record<string, unknown>>("/api/dashboard/cache/summary");
  return {
    title: "KV Cache",
    body: <div className="detail-wrap">
      <div className="cache-summary-grid">
        {["tracked_turn_count", "prompt_tokens", "hit_tokens", "miss_tokens"].map((key) => <div key={key} className="cache-metric-card"><div className="cache-metric-label">{key}</div><div className="cache-metric-value">{formatNumber(summary[key])}</div></div>)}
      </div>
      <pre className="json-tree">{jsonText(summary)}</pre>
    </div>,
  };
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
