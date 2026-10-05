import {
  MessageSquarePlus,
  Search,
  Pin,
  PinOff,
  ArrowUpDown,
  SunMoon,
  X,
} from "lucide-react";
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ConversationNavigation, ConversationSessionRow, type ConversationRowDrag, type ConversationSession } from "./conversation-navigation";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { activateShellRailAction, useShellRailActions } from "./shell-rail-actions";
import { ProjectNavigation, ProjectNavigationRow, type ProjectSessionItem } from "./project-navigation";
import { sessionLabel } from "./web-chat-message-data";
import type { NavigationPin, NavigationPinsState } from "./use-navigation-pins";
import {
  SIDEBAR_RAIL_MAX_REM, SIDEBAR_RAIL_MIN_REM, SIDEBAR_RAIL_STEP_REM,
  type SidebarRailControl,
} from "./use-sidebar-rail";
import type { NavigationRowAction } from "./navigation-row-menu";
import type { PendingProjectRow, ProjectMemory, ProjectRow } from "./web-projects";
import { ProjectDirectoryDialog } from "./project-directory-dialog";
import { Folder } from "lucide-react";

export interface DesktopSidebarSession extends Omit<ConversationSession, "active" | "state"> {
  active: boolean;
  projectId?: string;
  projectScoped?: boolean;
  updatedAt?: string;
  createdAt?: string;
}

/** 会话排序是客户端展示投影：只作用于最近会话区与项目内子列表，置顶区保持服务端顺序。 */
type SessionSortId = "activity" | "created" | "title" | "manual";

const SESSION_SORT_CHOICES: readonly { id: SessionSortId; label: string }[] = [
  { id: "activity", label: "最近活动" },
  { id: "created", label: "最近创建" },
  { id: "title", label: "标题" },
  { id: "manual", label: "手动排序" },
];
const SESSION_SORT_KEY = "akashic.chat.session-sort";
const SESSION_ORDER_KEY = "akashic.chat.session-order";

function readSessionSort(): SessionSortId {
  try {
    const value = localStorage.getItem(SESSION_SORT_KEY);
    return SESSION_SORT_CHOICES.some((choice) => choice.id === value) ? value as SessionSortId : "activity";
  } catch {
    return "activity";
  }
}

function readSessionOrder(): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(SESSION_ORDER_KEY) ?? "[]");
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

/** 把持久化顺序与当前全集合并：新出现（未排序过）的会话按现有展示序排到最前。 */
function mergeSessionOrder(stored: readonly string[], ids: readonly string[]): string[] {
  const display = new Set(ids);
  const kept = stored.filter((id) => display.has(id));
  const ordered = new Set(kept);
  const fresh = ids.filter((id) => !ordered.has(id));
  return [...fresh, ...kept];
}

/** 时间倒序；缺失时间戳的行（如置顶解析补齐的目录外会话）稳定排到末尾。 */
function compareSessionTime(a: string | undefined, b: string | undefined): number {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return Date.parse(b) - Date.parse(a);
}

function sortSessions(rows: DesktopSidebarSession[], sort: SessionSortId, order: readonly string[]): DesktopSidebarSession[] {
  if (sort === "title") return [...rows].sort((a, b) => a.title.localeCompare(b.title, "zh-Hans-CN"));
  if (sort === "created") return [...rows].sort((a, b) => compareSessionTime(a.createdAt, b.createdAt));
  if (sort === "manual") {
    const rank = new Map(order.map((id, index) => [id, index]));
    return [...rows].sort((a, b) => (rank.get(a.id) ?? -1) - (rank.get(b.id) ?? -1));
  }
  return [...rows].sort((a, b) => compareSessionTime(a.updatedAt, b.updatedAt));
}

export interface DesktopSidebarProjects {
  items: ProjectRow[];
  pending: PendingProjectRow[];
  pendingError: string;
  activeProjectId: string;
  memoryInstalled: boolean;
  onNewChat: (projectId: string) => void;
  onCreate: (name: string, memory: ProjectMemory, directory: string | null) => Promise<void>;
  onContinue: (key: string) => Promise<void>;
  onStop: (key: string) => void;
  onOpenCreate?: () => void;
  onBindDirectory: (projectId: string, path: string) => Promise<void>;
}

export interface DesktopSidebarProps {
  embeddedShell: boolean;
  surface: "chat" | "runtime";
  sessions: DesktopSidebarSession[];
  activeSessionId: string;
  pendingSessionId: string;
  chatReady: boolean;
  themeLabel: string;
  projects?: DesktopSidebarProjects;
  navigationPins: NavigationPinsState;
  onSelectSession: (sessionId: string) => void;
  onPrefetchSession?: (sessionId: string) => void;
  onCycleTheme: () => void;
  onNewChat: () => void;
  /** 会话软删：行内两步确认后调用；Promise 拒绝时行保持原位。 */
  onDeleteSession?: (sessionId: string, title: string) => Promise<void>;
  /** 会话重命名：双击或菜单进入行内编辑；Promise 拒绝时编辑态保留。 */
  onRenameSession?: (sessionId: string, title: string) => Promise<void>;
  /** 最近一次删除的撤销窗口；null 时不在导航区展示提示行。 */
  deletedNotice?: { key: string; title: string } | null;
  onRestoreSession?: (sessionId: string) => void;
  onDismissDeletedNotice?: () => void;
  /** 宽屏侧栏的宽度控制；抽屉（compact）不带此 prop，因此不渲染拖拽柄。 */
  rail?: SidebarRailControl;
}

/** 会话导航竖栏；底部动作由宿主 Shell 经 shell.rail-actions.v1 桥下发，主题行是 chat 域原生。 */
export const DesktopSidebar = memo(function DesktopSidebar({
  embeddedShell,
  surface,
  sessions,
  activeSessionId,
  pendingSessionId,
  themeLabel,
  projects,
  navigationPins,
  onSelectSession,
  onPrefetchSession,
  onCycleTheme,
  onNewChat,
  onDeleteSession,
  onRenameSession,
  deletedNotice,
  onRestoreSession,
  onDismissDeletedNotice,
  rail,
}: DesktopSidebarProps) {
  const [query, setQuery] = useState("");
  const sidebarRef = useRef<HTMLElement>(null);
  const [directoryProjectId, setDirectoryProjectId] = useState("");
  const [sessionSort, setSessionSort] = useState(readSessionSort);
  const [sessionOrder, setSessionOrder] = useState(readSessionOrder);
  // 行拖拽的瞬态：被拖源 id + 当前插入指示（目标行 id 与上/下半）。提交时写入
  // sessionOrder 并把排序切到「手动排序」——拖拽本身就是选择手动模式的意图。
  const [drag, setDrag] = useState<{ id: string; over: { id: string; half: "before" | "after" } | null } | null>(null);
  const directoryProject = projects?.items.find((project) => project.id === directoryProjectId);
  const directoryAction = (project: ProjectRow): NavigationRowAction => ({
    label: project.directory ? "查看固定目录" : "选择目录",
    icon: <Folder size={18} aria-hidden="true" />,
    onSelect: () => setDirectoryProjectId(project.id),
  });
  const needle = query.trim().toLowerCase();
  const searching = Boolean(needle);
  const allSessions = useMemo(() => {
    const rows = new Map<string, DesktopSidebarSession>(navigationPins.sessions.map((session) => [session.key, {
      id: session.key, title: sessionLabel(session), preview: session.message_count === undefined ? "" : `${session.message_count} 条消息`,
      active: activeSessionId === session.key,
      projectId: session.scope?.project, projectScoped: Object.hasOwn(session.scope ?? {}, "project"),
    }]));
    // 目录里的新鲜 metadata 优先；置顶解析补齐不在最近分页内的会话。
    for (const session of sessions) rows.set(session.id, session);
    return [...rows.values()];
  }, [navigationPins.sessions, sessions, activeSessionId]);
  const filteredSessions = useMemo(() => needle
    ? allSessions.filter((session) => `${session.title} ${session.preview}`.toLowerCase().includes(needle))
    : allSessions, [allSessions, needle]);
  // 排序只重排展示顺序：最近会话区与项目内子列表共享同一份投影，置顶区不经过它。
  const mergedOrder = useMemo(
    () => mergeSessionOrder(sessionOrder, allSessions.map((session) => session.id)),
    [sessionOrder, allSessions],
  );
  const sortedSessions = useMemo(() => sortSessions(filteredSessions, sessionSort, mergedOrder), [filteredSessions, sessionSort, mergedOrder]);
  useEffect(() => {
    try { localStorage.setItem(SESSION_SORT_KEY, sessionSort); }
    catch { /* 禁用存储时仍允许本次浏览的排序选择。 */ }
  }, [sessionSort]);
  useEffect(() => {
    try { localStorage.setItem(SESSION_ORDER_KEY, JSON.stringify(sessionOrder)); }
    catch { /* 禁用存储时仍允许本次浏览的手动排序。 */ }
  }, [sessionOrder]);
  // 行拖拽期间在 document 层接受 dragover/drop：行外不显示"禁止"光标，
  // dragend 的提交仍由行自己的 onDrop 完成。
  useEffect(() => {
    if (!drag) return;
    const acceptDrag = (event: DragEvent) => {
      event.preventDefault();
      if (event.dataTransfer !== null) event.dataTransfer.dropEffect = "move";
    };
    const acceptDrop = (event: DragEvent) => { event.preventDefault(); };
    document.addEventListener("dragover", acceptDrag);
    document.addEventListener("drop", acceptDrop);
    return () => {
      document.removeEventListener("dragover", acceptDrag);
      document.removeEventListener("drop", acceptDrop);
    };
  }, [drag]);
  const commitManualDrop = (targetId: string, half: "before" | "after") => {
    if (!drag || drag.id === targetId) return;
    const order = mergedOrder.filter((id) => id !== drag.id);
    const targetIndex = order.indexOf(targetId);
    if (targetIndex === -1) return;
    order.splice(targetIndex + (half === "after" ? 1 : 0), 0, drag.id);
    // 当前分页外的会话保留顺序尾巴，不因为本次可见提交丢掉位置。
    const offscreen = sessionOrder.filter((id) => !mergedOrder.includes(id));
    setSessionOrder([...order, ...offscreen]);
    setSessionSort("manual");
  };
  const sessionDrag = searching ? undefined : (session: ConversationSession): ConversationRowDrag => ({
    active: drag !== null,
    source: drag?.id === session.id,
    marker: drag?.over?.id === session.id ? drag.over.half : undefined,
    start: () => setDrag({ id: session.id, over: null }),
    end: () => setDrag(null),
    hover: (half) => setDrag((current) => current
      && (current.over?.id !== session.id || current.over.half !== half)
      ? { ...current, over: { id: session.id, half } }
      : current),
    drop: (half) => {
      commitManualDrop(session.id, half);
      setDrag(null);
    },
  });
  // 目录未解析/归档不改变 Session.scope；回退到最近会话也不获得置顶资格。
  const knownProjects = useMemo(() => new Map(projects?.items.map((project) => [project.id, project])), [projects?.items]);
  const sessionsByProject = useMemo(() => {
    const groups = new Map<string, ProjectSessionItem[]>();
    for (const session of sortedSessions) {
      if (!session.projectId || !knownProjects.has(session.projectId)) continue;
      const group = groups.get(session.projectId) ?? [];
      group.push({
        id: session.id, title: session.title,
        active: surface === "chat" && session.active,
      });
      groups.set(session.projectId, group);
    }
    return groups;
  }, [sortedSessions, knownProjects, surface]);
  const pinnedProjects = new Set(navigationPins.pins.filter((pin) => pin.kind === "project").map((pin) => pin.id));
  const pinnedSessions = new Set(navigationPins.pins.filter((pin) => pin.kind === "session").map((pin) => pin.id));
  const recentSessions = sortedSessions.filter((session) => !pinnedSessions.has(session.id)
    && (!session.projectId || !knownProjects.has(session.projectId)));
  const visibleProject = (project: ProjectRow) => !needle || project.name.toLowerCase().includes(needle)
    || Boolean(sessionsByProject.get(project.id)?.length);
  const otherProjects = projects?.items.filter((project) => !pinnedProjects.has(project.id) && visibleProject(project)) ?? [];
  const hasMatches = recentSessions.length > 0 || otherProjects.length > 0 || navigationPins.pins.some((pin) => {
    if (pin.kind === "project") {
      const project = knownProjects.get(pin.id);
      if (project) return visibleProject(project);
    } else {
      const session = allSessions.find((item) => item.id === pin.id);
      if (session && !session.projectId && !session.projectScoped) {
        return `${session.title} ${session.preview}`.toLowerCase().includes(needle);
      }
    }
    return pin.id.toLowerCase().includes(needle);
  });
  const pinAction = (pin: NavigationPin, pinned: boolean): NavigationRowAction[] => [{
    label: pinned ? "取消置顶" : "置顶",
    icon: pinned ? <PinOff size={18} aria-hidden="true" /> : <Pin size={18} aria-hidden="true" />,
    disabled: !navigationPins.ready || navigationPins.pending,
    onSelect: () => { void navigationPins.setPinned(pin, !pinned); },
  }];
  const sessionView = (session: DesktopSidebarSession): ConversationSession => ({
    ...session, active: surface === "chat" && session.active,
  });
  const deleteHandler = onDeleteSession
    ? (session: ConversationSession) => onDeleteSession(session.id, session.title)
    : undefined;
  const railActions = useShellRailActions();

  return (
    <aside ref={sidebarRef} className="chat-sidebar chat-sidebar--entry">
      <div className="chat-sidebar__toolbar">
        <button type="button" className="chat-sidebar__new" onClick={() => onNewChat()}>
          <MessageSquarePlus size={18} strokeWidth={1.75} aria-hidden="true" />
          <span>新会话</span>
        </button>
        <label className="chat-sidebar__search">
          <Search size={14} aria-hidden="true" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索会话"
            aria-label="搜索会话"
          />
        </label>
      </div>

      <div className="chat-sidebar__navigation">
      {searching && !hasMatches ? <p className="navigation-search-empty" role="status">没有匹配的会话或项目</p> : null}
      {navigationPins.error ? <div className="navigation-pins-error" role="alert">
        <span>{navigationPins.error}</span>
        <button type="button" disabled={navigationPins.pending} onClick={() => { void navigationPins.reload(); }}>刷新置顶列表</button>
      </div> : null}
      {deletedNotice ? <div className="session-delete-notice" role="status">
        <span title={deletedNotice.title}>已删除「{deletedNotice.title}」</span>
        <button type="button" onClick={() => onRestoreSession?.(deletedNotice.key)}>撤销</button>
        <button type="button" className="session-delete-notice__dismiss" aria-label="关闭提示" onClick={onDismissDeletedNotice}>
          <X size={14} aria-hidden="true" />
        </button>
      </div> : null}
      {navigationPins.pins.length ? <section className="pinned-navigation" aria-label="置顶">
        <header className="project-navigation__header"><span>置顶</span></header>
        {navigationPins.pins.map((pin) => {
          if (pin.kind === "project") {
            const project = knownProjects.get(pin.id);
            if (project && projects) return visibleProject(project) ? <ProjectNavigationRow
              key={`project:${pin.id}`} project={project}
              items={sessionsByProject.get(pin.id) ?? []}
              open={searching || navigationPins.expandedProjects.has(pin.id)}
              active={projects.activeProjectId === pin.id} pendingSessionId={pendingSessionId}
              onToggle={() => navigationPins.toggleProject(pin.id)}
              onNewChat={() => projects.onNewChat(pin.id)}
              onSelectSession={onSelectSession} onPrefetchSession={onPrefetchSession}
              actions={[directoryAction(project), ...pinAction(pin, true)]} searching={searching}
            /> : null;
          } else {
            const session = allSessions.find((item) => item.id === pin.id);
            if (session && !session.projectId && !session.projectScoped) return !needle
              || `${session.title} ${session.preview}`.toLowerCase().includes(needle) ? <ConversationSessionRow
                key={`session:${pin.id}`} session={sessionView(session)} pendingSessionId={pendingSessionId}
                onActivate={onSelectSession} onPrefetch={onPrefetchSession}
                actions={pinAction(pin, true)}
                onDelete={onDeleteSession ? () => onDeleteSession(session.id, session.title) : undefined}
                onRename={onRenameSession ? (title: string) => onRenameSession(session.id, title) : undefined}
              /> : null;
          }
          if (needle && !pin.id.toLowerCase().includes(needle)) return null;
          const title = pin.kind === "project" ? "项目暂不可用" : "会话暂不可用";
          return <ConversationSessionRow key={`${pin.kind}:${pin.id}`}
            session={{ id: pin.id, title, preview: pin.id, active: false, unavailable: true }}
            onActivate={onSelectSession} actions={pinAction(pin, true)} />;
        })}
      </section> : null}
      {projects ? <ProjectNavigation
        projects={otherProjects}
        pending={projects.pending}
        pendingError={projects.pendingError}
        sessionsByProject={sessionsByProject}
        activeProjectId={projects.activeProjectId}
        pendingSessionId={pendingSessionId}
        memoryInstalled={projects.memoryInstalled}
        onSelectSession={onSelectSession}
        onPrefetchSession={onPrefetchSession}
        onNewProjectChat={projects.onNewChat}
        onCreateProject={projects.onCreate}
        onContinueProject={projects.onContinue}
        onStopProject={projects.onStop}
        onOpenCreateProject={projects.onOpenCreate}
        expandedProjects={navigationPins.expandedProjects}
        onToggleProject={navigationPins.toggleProject}
        projectActions={(project) => [directoryAction(project), ...pinAction({ kind: "project", id: project.id }, false)]}
        searching={searching}
        heading={pinnedProjects.size ? "其他项目" : "项目"}
      /> : null}

      <ConversationNavigation
        destinationHeading={false}
        sessionHeading="最近会话"
        sessionHeadingAction={<SessionSortMenu sort={sessionSort} onSort={setSessionSort} />}
        destinations={[]}
        actions={[]}
        sessions={recentSessions.map(sessionView)}
        sessionDrag={sessionDrag}
        onSessionDelete={deleteHandler}
        onSessionRename={onRenameSession ? (session, title) => onRenameSession(session.id, title) : undefined}
        sessionActions={(session) => {
          const row = allSessions.find((item) => item.id === session.id);
          return row && !row.projectId && !row.projectScoped
            ? pinAction({ kind: "session", id: row.id }, false) : [];
        }}
        onSessionActivate={onSelectSession}
        onSessionPrefetch={onPrefetchSession}
        pendingSessionId={pendingSessionId}
        sessionAfterContent={surface === "chat" && activeSessionId ? (
          <PluginUiSlot name="drawer.panel" sessionId={activeSessionId} />
        ) : undefined}
      />
      </div>
      {railActions.length || embeddedShell ? <div className="chat-sidebar__footer">
        {railActions.map((action) => (
          <button key={action.id} type="button" className="chat-sidebar__action"
            onClick={() => activateShellRailAction(action.id)}>
            <span className="chat-sidebar__action-icon" aria-hidden="true"
              dangerouslySetInnerHTML={{ __html: action.iconSvg }} />
            <span>{action.label}</span>
          </button>
        ))}
        {embeddedShell ? <button type="button" className="chat-sidebar__action"
          onClick={() => onCycleTheme()}
          title={`切换主题，当前为${themeLabel}`}
          aria-label={`切换主题，当前为${themeLabel}`}>
          <SunMoon size={18} aria-hidden="true" />
          <span>主题 · {themeLabel}</span>
        </button> : null}
      </div> : null}
      {rail ? <SidebarRailResizer rail={rail} sidebarRef={sidebarRef} /> : null}
      {directoryProject && projects ? <ProjectDirectoryDialog project={directoryProject}
        onClose={() => setDirectoryProjectId("")} onChoose={(path) => projects.onBindDirectory(directoryProject.id, path)}
        onCloseFocus={() => Array.from(sidebarRef.current?.querySelectorAll<HTMLElement>("[data-project-id]") ?? [])
          .find((row) => row.dataset.projectId === directoryProject.id)?.querySelector<HTMLButtonElement>(".navigation-menu-trigger")?.focus()} /> : null}
    </aside>
  );
});

/** "最近会话"分组标题行的排序入口；选中项持久化，菜单复用行菜单的浮层样式。 */
function SessionSortMenu({ sort, onSort }: { sort: SessionSortId; onSort: (sort: SessionSortId) => void }) {
  const current = SESSION_SORT_CHOICES.find((choice) => choice.id === sort) ?? SESSION_SORT_CHOICES[0];
  return <DropdownMenu modal={false}>
    <DropdownMenuTrigger asChild>
      <button type="button" className="project-navigation__icon conversation-sort-trigger"
        aria-label={`会话排序：${current.label}`} title="会话排序">
        <ArrowUpDown size={15} aria-hidden="true" />
      </button>
    </DropdownMenuTrigger>
    <DropdownMenuContent className="navigation-row-menu" align="end" sideOffset={4} collisionPadding={12} aria-label="会话排序">
      <DropdownMenuRadioGroup value={sort} aria-label="会话排序">
        {SESSION_SORT_CHOICES.map((choice) => <DropdownMenuRadioItem key={choice.id} value={choice.id}
          className="session-sort-menu__choice" onSelect={() => onSort(choice.id)}>
          <span>{choice.label}</span>
        </DropdownMenuRadioItem>)}
      </DropdownMenuRadioGroup>
    </DropdownMenuContent>
  </DropdownMenu>;
}

/** 侧栏右缘拖拽柄：只写 --chat-rail-width；双击复位，方向键以 0.5rem 步进。 */
function SidebarRailResizer({ rail, sidebarRef }: {
  rail: SidebarRailControl;
  sidebarRef: RefObject<HTMLElement | null>;
}) {
  const [dragging, setDragging] = useState(false);
  const [nowRem, setNowRem] = useState(SIDEBAR_RAIL_MIN_REM);
  useLayoutEffect(() => {
    const aside = sidebarRef.current;
    if (!aside) return;
    const rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    setNowRem(aside.getBoundingClientRect().width / rootPx);
  }, [rail.widthRem, sidebarRef]);
  const widthPx = () => sidebarRef.current?.getBoundingClientRect().width ?? 0;
  return <div
    className="chat-sidebar__resizer"
    role="separator"
    aria-orientation="vertical"
    aria-label="侧栏宽度"
    aria-valuemin={SIDEBAR_RAIL_MIN_REM}
    aria-valuemax={SIDEBAR_RAIL_MAX_REM}
    aria-valuenow={Math.round(nowRem * 2) / 2}
    tabIndex={0}
    data-dragging={dragging || undefined}
    onPointerDown={(event) => {
      if (!event.isPrimary || event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      rail.dragStart(event.clientX, widthPx());
      setDragging(true);
    }}
    onPointerMove={(event) => {
      if (dragging) rail.dragTo(event.clientX);
    }}
    onPointerUp={(event) => {
      if (!dragging) return;
      event.currentTarget.releasePointerCapture(event.pointerId);
      setDragging(false);
    }}
    onPointerCancel={() => setDragging(false)}
    onDoubleClick={(event) => {
      event.preventDefault();
      rail.reset();
    }}
    onKeyDown={(event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      rail.stepBy(event.key === "ArrowRight" ? SIDEBAR_RAIL_STEP_REM : -SIDEBAR_RAIL_STEP_REM, widthPx());
    }}
  />;
}
