import {
  Check,
  MessageSquarePlus,
  Search,
  Pin,
  PinOff,
  ArrowUpDown,
} from "lucide-react";
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ConversationNavigation, ConversationSessionRow, type ConversationSession } from "./conversation-navigation";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { ProjectNavigation, ProjectNavigationRow, type ProjectSessionItem } from "./project-navigation";
import { formatNavigationTime, sessionLabel } from "./web-chat-message-data";
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
type SessionSortId = "activity" | "created" | "title";

const SESSION_SORT_CHOICES: readonly { id: SessionSortId; label: string }[] = [
  { id: "activity", label: "最近活动" },
  { id: "created", label: "最近创建" },
  { id: "title", label: "标题" },
];
const SESSION_SORT_KEY = "akashic.chat.session-sort";

function readSessionSort(): SessionSortId {
  try {
    const value = localStorage.getItem(SESSION_SORT_KEY);
    return SESSION_SORT_CHOICES.some((choice) => choice.id === value) ? value as SessionSortId : "activity";
  } catch {
    return "activity";
  }
}

/** 时间倒序；缺失时间戳的行（如置顶解析补齐的目录外会话）稳定排到末尾。 */
function compareSessionTime(a: string | undefined, b: string | undefined): number {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return Date.parse(b) - Date.parse(a);
}

function sortSessions(rows: DesktopSidebarSession[], sort: SessionSortId): DesktopSidebarSession[] {
  if (sort === "title") return [...rows].sort((a, b) => a.title.localeCompare(b.title, "zh-Hans-CN"));
  if (sort === "created") return [...rows].sort((a, b) => compareSessionTime(a.createdAt, b.createdAt));
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
  /** 宽屏侧栏的宽度控制；抽屉（compact）不带此 prop，因此不渲染拖拽柄。 */
  rail?: SidebarRailControl;
}

/** Session-only vertical rail — product destinations live on the L-shape top band. */
export const DesktopSidebar = memo(function DesktopSidebar({
  surface,
  sessions,
  activeSessionId,
  pendingSessionId,
  projects,
  navigationPins,
  onSelectSession,
  onPrefetchSession,
  onNewChat,
  rail,
}: DesktopSidebarProps) {
  const [query, setQuery] = useState("");
  const sidebarRef = useRef<HTMLElement>(null);
  const [directoryProjectId, setDirectoryProjectId] = useState("");
  const [sessionSort, setSessionSort] = useState(readSessionSort);
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
      updatedLabel: formatNavigationTime(session.updated_at), active: activeSessionId === session.key,
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
  const sortedSessions = useMemo(() => sortSessions(filteredSessions, sessionSort), [filteredSessions, sessionSort]);
  useEffect(() => {
    try { localStorage.setItem(SESSION_SORT_KEY, sessionSort); }
    catch { /* 禁用存储时仍允许本次浏览的排序选择。 */ }
  }, [sessionSort]);
  // 目录未解析/归档不改变 Session.scope；回退到最近会话也不获得置顶资格。
  const knownProjects = useMemo(() => new Map(projects?.items.map((project) => [project.id, project])), [projects?.items]);
  const sessionsByProject = useMemo(() => {
    const groups = new Map<string, ProjectSessionItem[]>();
    for (const session of sortedSessions) {
      if (!session.projectId || !knownProjects.has(session.projectId)) continue;
      const group = groups.get(session.projectId) ?? [];
      group.push({
        id: session.id, title: session.title, updatedLabel: session.updatedLabel,
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
    state: surface === "chat" && session.active ? <Check size={18} /> : null,
  });

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
