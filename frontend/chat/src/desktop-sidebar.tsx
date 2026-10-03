import {
  Check,
  MessageSquarePlus,
  Search,
  Pin,
  PinOff,
} from "lucide-react";
import { memo, useMemo, useRef, useState } from "react";
import { ConversationNavigation, ConversationSessionRow, type ConversationSession } from "./conversation-navigation";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { ProjectNavigation, ProjectNavigationRow, type ProjectSessionItem } from "./project-navigation";
import { formatNavigationTime, sessionLabel } from "./web-chat-message-data";
import type { NavigationPin, NavigationPinsState } from "./use-navigation-pins";
import type { NavigationRowAction } from "./navigation-row-menu";
import type { PendingProjectRow, ProjectMemory, ProjectRow } from "./web-projects";
import { ProjectDirectoryDialog } from "./project-directory-dialog";
import { Folder } from "lucide-react";

export interface DesktopSidebarSession extends Omit<ConversationSession, "active" | "state"> {
  active: boolean;
  projectId?: string;
  projectScoped?: boolean;
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
}: DesktopSidebarProps) {
  const [query, setQuery] = useState("");
  const sidebarRef = useRef<HTMLElement>(null);
  const [directoryProjectId, setDirectoryProjectId] = useState("");
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
  // 目录未解析/归档不改变 Session.scope；回退到最近会话也不获得置顶资格。
  const knownProjects = useMemo(() => new Map(projects?.items.map((project) => [project.id, project])), [projects?.items]);
  const sessionsByProject = useMemo(() => {
    const groups = new Map<string, ProjectSessionItem[]>();
    for (const session of filteredSessions) {
      if (!session.projectId || !knownProjects.has(session.projectId)) continue;
      const group = groups.get(session.projectId) ?? [];
      group.push({
        id: session.id, title: session.title, updatedLabel: session.updatedLabel,
        active: surface === "chat" && session.active,
      });
      groups.set(session.projectId, group);
    }
    return groups;
  }, [filteredSessions, knownProjects, surface]);
  const pinnedProjects = new Set(navigationPins.pins.filter((pin) => pin.kind === "project").map((pin) => pin.id));
  const pinnedSessions = new Set(navigationPins.pins.filter((pin) => pin.kind === "session").map((pin) => pin.id));
  const recentSessions = filteredSessions.filter((session) => !pinnedSessions.has(session.id)
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
      {directoryProject && projects ? <ProjectDirectoryDialog project={directoryProject}
        onClose={() => setDirectoryProjectId("")} onChoose={(path) => projects.onBindDirectory(directoryProject.id, path)}
        onCloseFocus={() => Array.from(sidebarRef.current?.querySelectorAll<HTMLElement>("[data-project-id]") ?? [])
          .find((row) => row.dataset.projectId === directoryProject.id)?.querySelector<HTMLButtonElement>(".navigation-menu-trigger")?.focus()} /> : null}
    </aside>
  );
});
