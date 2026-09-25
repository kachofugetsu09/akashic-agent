import {
  Check,
  MessageSquarePlus,
  Search,
  Smartphone,
} from "lucide-react";
import { memo, useMemo, useState } from "react";
import { ConversationNavigation, type ConversationSession } from "./conversation-navigation";
import { MobilePluginSlot } from "./mobile-plugin-runtime";
import { ProjectNavigation, type ProjectSessionItem } from "./project-navigation";
import type { PendingProjectRow, ProjectMemory, ProjectRow } from "./web-projects";

export interface DesktopSidebarSession extends Omit<ConversationSession, "active" | "state"> {
  active: boolean;
  projectId?: string;
}

export interface DesktopSidebarProjects {
  items: ProjectRow[];
  pending: PendingProjectRow[];
  pendingError: string;
  activeProjectId: string;
  memoryInstalled: boolean;
  onNewChat: (projectId: string) => void;
  onCreate: (name: string, memory: ProjectMemory) => Promise<void>;
  onContinue: (key: string) => Promise<void>;
  onStop: (key: string) => void;
  onOpenCreate?: () => void;
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
  onSelectSession: (sessionId: string) => void;
  onCycleTheme: () => void;
  onOpenPairing: () => void;
  onNewChat: () => void;
}

/** Session-only vertical rail — product destinations live on the L-shape top band. */
export const DesktopSidebar = memo(function DesktopSidebar({
  surface,
  sessions,
  activeSessionId,
  pendingSessionId,
  projects,
  onSelectSession,
  onOpenPairing,
  onNewChat,
}: DesktopSidebarProps) {
  const [query, setQuery] = useState("");
  const filteredSessions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return sessions;
    return sessions.filter((session) => `${session.title} ${session.preview}`.toLowerCase().includes(needle));
  }, [query, sessions]);
  // 1. 已知项目的对话进入项目分组；未知或已归档项目的对话仍留在最近会话里。
  const knownProjects = useMemo(() => new Set(projects?.items.map((project) => project.id)), [projects?.items]);
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
  const recentSessions = useMemo(
    () => filteredSessions.filter((session) => !session.projectId || !knownProjects.has(session.projectId)),
    [filteredSessions, knownProjects],
  );

  return (
    <aside className="chat-sidebar chat-sidebar--entry">
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

      {projects ? (
        <ProjectNavigation
          projects={projects.items}
          pending={projects.pending}
          pendingError={projects.pendingError}
          sessionsByProject={sessionsByProject}
          activeProjectId={projects.activeProjectId}
          pendingSessionId={pendingSessionId}
          memoryInstalled={projects.memoryInstalled}
          onSelectSession={onSelectSession}
          onNewProjectChat={projects.onNewChat}
          onCreateProject={projects.onCreate}
          onContinueProject={projects.onContinue}
          onStopProject={projects.onStop}
          onOpenCreateProject={projects.onOpenCreate}
        />
      ) : null}

      <ConversationNavigation
        destinationHeading={false}
        sessionHeading="最近会话"
        destinations={[]}
        sessions={recentSessions.map((session) => ({
          ...session,
          active: surface === "chat" && session.active,
          state: surface === "chat" && session.active ? <Check size={18} /> : null,
        }))}
        onSessionActivate={onSelectSession}
        pendingSessionId={pendingSessionId}
        sessionAfterContent={surface === "chat" && activeSessionId ? (
          <MobilePluginSlot name="drawer.panel" sessionId={activeSessionId} />
        ) : undefined}
        actions={[
          {
            id: "connect-mobile",
            icon: <Smartphone size={18} />,
            label: "连接手机",
            onActivate: onOpenPairing,
          },
        ]}
      />
    </aside>
  );
});
