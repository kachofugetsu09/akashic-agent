import {
  Check,
  MessageSquarePlus,
  Search,
  Smartphone,
} from "lucide-react";
import { memo, useMemo, useState } from "react";
import { ConversationNavigation, type ConversationSession } from "./conversation-navigation";
import { MobilePluginSlot } from "./mobile-plugin-runtime";

export interface DesktopSidebarSession extends Omit<ConversationSession, "active" | "state"> {
  active: boolean;
}

export interface DesktopSidebarProps {
  embeddedShell: boolean;
  surface: "chat" | "runtime";
  sessions: DesktopSidebarSession[];
  activeSessionId: string;
  pendingSessionId: string;
  chatReady: boolean;
  themeLabel: string;
  onSelectSession: (sessionId: string) => void;
  onOpenRuntime: () => void;
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

  return (
    <aside className="chat-sidebar chat-sidebar--entry">
      <div className="chat-sidebar__toolbar">
        <button type="button" className="chat-sidebar__new" onClick={onNewChat}>
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

      <ConversationNavigation
        destinationHeading={false}
        sessionHeading={undefined}
        destinations={[]}
        sessions={filteredSessions.map((session) => ({
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
