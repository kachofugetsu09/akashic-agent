import {
  BookOpenText,
  Check,
  MessageSquarePlus,
  Palette,
  Puzzle,
  Search,
  SlidersHorizontal,
  Smartphone,
} from "lucide-react";
import { memo, useMemo, useState } from "react";
import { akashicBrandIcon } from "./akashic-brand";
import { ConversationNavigation, type ConversationSession } from "./conversation-navigation";
import { MaterialIconButton } from "../../theme/src/material-react";
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

/** Render desktop navigation from controlled data and semantic activation callbacks. */
export const DesktopSidebar = memo(function DesktopSidebar({
  embeddedShell,
  surface,
  sessions,
  activeSessionId,
  pendingSessionId,
  chatReady,
  themeLabel,
  onSelectSession,
  onOpenRuntime,
  onCycleTheme,
  onOpenPairing,
  onNewChat,
}: DesktopSidebarProps) {
  const dashboardHref = chatReady ? "/" : undefined;
  const [query, setQuery] = useState("");
  const filteredSessions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return sessions;
    return sessions.filter((session) => `${session.title} ${session.preview}`.toLowerCase().includes(needle));
  }, [query, sessions]);

  return (
    <aside className="chat-sidebar chat-sidebar--entry">
      {!embeddedShell ? (
        <header className="chat-sidebar-brand">
          <span
            className="chat-sidebar-brand__mark"
            style={{ WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` }}
            aria-hidden="true"
          />
          <span><strong>Akashic</strong><small>Chat</small></span>
        </header>
      ) : null}
      <div className="chat-sidebar__new">
        <MaterialIconButton variant="tonal" label="新会话" onClick={onNewChat}>
          <MessageSquarePlus size={18} aria-hidden="true" />
        </MaterialIconButton>
        <button type="button" className="chat-sidebar__new-label" onClick={onNewChat}>新会话</button>
      </div>
      <label className="chat-sidebar__search">
        <Search size={14} aria-hidden="true" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索会话"
          aria-label="搜索会话"
        />
      </label>

      <ConversationNavigation
        destinationHeading={false}
        sessionHeading={undefined}
        destinations={embeddedShell ? [] : [
          {
            id: "models",
            icon: <SlidersHorizontal size={20} />,
            label: "模型与认证",
            description: "Provider · API Key · 推理强度",
            href: "/settings",
          },
          {
            id: "runtime",
            icon: <BookOpenText size={20} />,
            label: "知识与运行",
            description: "记忆 · MCP · 定时任务",
            active: surface === "runtime",
            onActivate: onOpenRuntime,
          },
          {
            id: "plugins",
            icon: <Puzzle size={20} />,
            label: "插件",
            description: "打开 Dashboard 插件工作台",
            href: dashboardHref,
            disabled: dashboardHref === undefined,
          },
        ]}
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
          ...(embeddedShell ? [] : [{
            id: "theme",
            icon: <Palette size={18} />,
            label: `主题 · ${themeLabel}`,
            onActivate: onCycleTheme,
          }]),
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
