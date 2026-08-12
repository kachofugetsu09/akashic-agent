import {
  BookOpenText,
  Check,
  MessageSquarePlus,
  Palette,
  Puzzle,
  SlidersHorizontal,
  Smartphone,
} from "lucide-react";
import { memo } from "react";
import { akashicBrandIcon } from "./akashic-brand";
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
  return (
    <aside className="chat-sidebar">
      {!embeddedShell ? (
        <header className="chat-sidebar-brand">
          <span
            className="chat-sidebar-brand__mark"
            style={{ WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` }}
            aria-hidden="true"
          />
          <span><strong>Akashic</strong><small>Dashboard</small></span>
        </header>
      ) : null}
      <ConversationNavigation
        destinationHeading={embeddedShell ? undefined : "工作空间"}
        sessionHeading="最近会话"
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
        sessions={sessions.map((session) => ({
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
          {
            id: "new-chat",
            icon: <MessageSquarePlus size={18} />,
            label: "新聊天",
            primary: true,
            onActivate: onNewChat,
          },
        ]}
      />
    </aside>
  );
});
