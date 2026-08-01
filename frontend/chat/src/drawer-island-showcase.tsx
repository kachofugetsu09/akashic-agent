import {
  Check,
  FileText,
  LibraryBig,
  MessageSquarePlus,
  Puzzle,
  RefreshCw,
  RotateCcw,
  Settings,
  X,
} from "lucide-react";
import { ConversationNavigation } from "./conversation-navigation";
import "./drawer-island-showcase.css";

const SESSIONS = [
  {
    id: "heart-rate",
    title: "akashic 帮我看看现在心率怎么样",
    preview: "好像手坠到了肩膀那块…",
    updatedLabel: "昨天",
    active: true,
  },
  {
    id: "mobile-link",
    title: "请只回复：手机端链路验证通过",
    preview: "可以看到吗 Akashic",
    updatedLabel: "7/15",
    active: false,
  },
  {
    id: "drawer-study",
    title: "恢复统一前的移动端抽屉",
    preview: "复刻 v0.8.15 的导航层级",
    updatedLabel: "现在",
    active: false,
  },
];

/** Render the real shared navigation component with the final pre-unification mobile hierarchy. */
export function DrawerIslandShowcase() {
  return (
    <main className="legacy-drawer-showcase">
      <header className="legacy-drawer-showcase__header">
        <h1>统一前的移动端抽屉</h1>
        <p>取自 akashic-mobile v0.8.15（315b4ba）。共享组件保留新能力，视觉层级恢复旧版。</p>
      </header>

      <section className="legacy-drawer-device" aria-label="v0.8.15 抽屉复刻">
        <div className="legacy-drawer-underlay" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div className="legacy-drawer-scrim" aria-hidden="true" />
        <ConversationNavigation
          className="legacy-drawer"
          dialog
          closeAction={(
            <button className="legacy-drawer__close" type="button" aria-label="关闭会话抽屉">
              <X size={24} />
            </button>
          )}
          destinations={[
            {
              id: "runtime",
              icon: <LibraryBig size={21} />,
              label: "知识与运行",
              description: "记忆 · MCP · 定时任务",
              featured: true,
              onActivate: () => undefined,
            },
            {
              id: "plugins",
              icon: <Puzzle size={20} />,
              label: "插件",
              badge: 5,
              onActivate: () => undefined,
            },
          ]}
          sessions={SESSIONS.map((session) => ({
            ...session,
            state: session.active ? <Check size={18} /> : null,
            onActivate: () => undefined,
          }))}
          sessionAfterContent={(
            <button className="legacy-memory-summary" type="button" data-capability-id="memory">
              <span>记忆整理</span>
              <strong>10 条待整理</strong>
            </button>
          )}
          actions={[
            { id: "settings", icon: <Settings size={18} />, label: "设置", onActivate: () => undefined },
            { id: "diagnostics", icon: <FileText size={18} />, label: "导出诊断报告", onActivate: () => undefined },
            { id: "resync", icon: <RotateCcw size={18} />, label: "清理缓存并同步", onActivate: () => undefined },
            { id: "pairing", icon: <RefreshCw size={18} />, label: "重新扫码", onActivate: () => undefined },
            { id: "new-chat", icon: <MessageSquarePlus size={18} />, label: "新聊天", primary: true, onActivate: () => undefined },
          ]}
        />
      </section>
    </main>
  );
}
