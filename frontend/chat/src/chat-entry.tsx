import { Suspense } from "react";
import { createRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { setKeyScope, type ChatHandle } from "./chat-host";
import { observeChatLayout } from "./chat-layout";
import { exposeChatRuntime } from "./chat-ui-runtime";
import { DesktopChatApp } from "./desktop-chat-app";
import { notifyShown } from "./plugin-ui-runtime";
import { setShellRailActions } from "./shell-rail-actions";
import { receiveWorkbenchQuote, setHostShortcuts, setWorkbenchIntents } from "./shell-workbench";
import { WebUiErrorBoundary } from "./webui-error-boundary";

// 两种入口共用的渲染：frame 是承载宿主状态属性的元素（iframe 里是 html，同文档挂载时是聊天根节点）。
export function renderChat(container: HTMLElement, frame: HTMLElement, embeddedShell: boolean): { handle: ChatHandle; unmount(): void } {
  exposeChatRuntime();
  const stopLayout = observeChatLayout(frame);
  const stopKeys = setKeyScope(frame === document.documentElement ? null : frame);
  const root = createRoot(container);
  root.render(
    <WebUiErrorBoundary>
      <TooltipProvider>
        <Suspense fallback={<div className="webui-entry-loading">正在载入界面…</div>}>
          <DesktopChatApp embeddedShell={embeddedShell} />
        </Suspense>
      </TooltipProvider>
    </WebUiErrorBoundary>,
  );
  const handle: ChatHandle = {
    setRailActions: setShellRailActions,
    setTools(present, shortcuts) {
      frame.dataset.shellTools = present ? "true" : "false";
      setHostShortcuts(shortcuts);
    },
    setWorkbenchIntents,
    quote(quote) {
      if (typeof quote === "object" && quote !== null) receiveWorkbenchQuote(quote as Record<string, unknown>);
    },
    shown: notifyShown,
  };
  return { handle, unmount: () => { root.unmount(); stopLayout(); stopKeys(); } };
}
