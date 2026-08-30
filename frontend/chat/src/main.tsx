import { Suspense } from "react";
import { createRoot } from "react-dom/client";
import { initializeTheme, setTheme, startCrossPortThemeSync } from "../../theme/src/theme-runtime";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DesktopChatApp } from "./desktop-chat-app";
import { WebUiErrorBoundary } from "./webui-error-boundary";

export type { AgentBlock, ChatMessage, MessageAttachment, ThinkingBlock, ToolBlock } from "./chat-message";

const entryParams = new URLSearchParams(window.location.search);
const embeddedShell = entryParams.get("embedded") === "1";
const embeddedRuntime = embeddedShell && entryParams.get("surface") === "runtime";
if (embeddedShell) document.documentElement.dataset.embeddedShell = "true";
initializeTheme();
startCrossPortThemeSync();
if (embeddedShell) {
  const parentOrigins = new Set([
    window.location.origin,
    `${window.location.protocol}//${window.location.hostname}:5173`,
  ]);
  window.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (!parentOrigins.has(event.origin) || typeof event.data !== "object" || event.data === null) return;
    const message = event.data as Record<string, unknown>;
    if (message.type !== "akashic.theme" || typeof message.themeId !== "string") return;
    setTheme(message.themeId, false);
  });
}

function rootContent() {
  return <DesktopChatApp embeddedShell={embeddedShell} embeddedRuntime={embeddedRuntime} />;
}

createRoot(document.getElementById("root")!).render(
  <WebUiErrorBoundary>
    <TooltipProvider>
      <Suspense fallback={<div className="webui-entry-loading">正在载入界面…</div>}>
        {rootContent()}
      </Suspense>
    </TooltipProvider>
  </WebUiErrorBoundary>,
);
