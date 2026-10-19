// iframe 宿主的消息适配：把父窗口消息转成 ChatHandle 调用，把 ChatHostLink 调用转成消息。外壳改为同文档挂载后整个删除。
import { setTheme } from "../../theme/src/theme-runtime";
import { createUuid } from "./browser-uuid.ts";
import type { ChatHandle, ChatHostLink } from "./chat-host";
import type { WorkbenchIntentStatus } from "./shell-workbench";

const RESULT_TIMEOUT_MS = 12_000;
const STATUSES: readonly WorkbenchIntentStatus[] = ["handled", "unhandled", "invalid", "failed"];
const pending = new Map<string, (status: WorkbenchIntentStatus) => void>();

function post(message: Record<string, unknown>): void {
  window.parent.postMessage(message, window.location.origin);
}

export const frameLink: ChatHostLink = {
  activateRailAction: (id) => post({ type: "akashic.rail-action-activate", id }),
  reportSession: (sessionId) => post({ type: "akashic.chat-session", sessionId }),
  reportOverlay: (open) => post({ type: "akashic.chat-overlay", open }),
  // 超时按 failed；回执经 akashic.workbench-intent-result 回来。
  requestIntent(name, session, payload) {
    const id = createUuid();
    return new Promise((resolve) => {
      const timer = window.setTimeout(() => settle("failed"), RESULT_TIMEOUT_MS);
      function settle(status: WorkbenchIntentStatus) {
        window.clearTimeout(timer);
        pending.delete(id);
        resolve(status);
      }
      pending.set(id, settle);
      post({ type: "akashic.workbench-intent", id, name, session, payload });
    });
  },
};

function settleIntent(message: Record<string, unknown>): void {
  if (typeof message.id !== "string" || !STATUSES.includes(message.status as WorkbenchIntentStatus)) return;
  pending.get(message.id)?.(message.status as WorkbenchIntentStatus);
}

// 底栏动作、工具区状态与主题来自同源父页面（含本地开发端口）；意图回执与引用只认直接父窗口。
export function listenFrameHost(handle: ChatHandle): void {
  const parentOrigins = new Set([window.location.origin, `${window.location.protocol}//${window.location.hostname}:5173`]);
  window.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (!parentOrigins.has(event.origin) || typeof event.data !== "object" || event.data === null) return;
    const message = event.data as Record<string, unknown>;
    const direct = event.source === window.parent;
    switch (message.type) {
      case "akashic.rail-actions": handle.setRailActions(message.actions); break;
      case "akashic.chat-tools": handle.setTools(message.present === true, message.shortcuts); break;
      case "akashic.theme": if (typeof message.themeId === "string") setTheme(message.themeId, false); break;
      case "akashic.workbench-intents": if (direct) handle.setWorkbenchIntents(message.names); break;
      case "akashic.workbench-intent-result": if (direct) settleIntent(message); break;
      case "akashic.composer-quote": if (direct) handle.quote(message); break;
      case "akashic.frame.shown": if (direct && event.origin === window.location.origin) handle.shown(); break;
    }
  });
}
