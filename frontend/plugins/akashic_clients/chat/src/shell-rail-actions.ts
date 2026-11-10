import { useSyncExternalStore } from "react";

import { hostLink } from "./chat-host";
import { setWorkbenchSession } from "./shell-workbench";

export interface ShellRailActionItem {
  id: string;
  order?: number;
  label: string;
  iconSvg: string;
}

let current: readonly ShellRailActionItem[] = [];
const listeners = new Set<() => void>();

function isRailAction(entry: unknown): entry is ShellRailActionItem {
  if (typeof entry !== "object" || entry === null) return false;
  const candidate = entry as Record<string, unknown>;
  return typeof candidate.id === "string"
    && typeof candidate.label === "string"
    && typeof candidate.iconSvg === "string"
    && candidate.iconSvg.startsWith("<svg");
}

/** 宿主下发的底栏动作投影；夹具页面没有宿主，永远为空。 */
export function setShellRailActions(value: unknown): void {
  if (!Array.isArray(value)) return;
  current = value.filter(isRailAction);
  for (const listener of listeners) listener();
}

export function useShellRailActions(): readonly ShellRailActionItem[] {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => current,
  );
}

/** 激活交回宿主：动作的处理函数留在宿主，聊天只回传动作 id。 */
export function activateShellRailAction(id: string): void {
  hostLink()?.activateRailAction(id);
}

/** 当前会话交给宿主转交工具区标签；新建对话尚未发送时为 null。夹具页面没有宿主，静默。 */
export function reportShellSession(sessionId: string): void {
  setWorkbenchSession(sessionId);
  hostLink()?.reportSession(sessionId || null);
}

/** 模态层开合时告知宿主，让它收起浮在页面上的控件。夹具页面没有宿主，静默。 */
export function reportShellOverlay(open: boolean): void {
  hostLink()?.reportOverlay(open);
}
