import { useSyncExternalStore } from "react";

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

/** 宿主 Shell 经 postMessage 下发的底栏动作投影；standalone 模式下永远为空。 */
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

/** 激活回宿主域：函数不跨 realm,chat 只回传动作 id。 */
export function activateShellRailAction(id: string): void {
  window.parent.postMessage({ type: "akashic.rail-action-activate", id }, window.location.origin);
}

/** iframe 内的遮罩盖不住宿主 chrome：模态层开合时告知宿主，让它收起浮在页面上的控件。standalone 下无宿主，静默。 */
export function reportShellOverlay(open: boolean): void {
  if (window.parent === window) return;
  window.parent.postMessage({ type: "akashic.chat-overlay", open }, window.location.origin);
}
