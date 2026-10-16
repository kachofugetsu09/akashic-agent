// 聊天端与宿主工具区的意图、引用桥。只在嵌入宿主时工作；standalone 下没有接受者，调用方回到原有行为。
import { useSyncExternalStore } from "react";

import { createUuid } from "./browser-uuid.ts";

export type WorkbenchIntentStatus = "handled" | "unhandled" | "invalid" | "failed";

export interface WorkbenchQuote {
  session: string;
  text: string;
  source: string;
}

const RESULT_TIMEOUT_MS = 12_000;
const STATUSES: readonly WorkbenchIntentStatus[] = ["handled", "unhandled", "invalid", "failed"];
let accepted: ReadonlySet<string> = new Set();
const listeners = new Set<() => void>();
const pending = new Map<string, (status: WorkbenchIntentStatus) => void>();
let quoteHandler: ((quote: WorkbenchQuote) => void) | null = null;

/** 宿主下发的可接受意图名集合；只用来决定是否显示“在侧栏打开”。 */
export function setWorkbenchIntents(value: unknown): void {
  if (!Array.isArray(value)) return;
  const next = new Set(value.filter((name): name is string => typeof name === "string"));
  if (next.size === accepted.size && [...next].every((name) => accepted.has(name))) return;
  accepted = next;
  for (const listener of [...listeners]) listener();
}

/** 能力集合变化的订阅：宿主的首次广播可能晚于插槽挂载，已挂载的读者靠它刷新。 */
export function subscribeWorkbenchIntents(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function useWorkbenchAccepts(name: string): boolean {
  return useSyncExternalStore(subscribeWorkbenchIntents, () => accepted.has(name));
}

export function workbenchAccepts(name: string): boolean {
  return accepted.has(name);
}

export interface HostShortcut {
  label: string;
  keys: readonly string[];
}

let hostShortcuts: readonly HostShortcut[] = [];

/** 宿主工具区的快捷键（宿主自己处理），只用于快捷键面板展示；不合规的条目丢掉。 */
export function setHostShortcuts(value: unknown): void {
  if (!Array.isArray(value)) return;
  hostShortcuts = value.slice(0, 16).filter((item): item is HostShortcut => Boolean(item) && typeof item === "object"
    && typeof item.label === "string" && item.label.length <= 40 && Array.isArray(item.keys) && item.keys.length <= 4
    && item.keys.every((key: unknown) => typeof key === "string" && key.length <= 8));
}

export function getHostShortcuts(): readonly HostShortcut[] {
  return hostShortcuts;
}

/**
 * 只在用户点击或键盘激活的处理函数里同步调用；Agent 侧事件不走这里，这是“只有用户能打开工具区”的保证。
 * 没有宿主或没有接受者时立即返回 unhandled，超时按 failed。
 */
export function openInWorkbench(name: string, session: string, payload: unknown): Promise<WorkbenchIntentStatus> {
  if (window.parent === window || !accepted.has(name)) return Promise.resolve("unhandled");
  if (!session) return Promise.resolve("invalid");
  const id = createUuid();
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => settle("failed"), RESULT_TIMEOUT_MS);
    function settle(status: WorkbenchIntentStatus) {
      window.clearTimeout(timer);
      pending.delete(id);
      resolve(status);
    }
    pending.set(id, settle);
    window.parent.postMessage({ type: "akashic.workbench-intent", id, name, session, payload }, window.location.origin);
  });
}

export function settleWorkbenchIntent(message: Record<string, unknown>): void {
  if (typeof message.id !== "string" || !STATUSES.includes(message.status as WorkbenchIntentStatus)) return;
  pending.get(message.id)?.(message.status as WorkbenchIntentStatus);
}

/** 输入框登记引用的落点；同一时刻只有当前对话视图一个。 */
export function onWorkbenchQuote(handler: (quote: WorkbenchQuote) => void): () => void {
  quoteHandler = handler;
  return () => { if (quoteHandler === handler) quoteHandler = null; };
}

export function receiveWorkbenchQuote(message: Record<string, unknown>): void {
  const { session, text, source } = message;
  if (typeof session !== "string" || typeof text !== "string" || typeof source !== "string") return;
  quoteHandler?.({ session, text, source });
}
