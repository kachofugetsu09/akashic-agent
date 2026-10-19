// 聊天端与宿主工具区的意图、引用。只在有宿主时工作；夹具页面没有接受者，调用方回到原有行为。
import { useSyncExternalStore } from "react";

import { hostLink } from "./chat-host";

export type WorkbenchIntentStatus = "handled" | "unhandled" | "invalid" | "failed";

export interface WorkbenchQuote {
  session: string;
  text: string;
  source: string;
}

let accepted: ReadonlySet<string> = new Set();
const listeners = new Set<() => void>();
let quoteHandler: ((quote: WorkbenchQuote) => void) | null = null;
let currentSession = "";
/** 打开会话工作目录里的一个文件：{ path, line? }，path 相对工作目录；由声明接受它的工具决定怎么打开（如“改动”定位到该文件）。 */
export const FILE_INTENT = "workspace.file.open.v1";

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

/** 当前会话随 reportShellSession 更新；回复里的文件链接用它投递意图。 */
export function setWorkbenchSession(sessionId: string): void {
  currentSession = sessionId;
}

// 像工作目录里文件的行内代码：带常见扩展名（或 Makefile 这类固定文件名）的相对路径，可带 :行号 或 :行:列；
// 网址、绝对路径、带 .. 的、带空格的、order.currency 这类属性访问、Node.js 这类框架名都不算。
const FILE_PATH = /^(?:\.\/)?(?:(?!\.\.\/)[\w@.-]+\/)*(?:[\w@-][\w@.-]*\.(?:py|pyi|ts|tsx|js|jsx|mjs|cjs|json|md|mdx|toml|ya?ml|css|scss|html|vue|svelte|rs|go|java|kt|swift|c|h|cc|cpp|hpp|cs|rb|php|lua|dart|scala|proto|sh|sql|txt|ini|cfg|xml|lock)|Makefile|Dockerfile|LICENSE)(?::\d+){0,2}$/;
const FRAMEWORK_NAME = /^[A-Z][a-z]+\.js$/;

export interface WorkspaceFileRef { path: string; line: number | null }

/** 行内代码若像工作目录里的文件，返回去掉 ./ 的相对路径与可选行号；否则 null。 */
export function workspaceFileRef(code: string): WorkspaceFileRef | null {
  if (code.length > 300 || !FILE_PATH.test(code) || FRAMEWORK_NAME.test(code)) return null;
  const [, path, line] = /^(?:\.\/)?(.+?)(?::(\d+))?(?::\d+)?$/.exec(code)!;
  return { path, line: line ? Number(line) : null };
}

/** 只在用户点击时调用，同 openInWorkbench。 */
export function openFileInWorkbench(file: WorkspaceFileRef): Promise<WorkbenchIntentStatus> {
  return openInWorkbench(FILE_INTENT, currentSession, file.line === null ? { path: file.path } : file);
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
 * 没有宿主或没有接受者时立即返回 unhandled。
 */
export function openInWorkbench(name: string, session: string, payload: unknown): Promise<WorkbenchIntentStatus> {
  const link = hostLink();
  if (!link || !accepted.has(name)) return Promise.resolve("unhandled");
  if (!session) return Promise.resolve("invalid");
  return link.requestIntent(name, session, payload);
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
