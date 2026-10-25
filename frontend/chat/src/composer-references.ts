import { useMemo, useSyncExternalStore } from "react";

/** 一条待发送的会话引用：只记身份与标题，正文由模型用 read_session 去读。 */
export interface ComposerReference {
  sessionId: string;
  title: string;
}

/** 拖拽时写入的自定义 MIME：与侧栏内部排序用的 text/plain 分开，互不干扰。 */
export const SESSION_DRAG_TYPE = "application/x-akashic-session";
/** 一次最多引用几个会话；与后端 session_reference.MAX_REFERENCES 一致。 */
export const MAX_REFERENCES = 4;

interface State {
  references: readonly ComposerReference[];
}

// 引用属于「这个编辑位置」，所以按 draftKey 分开存：新会话与各会话的草稿互不串。
let byDraft: ReadonlyMap<string, readonly ComposerReference[]> = new Map();
const listeners = new Set<() => void>();
const sentListeners = new Map<string, Set<() => void>>();

function emit() {
  listeners.forEach((listener) => listener());
}

function write(draftKey: string, next: readonly ComposerReference[]) {
  const copy = new Map(byDraft);
  if (next.length === 0) copy.delete(draftKey);
  else copy.set(draftKey, next);
  byDraft = copy;
  emit();
}

/** 加入一条引用；同一会话重复拖入只保留一条，超过上限直接忽略。 */
export function addReference(draftKey: string, reference: ComposerReference): boolean {
  const current = byDraft.get(draftKey) ?? [];
  if (current.some((item) => item.sessionId === reference.sessionId)) return false;
  if (current.length >= MAX_REFERENCES) return false;
  write(draftKey, [...current, reference]);
  return true;
}

export function removeReference(draftKey: string, sessionId: string) {
  const current = byDraft.get(draftKey) ?? [];
  write(draftKey, current.filter((item) => item.sessionId !== sessionId));
}

export function clearReferences(draftKey: string) {
  write(draftKey, []);
}

/** 取当前草稿的引用（非 React 上下文，用于发送时组装 parts）。 */
export function composerReferences(draftKey: string): readonly ComposerReference[] {
  return byDraft.get(draftKey) ?? [];
}

/** 服务端确认收到后清空并通知；与 composer-contributions 的 onSent 同义。 */
export function notifyReferencesSent(draftKey: string) {
  clearReferences(draftKey);
  for (const callback of sentListeners.get(draftKey) ?? []) {
    try { callback(); }
    catch (error) { console.error("[chat-ui] reference onSent failed", error); }
  }
}

export function onReferencesSent(draftKey: string, callback: () => void): () => void {
  const set = sentListeners.get(draftKey) ?? new Set();
  sentListeners.set(draftKey, set);
  set.add(callback);
  return () => { set.delete(callback); };
}

/** 解析拖拽载荷；不是我们的类型就返回 null，让别的放置区处理。 */
export function readSessionDrag(dataTransfer: DataTransfer | null): ComposerReference | null {
  if (!dataTransfer) return null;
  const raw = dataTransfer.getData(SESSION_DRAG_TYPE);
  if (!raw) return null;
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object") return null;
    const record = value as Record<string, unknown>;
    if (typeof record.sessionId !== "string" || !record.sessionId) return null;
    return {
      sessionId: record.sessionId,
      title: typeof record.title === "string" ? record.title : record.sessionId,
    };
  } catch {
    return null;
  }
}

export function useComposerReferences(draftKey: string): readonly ComposerReference[] {
  const current = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
    () => byDraft,
  );
  return useMemo(() => current.get(draftKey) ?? [], [current, draftKey]);
}

/** 组装要随消息提交的输入片段；没有引用时不产生任何 part。 */
export function referenceParts(draftKey: string): { kind: string; value: unknown }[] {
  const references = composerReferences(draftKey);
  if (references.length === 0) return [];
  return [{
    kind: "session.reference",
    value: { sessions: references.map((item) => ({ session_id: item.sessionId, title: item.title })) },
  }];
}

export type { State };
