import { useMemo, useSyncExternalStore } from "react";

/** 插件随下一条消息附带的片段；kind 由声明它的后端 owner 校验。 */
export interface ComposerInputPart {
  kind: string;
  value: unknown;
}

interface Contribution {
  parts: readonly ComposerInputPart[];
  block: string | null;
}

// 输入框工具位里每个挂载各占一份：卸载即撤回，没有插件时发送不受影响。
type Owner = object;
let entries: ReadonlyMap<Owner, Contribution> = new Map();
let busy = false;
const listeners = new Set<() => void>();
const sentListeners = new Map<Owner, Set<() => void>>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };

function update(owner: Owner, next: Partial<Contribution> | null) {
  const copy = new Map(entries);
  if (next) copy.set(owner, { parts: [], block: null, ...copy.get(owner), ...next });
  else copy.delete(owner);
  entries = copy;
  listeners.forEach((listener) => listener());
}

/** 给 composer.toolbar 每次挂载一份宿主能力；release 后句柄整体失效，晚到的回调不能重新登记片段或阻止发送。 */
export function composerControls() {
  const owner: Owner = {};
  let live = true;
  const controls = {
    busy: () => busy,
    contribute(parts: readonly ComposerInputPart[]) {
      if (!Array.isArray(parts) || parts.some((part) => !part || typeof part.kind !== "string" || !part.kind)) {
        throw new Error("输入片段格式错误");
      }
      if (live) update(owner, { parts: parts.map((part) => ({ kind: part.kind, value: part.value })) });
    },
    block(reason: string | null) {
      if (reason !== null && typeof reason !== "string") throw new Error("阻止发送的原因必须是字符串");
      if (live) update(owner, { block: reason || null });
    },
    onSent(callback: () => void): () => void {
      if (!live) return () => {};
      const set = sentListeners.get(owner) ?? new Set();
      sentListeners.set(owner, set);
      set.add(callback);
      return () => { set.delete(callback); };
    },
  };
  const release = () => {
    live = false;
    sentListeners.delete(owner);
    if (entries.has(owner)) update(owner, null);
  };
  return { controls, release };
}

/** 生成中由宿主设置；插件经 busy() 读取，并在 onInvalidate 时重新判断。 */
export function setComposerBusy(value: boolean) {
  busy = value;
}

function snapshot(current: ReadonlyMap<Owner, Contribution>) {
  const values = [...current.values()];
  return { parts: values.flatMap((value) => value.parts), blockReason: values.find((value) => value.block)?.block ?? "" };
}

/** 当前要附带的片段与第一条阻止原因。 */
export function useComposerContributions(): { parts: readonly ComposerInputPart[]; blockReason: string } {
  const current = useSyncExternalStore(subscribe, () => entries);
  return useMemo(() => snapshot(current), [current]);
}

export function composerContributions(): { parts: readonly ComposerInputPart[]; blockReason: string } {
  return snapshot(entries);
}

/** 服务端确认收到后通知各挂载，插件据此清掉“这条要附带”的状态。 */
export function notifyComposerSent() {
  for (const set of sentListeners.values()) {
    for (const callback of set) {
      try { callback(); }
      catch (error) { console.error("[plugin-ui] composer onSent failed", error); }
    }
  }
}
