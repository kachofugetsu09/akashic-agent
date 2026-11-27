import { Component, Fragment, useCallback, useSyncExternalStore, type ReactNode } from "react";
import {
  chatKernel,
  type ChatKernel,
  type ChatServiceRecord,
  type ChatSlotEntry,
  type ChatSlotHandle,
} from "./chat-kernel";

const EMPTY_ENTRIES: readonly ChatSlotEntry<unknown>[] = Object.freeze([]);

/**
 * 真正拦截渲染回调同步抛错的局部错误边界：
 * 必须通过子组件 <InvokeRender /> 触发插件 render()，确保同步 throw 落在 ErrorBoundary 内部。
 */
export class SafePluginBoundary extends Component<
  { fallback?: ReactNode; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (this.props.fallback ?? null) : this.props.children;
  }
}

export function InvokePluginRender({ render }: { render: () => ReactNode }): ReactNode {
  return render();
}

/** 订阅指定命名服务的只读冻结快照；底层按写时快照保证 Object.is 稳定。 */
export function useChatServices<T>(serviceId: string, kernel: ChatKernel = chatKernel): readonly ChatServiceRecord<T>[] {
  const subscribe = useCallback(
    (listener: () => void) => kernel.subscribeService(serviceId, listener),
    [kernel, serviceId],
  );
  const getSnapshot = useCallback(() => kernel.listServices<T>(serviceId), [kernel, serviceId]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** 读取指定命名服务的首个可用实现。 */
export function useChatService<T>(serviceId: string, kernel: ChatKernel = chatKernel): T | null {
  const records = useChatServices<T>(serviceId, kernel);
  return records.length > 0 ? records[0].value : null;
}

/** 订阅某个槽声明句柄的只读冻结贡献快照；声明撤销后自动读空。 */
export function useChatSlotEntries<TProps = unknown>(
  handle: ChatSlotHandle | null | undefined,
): readonly ChatSlotEntry<TProps>[] {
  const subscribe = useCallback(
    (listener: () => void) => (handle ? handle.subscribe(listener) : () => {}),
    [handle],
  );
  const getSnapshot = useCallback(
    () => (handle ? handle.entries<TProps>() : (EMPTY_ENTRIES as readonly ChatSlotEntry<TProps>[])),
    [handle],
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/**
 * 槽声明者渲染全部贡献条目的受保护出口：
 * - 接受声明返回的不透明 handle（声明即渲染权）；
 * - 每个条目在独立 SafePluginBoundary 内调用 render(props)，任一贡献崩溃仅回退自身，不影响兄弟或父级。
 */
export function ChatSlotOutlet<TProps = unknown>({
  handle,
  props,
  fallback,
  filter,
  entryFallback,
}: {
  handle: ChatSlotHandle | null | undefined;
  props: TProps;
  fallback?: ReactNode;
  filter?: (entry: ChatSlotEntry<TProps>) => boolean;
  entryFallback?: (entry: ChatSlotEntry<TProps>) => ReactNode;
}): ReactNode {
  const entries = useChatSlotEntries<TProps>(handle);
  const visible = filter ? entries.filter(filter) : entries;
  if (visible.length === 0) return fallback ?? null;
  return <>
    {visible.map((entry) => (
      <Fragment key={`${entry.ownerId}:${entry.ownerRevision}:${entry.id}`}>
        <SafePluginBoundary fallback={entryFallback ? entryFallback(entry) : null}>
          <InvokePluginRender render={() => entry.render(props)} />
        </SafePluginBoundary>
      </Fragment>
    ))}
  </>;
}
