type ScheduleFrame = (callback: FrameRequestCallback) => number;
type CancelFrame = (handle: number) => void;

const scheduleBrowserFrame: ScheduleFrame = (callback) => globalThis.requestAnimationFrame(callback);
const cancelBrowserFrame: CancelFrame = (handle) => globalThis.cancelAnimationFrame(handle);

/** 按消息缓存最新权威投影，并把高频通知合并到显示帧。 */
export class StreamProjectionStore<T extends { id: string }> {
  private readonly projections = new Map<string, T>();
  private readonly listeners = new Map<string, Set<() => void>>();
  private readonly dirtyKeys = new Set<string>();
  private readonly scheduleFrame: ScheduleFrame;
  private readonly cancelFrame: CancelFrame;
  private scheduledFrame: number | null = null;

  constructor(
    scheduleFrame: ScheduleFrame = scheduleBrowserFrame,
    cancelFrame: CancelFrame = cancelBrowserFrame,
  ) {
    this.scheduleFrame = scheduleFrame;
    this.cancelFrame = cancelFrame;
  }

  read(messageId: string, fallback: T): T {
    return this.projections.get(messageId) ?? fallback;
  }

  subscribe(messageId: string, listener: () => void): () => void {
    const listeners = this.listeners.get(messageId) ?? new Set();
    listeners.add(listener);
    this.listeners.set(messageId, listeners);
    return () => {
      listeners.delete(listener);
      if (listeners.size === 0) this.listeners.delete(messageId);
    };
  }

  /** 保存最新 target，并在下一显示帧只通知每个受影响消息一次。 */
  publishFrame(previousId: string, target: T): void {
    const changedKeys = this.updateProjections(previousId, target);
    for (const key of changedKeys) this.dirtyKeys.add(key);
    if (this.dirtyKeys.size === 0 || this.scheduledFrame !== null) return;
    this.scheduledFrame = this.scheduleFrame(() => this.flushFrame());
  }

  /** 取消待发布帧并立即提交 terminal 或结构变化。 */
  publishImmediate(previousId: string, target: T): void {
    const changedKeys = this.updateProjections(previousId, target);
    const publishKeys = new Set(this.dirtyKeys);
    for (const key of changedKeys) publishKeys.add(key);
    this.cancelScheduledFrame();
    this.dirtyKeys.clear();
    this.notify(publishKeys);
  }

  /** Drop projections already committed into the React-owned coarse snapshot. */
  reconcileBaseline(messages: readonly T[]): void {
    const baseline = new Map(messages.map((message) => [message.id, message]));
    for (const [key, projection] of this.projections) {
      if (baseline.get(projection.id) !== projection) continue;
      this.projections.delete(key);
      this.dirtyKeys.delete(key);
    }
    if (this.dirtyKeys.size === 0) this.cancelScheduledFrame();
  }

  clear(): void {
    this.cancelScheduledFrame();
    this.dirtyKeys.clear();
    this.projections.clear();
  }

  private updateProjections(previousId: string, target: T): string[] {
    const changedKeys: string[] = [];
    this.updateProjection(previousId, target, changedKeys);
    if (target.id !== previousId) this.updateProjection(target.id, target, changedKeys);
    return changedKeys;
  }

  private updateProjection(messageId: string, projection: T, changedKeys: string[]): void {
    if (this.projections.get(messageId) === projection) return;
    this.projections.set(messageId, projection);
    changedKeys.push(messageId);
  }

  private flushFrame(): void {
    this.scheduledFrame = null;
    const publishKeys = new Set(this.dirtyKeys);
    this.dirtyKeys.clear();
    this.notify(publishKeys);
  }

  private cancelScheduledFrame(): void {
    if (this.scheduledFrame === null) return;
    this.cancelFrame(this.scheduledFrame);
    this.scheduledFrame = null;
  }

  private notify(messageIds: ReadonlySet<string>): void {
    for (const messageId of messageIds) {
      const listeners = this.listeners.get(messageId);
      if (!listeners) continue;
      for (const listener of [...listeners]) listener();
    }
  }
}
