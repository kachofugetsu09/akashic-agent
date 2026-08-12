/**
 * 流式补丁的按行投影缓存：每个补丁到达即把权威 target 立即发布到对应消息行，
 * 客户端不做任何速率控制——显示节奏完全等于服务端投递节奏。
 * 服务端负责投递节流（移动端 delta 以 60Hz/4KiB 聚合），前端只保证：
 * 1. 流式更新不唤醒 app root，只通知受影响的消息行；
 * 2. terminal 补丁保留 canonical messageId 别名；
 * 3. 粗粒度 snapshot 提交后丢弃已提交进 React 的投影。
 */
export class StreamProjectionStore<T extends { id: string }> {
  private readonly projections = new Map<string, T>();
  private readonly listeners = new Map<string, Set<() => void>>();

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

  /** 立即发布权威 target；id 迁移时同步保留旧 id 的读取别名。 */
  publish(previousId: string, target: T): void {
    this.setProjection(previousId, target);
    if (target.id !== previousId) this.setProjection(target.id, target);
  }

  /** Drop projections already committed into the React-owned coarse snapshot. */
  reconcileBaseline(messages: readonly T[]): void {
    const baseline = new Map(messages.map((message) => [message.id, message]));
    for (const [key, projection] of this.projections) {
      if (baseline.get(projection.id) === projection) this.projections.delete(key);
    }
  }

  clear(): void {
    this.projections.clear();
  }

  private setProjection(messageId: string, projection: T): void {
    if (this.projections.get(messageId) === projection) return;
    this.projections.set(messageId, projection);
    const listeners = this.listeners.get(messageId);
    if (!listeners) return;
    for (const listener of listeners) listener();
  }
}
