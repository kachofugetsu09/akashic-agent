export interface StreamFrameScheduler {
  request(callback: (timestamp: number) => void): number;
  cancel(handle: number): void;
}

export type StreamAdvance<T> = (current: T, target: T, elapsedMs: number) => T;

interface PendingProjection<T> {
  previousId: string;
  target: T;
}

/** Keep stream presentation outside the app root and notify only affected message rows. */
export class StreamProjectionStore<T extends { id: string }> {
  private readonly scheduler: StreamFrameScheduler;
  private readonly advance: StreamAdvance<T>;
  private readonly projections = new Map<string, T>();
  private readonly pending = new Map<string, PendingProjection<T>>();
  private readonly listeners = new Map<string, Set<() => void>>();
  private frameHandle: number | null = null;
  private lastFrameAt: number | null = null;

  constructor(
    scheduler: StreamFrameScheduler,
    advance: StreamAdvance<T>,
  ) {
    this.scheduler = scheduler;
    this.advance = advance;
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

  /** Publish an immutable target and optionally reveal a terminal state immediately. */
  publish(previousId: string, previous: T, target: T, immediate: boolean): void {
    // 1. Preserve the visible prefix while replacing the authoritative target.
    const current = this.projections.get(previousId) ?? previous;
    if (immediate) {
      this.pending.delete(previousId);
      this.setProjection(previousId, target);
      if (target.id !== previousId) this.setProjection(target.id, target);
      this.cancelFrameWhenIdle();
      return;
    }
    if (!this.projections.has(previousId)) this.projections.set(previousId, current);
    this.pending.set(previousId, { previousId, target });

    // 2. One display clock drains every active message without waking the app root.
    if (this.frameHandle === null) {
      this.frameHandle = this.scheduler.request(this.advanceFrame);
    }
  }

  /** Drop projections already committed into the React-owned coarse snapshot. */
  reconcileBaseline(messages: readonly T[]): void {
    const baseline = new Map(messages.map((message) => [message.id, message]));
    for (const [key, projection] of this.projections) {
      if (baseline.get(projection.id) === projection) this.projections.delete(key);
    }
  }

  clear(): void {
    if (this.frameHandle !== null) this.scheduler.cancel(this.frameHandle);
    this.frameHandle = null;
    this.lastFrameAt = null;
    this.pending.clear();
    this.projections.clear();
  }

  private readonly advanceFrame = (timestamp: number) => {
    const elapsedMs = this.lastFrameAt === null ? 1000 / 60 : timestamp - this.lastFrameAt;
    this.lastFrameAt = timestamp;
    this.frameHandle = null;

    for (const [key, projection] of this.pending) {
      const current = this.projections.get(key);
      if (current === undefined) throw new Error(`stream projection missing current message: ${key}`);
      const next = this.advance(current, projection.target, elapsedMs);
      this.setProjection(projection.previousId, next);
      if (next.id !== projection.previousId) this.setProjection(next.id, next);
      if (next === projection.target) this.pending.delete(key);
    }

    if (this.pending.size > 0) {
      this.frameHandle = this.scheduler.request(this.advanceFrame);
    } else {
      this.lastFrameAt = null;
    }
  };

  private setProjection(messageId: string, projection: T): void {
    if (this.projections.get(messageId) === projection) return;
    this.projections.set(messageId, projection);
    const listeners = this.listeners.get(messageId);
    if (!listeners) return;
    for (const listener of listeners) listener();
  }

  private cancelFrameWhenIdle(): void {
    if (this.pending.size > 0 || this.frameHandle === null) return;
    this.scheduler.cancel(this.frameHandle);
    this.frameHandle = null;
    this.lastFrameAt = null;
  }
}

/** Reveal exactly one Unicode code point on every visible streaming frame. */
export function streamFrameBudget(_elapsedMs: number, backlog: number): number {
  return backlog > 0 ? 1 : 0;
}

export function appendCodePoints(current: string, target: string, budget: number) {
  if (budget <= 0 || current === target) return { text: current, appended: 0 };
  if (!target.startsWith(current)) return { text: target, appended: 0 };
  let appended = "";
  let count = 0;
  for (const codePoint of target.slice(current.length)) {
    if (count >= budget) break;
    appended += codePoint;
    count += 1;
  }
  return { text: current + appended, appended: count };
}
