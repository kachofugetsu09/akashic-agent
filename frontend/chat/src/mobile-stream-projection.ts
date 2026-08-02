export interface MobileStreamFrameScheduler {
  request(callback: (timestamp: number) => void): number;
  cancel(handle: number): void;
}

export type MobileStreamAdvance<T> = (current: T, target: T, elapsedMs: number) => T;

interface PendingProjection<T> {
  previousId: string;
  target: T;
}

/** Keep native stream state outside the app root and notify only the affected message row. */
export class MobileStreamProjectionStore<T extends { id: string }> {
  private readonly scheduler: MobileStreamFrameScheduler;
  private readonly advance: MobileStreamAdvance<T>;
  private readonly projections = new Map<string, T>();
  private readonly pending = new Map<string, PendingProjection<T>>();
  private readonly listeners = new Map<string, Set<() => void>>();
  private frameHandle: number | null = null;
  private lastFrameAt: number | null = null;

  constructor(
    scheduler: MobileStreamFrameScheduler,
    advance: MobileStreamAdvance<T>,
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

  /** Publish an immutable target, optionally bypassing smoothing for terminal state. */
  publish(previousId: string, previous: T, target: T, immediate: boolean): void {
    // 1. Preserve the currently visible prefix while replacing the authoritative target.
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

export interface MobileStreamPresentationBlock {
  id: string;
  detail: string;
}

export interface MobileStreamPresentationMessage {
  id: string;
  content: string;
  blocks: MobileStreamPresentationBlock[];
}

/** Advance text by a frame-sized Unicode slice while applying structural fields immediately. */
export function advanceMobileStreamPresentation<T extends MobileStreamPresentationMessage>(
  current: T,
  target: T,
  elapsedMs: number,
): T {
  if (current.id !== target.id) return target;

  // 1. Allocate roughly 100 code points per second, with bounded backlog catch-up.
  const backlog = mobileStreamBacklog(current, target);
  if (backlog === 0) return target;
  let budget = mobileStreamFrameBudget(elapsedMs, backlog);

  // 2. Preserve thinking order, then spend the remaining frame budget on the answer.
  const projectedBlocks = target.blocks.map((block, index) => {
    const previous = current.blocks[index];
    if (previous?.id !== block.id || !block.detail.startsWith(previous.detail)) {
      return block;
    }
    if (budget === 0) return previous.detail === block.detail ? block : { ...block, detail: previous.detail };
    const advanced = appendCodePoints(previous.detail, block.detail, budget);
    budget -= advanced.appended;
    return advanced.text === block.detail ? block : { ...block, detail: advanced.text };
  });
  const blocks = projectedBlocks.every((block, index) => block === target.blocks[index])
    ? target.blocks
    : projectedBlocks;
  const content = appendCodePoints(current.content, target.content, budget).text;
  if (
    content === target.content
    && blocks.every((block, index) => block === target.blocks[index])
  ) {
    return target;
  }
  return { ...target, content, blocks };
}

export function mobileStreamFrameBudget(elapsedMs: number, backlog: number): number {
  const boundedElapsed = Math.min(50, Math.max(8, elapsedMs));
  const base = Math.max(1, Math.round(boundedElapsed / 10));
  const catchUp = backlog > 24 ? Math.min(4, Math.ceil((backlog - 24) / 24)) : 0;
  return Math.min(backlog, base + catchUp);
}

function mobileStreamBacklog(
  current: MobileStreamPresentationMessage,
  target: MobileStreamPresentationMessage,
): number {
  let backlog = appendedCodePointCount(current.content, target.content);
  for (let index = 0; index < target.blocks.length; index += 1) {
    const previous = current.blocks[index];
    const block = target.blocks[index];
    if (previous?.id === block.id) backlog += appendedCodePointCount(previous.detail, block.detail);
  }
  return backlog;
}

function appendedCodePointCount(current: string, target: string): number {
  if (!target.startsWith(current)) return 0;
  const appended = target.slice(current.length);
  let count = 0;
  let index = 0;
  while (index < appended.length) {
    const codePoint = appended.codePointAt(index);
    if (codePoint === undefined) break;
    index += codePoint > 0xffff ? 2 : 1;
    count += 1;
  }
  return count;
}

function appendCodePoints(current: string, target: string, budget: number) {
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
