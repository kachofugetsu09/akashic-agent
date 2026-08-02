export const MOBILE_PLUGIN_QUERY_MAX_PENDING = 128;
export const MOBILE_PLUGIN_QUERY_MAX_ACTIVE = 4;
export const MOBILE_PLUGIN_QUERY_MAX_BACKGROUND = 2;

export interface MobilePluginQueuedRequest {
  ownerId: string;
  started: boolean;
}

/** 同时限制插件查询总量与并发，并让完成和 owner 撤销释放同一份容量。 */
export class MobilePluginQueryQueue<T extends MobilePluginQueuedRequest> {
  private readonly pending = new Map<string, T>();
  private readonly queued: string[] = [];
  private readonly isInteractive: (request: T) => boolean;
  private readonly maxPending: number;
  private readonly maxActive: number;
  private readonly maxBackground: number;
  private active = 0;
  private activeBackground = 0;

  constructor(
    isInteractive: (request: T) => boolean,
    maxPending = MOBILE_PLUGIN_QUERY_MAX_PENDING,
    maxActive = MOBILE_PLUGIN_QUERY_MAX_ACTIVE,
    maxBackground = MOBILE_PLUGIN_QUERY_MAX_BACKGROUND,
  ) {
    if (!Number.isSafeInteger(maxPending) || maxPending <= 0) {
      throw new Error("插件请求总量预算无效");
    }
    if (!Number.isSafeInteger(maxActive) || maxActive <= 0 || maxActive > maxPending) {
      throw new Error("插件请求并发预算无效");
    }
    if (!Number.isSafeInteger(maxBackground) || maxBackground <= 0 || maxBackground > maxActive) {
      throw new Error("插件后台请求并发预算无效");
    }
    this.isInteractive = isInteractive;
    this.maxPending = maxPending;
    this.maxActive = maxActive;
    this.maxBackground = maxBackground;
  }

  get pendingCount() {
    return this.pending.size;
  }

  get queuedCount() {
    return this.queued.length;
  }

  get activeCount() {
    return this.active;
  }

  get(requestId: string): T | undefined {
    return this.pending.get(requestId);
  }

  enqueue(requestId: string, request: T): void {
    if (!requestId || this.pending.has(requestId)) throw new Error("插件请求身份重复或为空");
    if (request.started) throw new Error("插件请求入队前已经启动");
    if (this.pending.size >= this.maxPending) {
      throw new Error(`插件未完成请求已达上限（最多 ${this.maxPending} 个）`);
    }
    this.pending.set(requestId, request);
    this.queued.push(requestId);
  }

  /** 按交互优先级取出一个可启动请求，并原子推进并发计数。 */
  startNext(): [string, T] | undefined {
    if (this.active >= this.maxActive) return undefined;

    // 1. 交互请求优先；后台请求还要遵守独立并发上限
    const interactiveIndex = this.queued.findIndex((requestId) => {
      const request = this.pending.get(requestId);
      if (!request) throw new Error("插件请求队列与 pending 索引失配");
      return this.isInteractive(request);
    });
    const nextIndex = interactiveIndex >= 0
      ? interactiveIndex
      : this.activeBackground < this.maxBackground ? 0 : -1;
    if (nextIndex < 0 || nextIndex >= this.queued.length) return undefined;

    // 2. 从等待队列迁到 active，但仍计入 pending 总量
    const [requestId] = this.queued.splice(nextIndex, 1);
    const request = this.pending.get(requestId);
    if (!request || request.started) throw new Error("插件请求启动状态失配");
    request.started = true;
    this.active += 1;
    if (!this.isInteractive(request)) this.activeBackground += 1;
    return [requestId, request];
  }

  complete(requestId: string): T | undefined {
    const request = this.pending.get(requestId);
    if (!request) return undefined;

    // 1. queued 与 active 共用 pending 容量，完成任一状态都会释放一个名额
    this.pending.delete(requestId);
    const queuedIndex = this.queued.indexOf(requestId);
    if (queuedIndex >= 0) this.queued.splice(queuedIndex, 1);
    if (request.started) {
      this.active -= 1;
      if (!this.isInteractive(request)) this.activeBackground -= 1;
    }

    // 2. 内部计数一旦失配立即暴露调度错误
    if (this.active < 0 || this.activeBackground < 0 || this.activeBackground > this.active) {
      throw new Error("插件请求并发计数失配");
    }
    return request;
  }

  removeOwner(ownerId: string): Array<[string, T]> {
    const owned = [...this.pending].filter(([, request]) => request.ownerId === ownerId);
    owned.forEach(([requestId, request]) => {
      if (this.complete(requestId) !== request) throw new Error("插件 owner 请求释放失配");
    });
    return owned;
  }

  clear(): Array<[string, T]> {
    const requests = [...this.pending];
    this.pending.clear();
    this.queued.length = 0;
    this.active = 0;
    this.activeBackground = 0;
    return requests;
  }
}
