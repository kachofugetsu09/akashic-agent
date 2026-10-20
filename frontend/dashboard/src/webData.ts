import type { WebResource, WebResourceOptions, WebUiDisposer } from "@akashic/web-ui-v1";

type Load<T> = (signal: AbortSignal) => Promise<T>;

// 快照只在同一标签页内跨刷新复用；超过上限的值只留在内存里。
const SNAPSHOT_LIMIT = 256 * 1024;
const SNAPSHOT_PREFIX = "akashic.web-data.v1:";

/** 一个模块拥有的数据资源：生命周期与模块激活相同，页面挂载只读不取。 */
class ModuleResource<T> implements WebResource<T> {
  current: T | undefined;
  error: Error | undefined;
  private readonly listeners = new Set<() => void>();
  private readonly controller = new AbortController();
  private inflight: Promise<T> | null = null;
  private loaded = false;
  private serialized: string | undefined;

  constructor(
    private readonly load: Load<T>,
    private readonly snapshotKey: string,
    readonly warm: boolean,
  ) {
    const snapshot = readSnapshot<T>(snapshotKey);
    this.serialized = snapshot?.raw;
    this.current = snapshot?.value;
  }

  get settled(): boolean {
    return this.loaded;
  }

  // 箭头属性保持同一身份，React useSyncExternalStore 不会每次渲染重新订阅。
  readonly subscribe = (listener: () => void): WebUiDisposer => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  // 1. 进行中的读取直接复用；2. 读取期间 current 保持旧值；3. 成功后写快照并通知。
  refresh(): Promise<T> {
    if (this.controller.signal.aborted) return Promise.reject(new Error("web module is disposed"));
    this.inflight ??= this.load(this.controller.signal).then((value) => {
      this.inflight = null;
      this.loaded = true;
      return this.accept(value);
    }, (reason: unknown) => {
      this.inflight = null;
      if (!this.controller.signal.aborted) {
        this.error = reason instanceof Error ? reason : new Error(String(reason));
        this.notify();
      }
      throw reason;
    });
    return this.inflight;
  }

  set(value: T): void {
    if (this.controller.signal.aborted) return;
    this.loaded = true;
    this.accept(value);
  }

  dispose(): void {
    this.controller.abort();
    this.listeners.clear();
  }

  // 内容未变时保留原对象且不通知：后台核对不引起重绘。
  private accept(value: T): T {
    const serialized = serialize(value);
    if (serialized !== undefined && serialized === this.serialized && this.error === undefined) return this.current as T;
    this.current = value;
    this.serialized = serialized;
    this.error = undefined;
    writeSnapshot(this.snapshotKey, serialized);
    this.notify();
    return value;
  }

  private notify(): void {
    for (const listener of [...this.listeners]) listener();
  }
}

/** 宿主侧的资源登记：模块释放时一并中止；预热在启动完成后的空闲时段执行。 */
export class WebDataRegistry {
  private readonly resources = new Set<ModuleResource<unknown>>();

  create<T>(scope: string, key: string, load: Load<T>, options: WebResourceOptions | undefined): {
    resource: WebResource<T>;
    dispose: WebUiDisposer;
  } {
    if (typeof key !== "string" || key === "") throw new Error("resource key must be a non-empty string");
    if (typeof load !== "function") throw new Error(`resource ${key} load must be a function`);
    const resource = new ModuleResource<T>(load, `${SNAPSHOT_PREFIX}${scope}:${key}`, options?.warm === true);
    this.resources.add(resource as ModuleResource<unknown>);
    return {
      resource,
      dispose: () => {
        resource.dispose();
        this.resources.delete(resource as ModuleResource<unknown>);
      },
    };
  }

  // 空闲时依次预热；每项之间让出主线程，不和首屏渲染抢时间。
  warmWhenIdle(): void {
    const pending = [...this.resources].filter((item) => item.warm && !item.settled);
    const next = (): void => {
      const resource = pending.shift();
      if (!resource) return;
      resource.refresh().catch(() => undefined).finally(() => idle(next));
    };
    idle(next);
  }
}

function idle(callback: () => void): void {
  if (typeof window.requestIdleCallback === "function") window.requestIdleCallback(callback, { timeout: 1500 });
  else window.setTimeout(callback, 50);
}

function serialize(value: unknown): string | undefined {
  try {
    return JSON.stringify(value);
  } catch {
    return undefined;
  }
}

function readSnapshot<T>(key: string): { raw: string; value: T } | undefined {
  try {
    const raw = window.sessionStorage.getItem(key);
    return raw === null ? undefined : { raw, value: JSON.parse(raw) as T };
  } catch {
    return undefined;
  }
}

function writeSnapshot(key: string, serialized: string | undefined): void {
  try {
    if (serialized === undefined || serialized.length > SNAPSHOT_LIMIT) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, serialized);
  } catch {
    // 存储不可用或已满时只保留内存值。
  }
}
