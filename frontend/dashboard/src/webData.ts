import type { WebRefreshOptions, WebResource, WebResourceOptions, WebUiDisposer } from "@akashic/web-ui-v1";

type Load<T> = (signal: AbortSignal) => Promise<T>;

/** 一个模块拥有的数据资源：生命周期与模块激活相同，页面挂载只读不取。 */
class ModuleResource<T> implements WebResource<T> {
  current: T | undefined;
  error: Error | undefined;
  private readonly listeners = new Set<() => void>();
  private readonly controller = new AbortController();
  // 最新一次读取或写入的代次；被取代的读取不得覆盖当前值。
  private generation = 0;
  private inflight: Promise<T> | null = null;
  private loaded = false;
  private serialized: string | undefined;

  constructor(private readonly load: Load<T>, readonly warm: boolean) {}

  get settled(): boolean {
    return this.loaded;
  }

  // 箭头属性保持同一身份，React useSyncExternalStore 不会每次渲染重新订阅。
  readonly subscribe = (listener: () => void): WebUiDisposer => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  // 1. 普通刷新并入进行中的读取；2. fresh 另起一次，旧读取作废；3. 作废的读取跟随最新结果。
  refresh(options?: WebRefreshOptions): Promise<T> {
    if (this.controller.signal.aborted) return Promise.reject(new Error("web module is disposed"));
    if (this.inflight && options?.fresh !== true) return this.inflight;
    const generation = ++this.generation;
    const read = this.load(this.controller.signal).then((value) => {
      if (this.controller.signal.aborted) throw new Error("web module is disposed");
      if (generation !== this.generation) return this.latest();
      this.inflight = null;
      this.loaded = true;
      return this.accept(value);
    }, (reason: unknown) => {
      if (this.controller.signal.aborted) throw reason;
      if (generation !== this.generation) return this.latest();
      this.inflight = null;
      this.error = reason instanceof Error ? reason : new Error(String(reason));
      this.notify();
      throw reason;
    });
    this.inflight = read;
    return read;
  }

  // 写操作的结果取代所有在途读取。
  set(value: T): void {
    if (this.controller.signal.aborted) return;
    this.generation += 1;
    this.inflight = null;
    this.loaded = true;
    this.accept(value);
  }

  dispose(): void {
    this.controller.abort();
    this.listeners.clear();
  }

  private latest(): Promise<T> {
    return this.inflight ?? Promise.resolve(this.current as T);
  }

  // 内容未变时保留原对象且不通知：后台核对不引起重绘。
  private accept(value: T): T {
    const serialized = serialize(value);
    if (serialized !== undefined && serialized === this.serialized && this.error === undefined) return this.current as T;
    this.current = value;
    this.serialized = serialized;
    this.error = undefined;
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

  create<T>(key: string, load: Load<T>, options: WebResourceOptions | undefined): {
    resource: WebResource<T>;
    dispose: WebUiDisposer;
  } {
    if (typeof key !== "string" || key === "") throw new Error("resource key must be a non-empty string");
    if (typeof load !== "function") throw new Error(`resource ${key} load must be a function`);
    const resource = new ModuleResource<T>(load, options?.warm === true);
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
