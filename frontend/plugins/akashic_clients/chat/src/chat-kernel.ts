/**
 * Akashic Chat 浏览器端正交微内核（React-free 纯核心）。
 *
 * 设计要点：
 * 1. 服务注册表（DI）：插件可通过 provide/list/subscribe 自由提供与消费命名服务，Core 零特判；
 * 2. 命名 list 槽：声明者仅可声明自己命名空间（`${pluginId}.`）下的槽，声明即渲染权（返回不透明 handle），
 *    声明撤销后旧 handle 立即失效读空，贡献者存活期间条目休眠保留、换代重新声明后自动接上；
 * 3. 事务性 lease 激活：同步 activate(ctx) 拒绝 thenable；激活中所有注册先进暂存区，
 *    全部通过后一次原子提交并刷新不可变快照再通知；任何抛错自动逆序回滚不留半激活残渣；
 * 4. 只读 getSnapshot：写时冻结不可变数组，未命中返回共享 EMPTY，满足 useSyncExternalStore 严格引用稳定。
 */

import type { ReactNode } from "react";
import type {
  ChatKernelDisposer,
  ChatServiceRecord,
  ChatSlotContribution,
  ChatSlotEntry,
  ChatSlotHandle,
} from "@akashic/chat-ui-v1";

export type {
  ChatKernelDisposer,
  ChatServiceRecord,
  ChatSlotContribution,
  ChatSlotEntry,
  ChatSlotHandle,
};

export interface ChatKernelQueryOptions {
  readonly signal?: AbortSignal;
}

export type ChatKernelQueryFn = (
  method: string,
  payload?: Record<string, unknown>,
  options?: ChatKernelQueryOptions,
) => Promise<Record<string, unknown>>;

export interface ChatKernelActivationContext {
  readonly pluginId: string;
  readonly pluginRevision: string;
  readonly signal: AbortSignal;
  query: ChatKernelQueryFn;
  onDispose(fn: ChatKernelDisposer): void;
  readonly services: {
    provide<T>(serviceId: string, value: T): ChatKernelDisposer;
    list<T>(serviceId: string): readonly ChatServiceRecord<T>[];
    get<T>(serviceId: string): T | null;
    subscribe(serviceId: string, listener: () => void): ChatKernelDisposer;
  };
  readonly slots: {
    declare(slotId: string): ChatSlotHandle;
    contribute<TProps = unknown>(
      slotId: string,
      entry: ChatSlotContribution<TProps>,
    ): ChatKernelDisposer;
  };
}

export interface ChatKernelLease {
  readonly pluginId: string;
  readonly pluginRevision: string;
  isActive(): boolean;
  dispose(): void;
}

const EMPTY_LIST: readonly never[] = Object.freeze([]);

function isThenable(value: unknown): boolean {
  return (typeof value === "object" || typeof value === "function")
    && value !== null
    && typeof (value as { then?: unknown }).then === "function";
}

function runDisposersReverse(disposers: ChatKernelDisposer[]): void {
  for (let i = disposers.length - 1; i >= 0; i -= 1) {
    try {
      disposers[i]();
    } catch {
      // 单个清理抛错不得阻断其余逆序清理。
    }
  }
}

interface InternalServiceEntry {
  readonly lease: LeaseState;
  readonly serviceId: string;
  readonly record: ChatServiceRecord<unknown>;
  removed: boolean;
}

interface InternalSlotDeclaration {
  readonly lease: LeaseState;
  readonly slotId: string;
  readonly listeners: Set<() => void>;
  active: boolean;
}

interface InternalSlotContribution {
  readonly lease: LeaseState;
  readonly slotId: string;
  readonly entry: ChatSlotEntry<unknown>;
  removed: boolean;
}

interface LeaseState {
  readonly pluginId: string;
  readonly pluginRevision: string;
  readonly abort: AbortController;
  admitting: boolean;
  committed: boolean;
  disposed: boolean;
  readonly cleanups: ChatKernelDisposer[];
}

function compareSlotEntries(a: ChatSlotEntry<unknown>, b: ChatSlotEntry<unknown>): number {
  if (a.order !== b.order) return a.order - b.order;
  if (a.ownerId !== b.ownerId) return a.ownerId < b.ownerId ? -1 : 1;
  if (a.id !== b.id) return a.id < b.id ? -1 : 1;
  return 0;
}

export class ChatKernel {
  private readonly serviceEntries = new Map<string, InternalServiceEntry[]>();
  private readonly serviceSnapshots = new Map<string, readonly ChatServiceRecord<unknown>[]>();
  private readonly serviceListeners = new Map<string, Set<() => void>>();

  private readonly slotDeclarations = new Map<string, InternalSlotDeclaration>();
  private readonly slotContributions = new Map<string, InternalSlotContribution[]>();
  private readonly slotSnapshots = new Map<string, readonly ChatSlotEntry<unknown>[]>();
  private readonly slotListeners = new Map<string, Set<() => void>>();

  /**
   * 同步激活一个插件实例，返回私有 lease 句柄：
   * - 激活中登记的所有服务、槽声明与槽贡献先进入暂存区；
   * - 若 activate 抛错或返回 Promise/thenable，不发布任何记录并逆序执行已注册清理；
   * - 全部校验通过后一次原子提交、刷新冻结快照并通知相关订阅者。
   */
  activate(
    pluginId: string,
    pluginRevision: string,
    query: ChatKernelQueryFn,
    run: (ctx: ChatKernelActivationContext) => void | ChatKernelDisposer,
  ): ChatKernelLease {
    const lease: LeaseState = {
      pluginId,
      pluginRevision,
      abort: new AbortController(),
      admitting: true,
      committed: false,
      disposed: false,
      cleanups: [],
    };

    const stagedServices: InternalServiceEntry[] = [];
    const stagedDeclarations: InternalSlotDeclaration[] = [];
    const stagedContributions: InternalSlotContribution[] = [];

    const assertLive = (action: string) => {
      if (!lease.admitting || lease.disposed) {
        throw new Error(`插件 ${pluginId}@${pluginRevision} 已撤销，禁止 ${action}`);
      }
    };

    const addCleanup = (fn: ChatKernelDisposer) => {
      let called = false;
      const once: ChatKernelDisposer = () => {
        if (called) return;
        called = true;
        fn();
      };
      lease.cleanups.push(once);
      return once;
    };

    const ctx: ChatKernelActivationContext = {
      pluginId,
      pluginRevision,
      signal: lease.abort.signal,
      query: (method, payload = {}, options) => {
        if (!lease.admitting || lease.disposed) {
          return Promise.reject(new Error(`插件 ${pluginId}@${pluginRevision} 已撤销，无法发起查询 ${method}`));
        }
        const mergedSignal = options?.signal
          ? AbortSignal.any([lease.abort.signal, options.signal])
          : lease.abort.signal;
        return query(method, payload, { signal: mergedSignal });
      },
      onDispose: (fn) => {
        assertLive("onDispose");
        if (typeof fn !== "function") throw new TypeError("onDispose 必须传入函数");
        addCleanup(fn);
      },
      services: {
        provide: <T,>(serviceId: string, value: T): ChatKernelDisposer => {
          assertLive(`provide(${serviceId})`);
          if (typeof serviceId !== "string" || !serviceId.trim()) {
            throw new TypeError("serviceId 必须是非空字符串");
          }
          const item: InternalServiceEntry = {
            lease,
            serviceId,
            record: Object.freeze({ ownerId: pluginId, ownerRevision: pluginRevision, value }),
            removed: false,
          };
          const undo = addCleanup(() => {
            if (item.removed) return;
            item.removed = true;
            if (!lease.committed) return;
            if (this.removeService(item)) {
              this.rebuildServiceSnapshot(serviceId);
              this.notifyService(serviceId);
            }
          });
          if (!lease.committed) {
            stagedServices.push(item);
          } else {
            this.insertService(item);
            this.rebuildServiceSnapshot(serviceId);
            this.notifyService(serviceId);
          }
          return undo;
        },
        list: <T,>(serviceId: string) => this.listServices<T>(serviceId),
        get: <T,>(serviceId: string) => this.getService<T>(serviceId),
        subscribe: (serviceId: string, listener: () => void) => {
          assertLive(`subscribe(${serviceId})`);
          const unsub = this.subscribeService(serviceId, listener);
          return addCleanup(unsub);
        },
      },
      slots: {
        declare: (slotId: string): ChatSlotHandle => {
          assertLive(`declare(${slotId})`);
          if (typeof slotId !== "string" || !slotId.startsWith(`${pluginId}.`) || slotId.length <= pluginId.length + 1) {
            throw new Error(`插件 ${pluginId} 只能声明自身命名空间下的槽（须以 "${pluginId}." 开头）: ${slotId}`);
          }
          const existing = this.slotDeclarations.get(slotId);
          if (existing?.active || stagedDeclarations.some((d) => d.slotId === slotId && d.active)) {
            throw new Error(`槽 ${slotId} 已被声明，禁止重复声明`);
          }
          const decl: InternalSlotDeclaration = { lease, slotId, listeners: new Set(), active: true };
          const undo = addCleanup(() => {
            if (!decl.active) return;
            decl.active = false;
            const prevListeners = [...decl.listeners];
            decl.listeners.clear();
            if (!lease.committed) return;
            if (this.slotDeclarations.get(slotId) === decl) {
              this.slotDeclarations.delete(slotId);
              this.slotSnapshots.delete(slotId);
              this.notifySlot(slotId);
            }
            for (const listener of prevListeners) {
              try { listener(); } catch (err) { console.error(`[chat-kernel] slot listener error (${slotId})`, err); }
            }
          });
          if (!lease.committed) {
            stagedDeclarations.push(decl);
          } else {
            this.slotDeclarations.set(slotId, decl);
            this.rebuildSlotSnapshot(slotId);
            this.notifySlot(slotId);
          }
          return {
            slotId,
            ownerId: pluginId,
            ownerRevision: pluginRevision,
            isCurrent: () => decl.active && lease.committed && this.slotDeclarations.get(slotId) === decl,
            entries: <TProps = unknown>() => {
              if (!decl.active || !lease.committed || this.slotDeclarations.get(slotId) !== decl) {
                return EMPTY_LIST as readonly ChatSlotEntry<TProps>[];
              }
              return (this.slotSnapshots.get(slotId) ?? EMPTY_LIST) as readonly ChatSlotEntry<TProps>[];
            },
            subscribe: (listener) => {
              // 订阅严格绑定到当前声明实例；声明撤销后自动停止并释放，失效句柄不再接受订阅。
              if (!decl.active || lease.disposed) return () => {};
              decl.listeners.add(listener);
              return () => { decl.listeners.delete(listener); };
            },
            dispose: undo,
          };
        },
        contribute: <TProps = unknown>(
          slotId: string,
          entry: ChatSlotContribution<TProps>,
        ): ChatKernelDisposer => {
          assertLive(`contribute(${slotId})`);
          if (typeof slotId !== "string" || !slotId.includes(".")) {
            throw new TypeError(`slotId 无效: ${slotId}`);
          }
          if (!entry || typeof entry.id !== "string" || !entry.id || typeof entry.render !== "function") {
            throw new TypeError(`槽贡献 ${slotId} 必须提供非空 id 与 render 函数`);
          }
          // 同一槽、同一激活 lease 下禁止重复贡献 ID（避免重复 React key）；已释放后可重新注册。
          const existingCommitted = this.slotContributions.get(slotId);
          const duplicate = (existingCommitted?.some((c) => !c.removed && c.lease === lease && c.entry.id === entry.id) ?? false)
            || stagedContributions.some((c) => !c.removed && c.slotId === slotId && c.entry.id === entry.id);
          if (duplicate) {
            throw new Error(`插件 ${pluginId} 在槽 ${slotId} 下重复注册贡献 id: ${entry.id}`);
          }
          const order = typeof entry.order === "number" && Number.isFinite(entry.order) ? entry.order : 0;
          const record: InternalSlotContribution = {
            lease,
            slotId,
            entry: Object.freeze({
              ownerId: pluginId,
              ownerRevision: pluginRevision,
              id: entry.id,
              order,
              ...(typeof entry.replacesMetaId === "string" && entry.replacesMetaId
                ? { replacesMetaId: entry.replacesMetaId }
                : {}),
              render: entry.render as (props: unknown) => ReactNode,
            }),
            removed: false,
          };
          const undo = addCleanup(() => {
            if (record.removed) return;
            record.removed = true;
            if (!lease.committed) return;
            if (this.removeContribution(record)) {
              this.rebuildSlotSnapshot(slotId);
              this.notifySlot(slotId);
            }
          });
          if (!lease.committed) {
            stagedContributions.push(record);
          } else {
            this.insertContribution(record);
            this.rebuildSlotSnapshot(slotId);
            this.notifySlot(slotId);
          }
          return undo;
        },
      },
    };

    try {
      const returned = run(ctx);
      if (isThenable(returned)) {
        throw new Error(`插件 ${pluginId} 的 activate(ctx) 必须是同步函数，不允许返回 Promise`);
      }
      if (returned !== undefined && typeof returned !== "function") {
        throw new TypeError(`插件 ${pluginId} 的 activate(ctx) 只能返回清理函数或 undefined`);
      }
      if (typeof returned === "function") {
        addCleanup(returned);
      }
    } catch (error) {
      lease.admitting = false;
      lease.disposed = true;
      lease.abort.abort(error);
      runDisposersReverse(lease.cleanups);
      lease.cleanups.length = 0;
      throw error;
    }

    // 原子提交：先写入全部结构并刷新所有受影响快照，再统一触发监听者通知。
    lease.committed = true;
    const touchedServices = new Set<string>();
    const touchedSlots = new Set<string>();

    for (const item of stagedServices) {
      if (item.removed) continue;
      this.insertService(item);
      touchedServices.add(item.serviceId);
    }
    for (const decl of stagedDeclarations) {
      if (!decl.active) continue;
      this.slotDeclarations.set(decl.slotId, decl);
      touchedSlots.add(decl.slotId);
    }
    for (const contrib of stagedContributions) {
      if (contrib.removed) continue;
      this.insertContribution(contrib);
      touchedSlots.add(contrib.slotId);
    }

    for (const serviceId of touchedServices) this.rebuildServiceSnapshot(serviceId);
    for (const slotId of touchedSlots) this.rebuildSlotSnapshot(slotId);
    for (const serviceId of touchedServices) this.notifyService(serviceId);
    for (const slotId of touchedSlots) this.notifySlot(slotId);

    return {
      pluginId,
      pluginRevision,
      isActive: () => !lease.disposed && lease.admitting,
      dispose: () => {
        if (lease.disposed) return;
        lease.admitting = false;
        lease.disposed = true;
        lease.abort.abort(new Error("plugin disposed"));
        runDisposersReverse(lease.cleanups);
        lease.cleanups.length = 0;
      },
    };
  }

  /** 宿主直连服务注册（用于桥接尚未迁移的 legacy slot，遵循显式优先、桥接补缺）。 */
  provideHostService<T>(ownerId: string, ownerRevision: string, serviceId: string, value: T): ChatKernelLease {
    return this.activate(
      ownerId,
      ownerRevision,
      () => Promise.reject(new Error("桥接宿主服务不支持直接 query")),
      (ctx) => {
        ctx.services.provide(serviceId, value);
      },
    );
  }

  listServices<T>(serviceId: string): readonly ChatServiceRecord<T>[] {
    return (this.serviceSnapshots.get(serviceId) ?? EMPTY_LIST) as readonly ChatServiceRecord<T>[];
  }

  getService<T>(serviceId: string): T | null {
    const list = this.serviceSnapshots.get(serviceId);
    return list && list.length > 0 ? (list[0].value as T) : null;
  }

  subscribeService(serviceId: string, listener: () => void): ChatKernelDisposer {
    let bucket = this.serviceListeners.get(serviceId);
    if (!bucket) {
      bucket = new Set();
      this.serviceListeners.set(serviceId, bucket);
    }
    bucket.add(listener);
    return () => {
      bucket?.delete(listener);
      if (bucket && bucket.size === 0) this.serviceListeners.delete(serviceId);
    };
  }

  subscribeSlot(slotId: string, listener: () => void): ChatKernelDisposer {
    let bucket = this.slotListeners.get(slotId);
    if (!bucket) {
      bucket = new Set();
      this.slotListeners.set(slotId, bucket);
    }
    bucket.add(listener);
    return () => {
      bucket?.delete(listener);
      if (bucket && bucket.size === 0) this.slotListeners.delete(slotId);
    };
  }

  private insertService(item: InternalServiceEntry): void {
    const list = this.serviceEntries.get(item.serviceId);
    if (list) list.push(item);
    else this.serviceEntries.set(item.serviceId, [item]);
  }

  private removeService(item: InternalServiceEntry): boolean {
    const list = this.serviceEntries.get(item.serviceId);
    if (!list) return false;
    const idx = list.indexOf(item);
    if (idx === -1) return false;
    list.splice(idx, 1);
    if (list.length === 0) this.serviceEntries.delete(item.serviceId);
    return true;
  }

  private rebuildServiceSnapshot(serviceId: string): void {
    const list = this.serviceEntries.get(serviceId);
    if (!list || list.length === 0) {
      this.serviceSnapshots.delete(serviceId);
      return;
    }
    // 确定性排序（按 ownerId 字典序），避免网络加载到达先后导致顺序抖动。
    const sorted = [...list]
      .sort((a, b) => (a.record.ownerId < b.record.ownerId ? -1 : a.record.ownerId > b.record.ownerId ? 1 : 0))
      .map((item) => item.record);
    this.serviceSnapshots.set(serviceId, Object.freeze(sorted));
  }

  private notifyService(serviceId: string): void {
    const bucket = this.serviceListeners.get(serviceId);
    if (!bucket) return;
    for (const listener of [...bucket]) {
      try {
        listener();
      } catch (err) {
        console.error(`[chat-kernel] service listener error (${serviceId})`, err);
      }
    }
  }

  private insertContribution(item: InternalSlotContribution): void {
    const list = this.slotContributions.get(item.slotId);
    if (list) list.push(item);
    else this.slotContributions.set(item.slotId, [item]);
  }

  private removeContribution(item: InternalSlotContribution): boolean {
    const list = this.slotContributions.get(item.slotId);
    if (!list) return false;
    const idx = list.indexOf(item);
    if (idx === -1) return false;
    list.splice(idx, 1);
    if (list.length === 0) this.slotContributions.delete(item.slotId);
    return true;
  }

  private rebuildSlotSnapshot(slotId: string): void {
    const decl = this.slotDeclarations.get(slotId);
    const list = this.slotContributions.get(slotId);
    if (!decl?.active || !list || list.length === 0) {
      this.slotSnapshots.delete(slotId);
      return;
    }
    const sorted = [...list].map((item) => item.entry).sort(compareSlotEntries);
    this.slotSnapshots.set(slotId, Object.freeze(sorted));
  }

  private notifySlot(slotId: string): void {
    const decl = this.slotDeclarations.get(slotId);
    if (decl?.active) {
      for (const listener of [...decl.listeners]) {
        try {
          listener();
        } catch (err) {
          console.error(`[chat-kernel] slot handle listener error (${slotId})`, err);
        }
      }
    }
    const bucket = this.slotListeners.get(slotId);
    if (!bucket) return;
    for (const listener of [...bucket]) {
      try {
        listener();
      } catch (err) {
        console.error(`[chat-kernel] slot listener error (${slotId})`, err);
      }
    }
  }
}

/** 聊天应用共享的单一浏览器微内核实例。 */
export const chatKernel = new ChatKernel();
