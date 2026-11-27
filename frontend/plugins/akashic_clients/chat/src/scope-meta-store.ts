import { useEffect, useMemo, useSyncExternalStore } from "react";
import type { ScopeMetaItem } from "@akashic/chat-ui-v1";
import {
  onPageShown,
  onPluginUiSlotInvalidated,
  scopeMetaSources,
  subscribePluginUiCatalog,
  type ScopeMetaSource,
} from "./plugin-ui-runtime";

export interface ScopeMetaRecord {
  readonly owner: string;
  readonly key: string;
  readonly item: ScopeMetaItem;
}

interface OwnerCache {
  revision: string;
  values: Map<string, readonly ScopeMetaItem[]>;
  stale: Set<string>;
  pending: Set<string>;
  failedUntil: Map<string, number>;
  controller: AbortController;
}

interface RequestedScopeEntry {
  scope: { dimension: string; value: string };
  refs: number;
}

const BATCH_SIZE = 100;
const RETRY_COOLDOWN_MS = 5_000;
const ownerCaches = new Map<string, OwnerCache>();
const requestedScopes = new Map<string, RequestedScopeEntry>();
const listeners = new Set<() => void>();
let storeVersion = 0;
let initialized = false;
let flushTimer: number | null = null;

const scopeKeyOf = (scope: { dimension: string; value: string }): string => `${scope.dimension}:${scope.value}`;

function emit() {
  storeVersion += 1;
  listeners.forEach((listener) => listener());
}

function ensureSubscribed() {
  if (initialized) return;
  initialized = true;
  subscribePluginUiCatalog(() => {
    reconcileOwners();
    scheduleFlush();
  });
  onPluginUiSlotInvalidated("scope.meta", () => {
    invalidateAllCaches();
    scheduleFlush();
  });
  onPageShown(() => {
    invalidateAllCaches();
    scheduleFlush();
  });
}

function invalidateAllCaches() {
  for (const cache of ownerCaches.values()) {
    cache.controller.abort();
    cache.controller = new AbortController();
    cache.pending.clear();
    cache.failedUntil.clear();
    // 保留现有值避免 UI 闪白，仅对当前仍挂载的 scope 标 stale，促使下次 flush 重拉。
    for (const key of cache.values.keys()) {
      if (requestedScopes.has(key)) cache.stale.add(key);
      else cache.values.delete(key);
    }
  }
}

function reconcileOwners() {
  const active = new Map(scopeMetaSources().map((source) => [source.pluginId, source]));
  let changed = false;
  for (const [pluginId, cache] of [...ownerCaches.entries()]) {
    const next = active.get(pluginId);
    if (!next || next.pluginRevision !== cache.revision) {
      cache.controller.abort();
      ownerCaches.delete(pluginId);
      changed = true;
    }
  }
  if (changed) emit();
}

function getOrCreateCache(source: ScopeMetaSource): OwnerCache {
  const existing = ownerCaches.get(source.pluginId);
  if (existing && existing.revision === source.pluginRevision) return existing;
  if (existing) existing.controller.abort();
  const created: OwnerCache = {
    revision: source.pluginRevision,
    values: new Map(),
    stale: new Set(),
    pending: new Set(),
    failedUntil: new Map(),
    controller: new AbortController(),
  };
  ownerCaches.set(source.pluginId, created);
  return created;
}

function normalizeItems(raw: unknown): readonly ScopeMetaItem[] {
  if (!Array.isArray(raw)) return [];
  const out: ScopeMetaItem[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const row = entry as Partial<Record<keyof ScopeMetaItem, unknown>>;
    if (typeof row.id !== "string" || !row.id || typeof row.label !== "string" || !row.label) continue;
    const tone = row.tone === "success" || row.tone === "primary" || row.tone === "warning" || row.tone === "muted"
      ? row.tone : undefined;
    out.push({
      id: row.id,
      label: row.label,
      ...(typeof row.description === "string" && row.description ? { description: row.description } : {}),
      ...(tone ? { tone } : {}),
      ...(typeof row.inline === "boolean" ? { inline: row.inline } : {}),
    });
  }
  return out;
}

function scheduleFlush() {
  if (flushTimer !== null) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    flushMissing();
  }, 0);
}

function flushMissing() {
  reconcileOwners();
  const sources = scopeMetaSources();
  if (!sources.length || !requestedScopes.size) return;
  const now = Date.now();
  for (const source of sources) {
    const cache = getOrCreateCache(source);
    const missing: { dimension: string; value: string }[] = [];
    for (const [key, entry] of requestedScopes.entries()) {
      if (cache.pending.has(key)) continue;
      const retryAt = cache.failedUntil.get(key) ?? 0;
      if (retryAt > now) continue;
      if (cache.values.has(key) && !cache.stale.has(key)) continue;
      missing.push(entry.scope);
    }
    if (!missing.length) continue;
    // 分片满足插件单次查询上限（如 akasha 200 项与 64 KiB 载荷限制）。
    for (let offset = 0; offset < missing.length; offset += BATCH_SIZE) {
      const batch = missing.slice(offset, offset + BATCH_SIZE);
      const keys = batch.map(scopeKeyOf);
      keys.forEach((key) => cache.pending.add(key));
      const { controller } = cache;
      const revision = source.pluginRevision;
      // 用 Promise.resolve().then 包装，防止 provider.load 同步抛错中断 flush 并卡住 pending。
      void Promise.resolve().then(() => source.provider.load(batch, source.context(controller.signal))).then(
        (result) => {
          const current = ownerCaches.get(source.pluginId);
          if (!current || current.revision !== revision || controller.signal.aborted) return;
          keys.forEach((key) => current.pending.delete(key));
          const record = result && typeof result === "object" && !Array.isArray(result) ? result : {};
          for (const key of keys) {
            if (!requestedScopes.has(key)) continue;
            current.values.set(key, normalizeItems((record as Record<string, unknown>)[key]));
            current.stale.delete(key);
            current.failedUntil.delete(key);
          }
          emit();
        },
        (error: unknown) => {
          const current = ownerCaches.get(source.pluginId);
          if (!current || current.revision !== revision || controller.signal.aborted) return;
          const retryAt = Date.now() + RETRY_COOLDOWN_MS;
          keys.forEach((key) => {
            current.pending.delete(key);
            if (requestedScopes.has(key)) current.failedUntil.set(key, retryAt);
          });
          console.warn(`[scope.meta] load failed for ${source.pluginId}`, error);
          emit();
        },
      );
    }
  }
}

function retainScope(dimension: string, value: string): () => void {
  const key = `${dimension}:${value}`;
  const existing = requestedScopes.get(key);
  if (existing) {
    existing.refs += 1;
  } else {
    requestedScopes.set(key, { scope: { dimension, value }, refs: 1 });
  }
  scheduleFlush();
  return () => {
    const entry = requestedScopes.get(key);
    if (!entry) return;
    entry.refs -= 1;
    if (entry.refs > 0) return;
    requestedScopes.delete(key);
    for (const cache of ownerCaches.values()) {
      cache.values.delete(key);
      cache.stale.delete(key);
      cache.pending.delete(key);
      cache.failedUntil.delete(key);
    }
  };
}

function readRecords(scope: { dimension: string; value: string } | null): ScopeMetaRecord[] {
  if (!scope) return [];
  const key = scopeKeyOf(scope);
  const sources = scopeMetaSources();
  const records: ScopeMetaRecord[] = [];
  for (const source of sources) {
    const cache = ownerCaches.get(source.pluginId);
    let items: readonly ScopeMetaItem[] | undefined = cache?.revision === source.pluginRevision
      ? cache.values.get(key)
      : undefined;
    if (items === undefined && source.provider.peek) {
      try {
        const peeked = source.provider.peek(scope);
        if (peeked !== undefined) items = normalizeItems(peeked);
      } catch {
        // 同步快照失败不影响后续正式查询。
      }
    }
    if (!items?.length) continue;
    for (const item of items) {
      records.push({ owner: source.pluginId, key: `${source.pluginId}:${item.id}`, item });
    }
  }
  return records;
}

function subscribe(listener: () => void): () => void {
  ensureSubscribed();
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** 订阅单个 scope 的结构化元信息（按 owner/revision 隔离、引用计数回收、自动去重与分片批量查询）。 */
export function useScopeMeta(scope: { dimension: string; value: string } | null): readonly ScopeMetaRecord[] {
  const version = useSyncExternalStore(subscribe, () => storeVersion, () => 0);
  const dimension = scope?.dimension ?? "";
  const value = scope?.value ?? "";
  useEffect(() => {
    if (!dimension || !value) return;
    ensureSubscribed();
    return retainScope(dimension, value);
  }, [dimension, value]);
  return useMemo(() => {
    void version;
    return dimension && value ? readRecords({ dimension, value }) : [];
  }, [dimension, value, version]);
}
