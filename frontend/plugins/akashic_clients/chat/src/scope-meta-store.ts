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
  pending: Set<string>;
  failed: Set<string>;
  controller: AbortController;
}

const BATCH_SIZE = 100;
const ownerCaches = new Map<string, OwnerCache>();
const requestedScopes = new Map<string, { dimension: string; value: string }>();
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
    cache.failed.clear();
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
    pending: new Set(),
    failed: new Set(),
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
  for (const source of sources) {
    const cache = getOrCreateCache(source);
    const missing: { dimension: string; value: string }[] = [];
    for (const [key, scope] of requestedScopes.entries()) {
      if (cache.values.has(key) || cache.pending.has(key) || cache.failed.has(key)) continue;
      missing.push(scope);
    }
    if (!missing.length) continue;
    // 分片满足插件单次查询上限（如 akasha 200 项与 64 KiB 载荷限制）。
    for (let offset = 0; offset < missing.length; offset += BATCH_SIZE) {
      const batch = missing.slice(offset, offset + BATCH_SIZE);
      const keys = batch.map(scopeKeyOf);
      keys.forEach((key) => cache.pending.add(key));
      const { controller } = cache;
      const revision = source.pluginRevision;
      void source.provider.load(batch, source.context(controller.signal)).then(
        (result) => {
          const current = ownerCaches.get(source.pluginId);
          if (!current || current.revision !== revision || controller.signal.aborted) return;
          keys.forEach((key) => current.pending.delete(key));
          const record = result && typeof result === "object" && !Array.isArray(result) ? result : {};
          for (const key of keys) {
            current.values.set(key, normalizeItems((record as Record<string, unknown>)[key]));
            current.failed.delete(key);
          }
          emit();
        },
        (error: unknown) => {
          const current = ownerCaches.get(source.pluginId);
          if (!current || current.revision !== revision || controller.signal.aborted) return;
          keys.forEach((key) => {
            current.pending.delete(key);
            current.failed.add(key);
          });
          console.warn(`[scope.meta] load failed for ${source.pluginId}`, error);
          emit();
        },
      );
    }
  }
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

/** 订阅单个 scope 的结构化元信息（按 owner/revision 隔离、自动去重与分片批量查询）。 */
export function useScopeMeta(scope: { dimension: string; value: string } | null): readonly ScopeMetaRecord[] {
  const version = useSyncExternalStore(subscribe, () => storeVersion, () => 0);
  const dimension = scope?.dimension ?? "";
  const value = scope?.value ?? "";
  useEffect(() => {
    if (!dimension || !value) return;
    ensureSubscribed();
    const key = `${dimension}:${value}`;
    requestedScopes.set(key, { dimension, value });
    scheduleFlush();
  }, [dimension, value, version]);
  return useMemo(() => {
    void version;
    return dimension && value ? readRecords({ dimension, value }) : [];
  }, [dimension, value, version]);
}
