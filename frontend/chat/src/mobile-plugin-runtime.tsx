import React, { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { MobilePluginQueryQueue } from "./mobile-plugin-query-queue";
import { MobilePluginResultCache } from "./mobile-plugin-result-cache";

export type MobilePluginSlotName =
  | "turn.before_reasoning"
  | "turn.before_tool"
  | "turn.after_answer"
  | "drawer.panel"
  | "dashboard.main";

export interface MobilePluginContext {
  slot: MobilePluginSlotName;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
  capabilities: {
    queryTransports: readonly ("inline" | "https")[];
  };
  query(
    method: string,
    payload?: Record<string, unknown>,
    options?: MobilePluginQueryOptions,
  ): Promise<Record<string, unknown>>;
}

export interface MobilePluginQueryOptions {
  cache?: "none" | "immutable";
  transport?: "inline" | "https";
}

export interface MobilePluginRenderer {
  mount(host: HTMLElement, context: MobilePluginContext): void | (() => void);
}

export interface MobilePluginDefinition {
  slots: Partial<Record<Exclude<MobilePluginSlotName, "dashboard.main">, MobilePluginRenderer>>;
  dashboard?: MobilePluginRenderer;
}

export interface MobilePluginCatalog {
  catalogRevision: string;
  updating: boolean;
  error?: string;
  plugins: MobilePluginCatalogItem[];
}

export interface MobilePluginCatalogItem {
  id: string;
  revision: string;
  moduleUrl: string;
  stylesheetUrl?: string;
  navigation?: {
    label: string;
    description: string;
  };
  slots: Exclude<MobilePluginSlotName, "dashboard.main">[];
}

export interface MobilePluginResult {
  requestId: string;
  resultJson?: string;
  error?: string;
}

export interface MobilePluginDashboardEntry {
  id: string;
  label: string;
  description: string;
}

const definitions = new Map<string, {
  revision: string;
  definition: MobilePluginDefinition;
}>();
const styleNodes = new Map<string, HTMLLinkElement>();
const listeners = new Set<() => void>();
interface PendingQuery {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
  ownerId: string;
  timeout: number;
  cacheKey?: string;
  slot: MobilePluginSlotName;
  started: boolean;
  abort?: AbortController;
  send: () => void;
}

const pendingQueries = new MobilePluginQueryQueue<PendingQuery>(
  (request) => isInteractiveSlot(request.slot),
);
const immutableResults = new MobilePluginResultCache();
let catalog: MobilePluginCatalog = {
  catalogRevision: "",
  updating: true,
  plugins: [],
};
let registryVersion = 0;
let activeRevision = "";
let activation: Promise<void> = Promise.resolve();
const quarantinedRevisions = new Map<string, Error>();
const MODULE_LOAD_TIMEOUT_MS = 5_000;
const SLOT_NAMES = new Set<Exclude<MobilePluginSlotName, "dashboard.main">>([
  "turn.before_reasoning",
  "turn.before_tool",
  "turn.after_answer",
  "drawer.panel",
]);

function emitChange() {
  registryVersion += 1;
  listeners.forEach((listener) => listener());
}

export function receiveMobilePluginCatalog(next: MobilePluginCatalog): Promise<void> {
  const revisionChanged = next.catalogRevision !== catalog.catalogRevision;
  if (revisionChanged) immutableResults.clear();
  catalog = next;
  if (next.updating || revisionChanged) rejectAllPending("插件界面正在更新");
  emitChange();
  if (next.updating || next.error || next.catalogRevision === activeRevision) {
    return Promise.resolve();
  }
  const quarantined = quarantinedRevisions.get(next.catalogRevision);
  if (quarantined) return Promise.reject(quarantined);
  activation = activation.catch(() => undefined).then(async () => {
    try {
      if (await activateCatalog(next)) activeRevision = next.catalogRevision;
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error("插件界面加载失败");
      quarantinedRevisions.set(next.catalogRevision, normalized);
      if (catalog.catalogRevision === next.catalogRevision) {
        catalog = { ...catalog, error: normalized.message };
        emitChange();
      }
      throw normalized;
    }
  });
  return activation;
}

/** 通过桌面适配器加载同一份内容寻址插件界面。 */
export async function loadWebPluginCatalog(signal?: AbortSignal): Promise<void> {
  const response = await fetch("/api/chat/plugin-ui/catalog", { signal });
  if (!response.ok) throw new Error(await webPluginError(response));
  await receiveMobilePluginCatalog(parseWebPluginCatalog(await response.json()));
}

function parseWebPluginCatalog(value: unknown): MobilePluginCatalog {
  const raw = requireRecord(value, "plugin catalog");
  const revision = requireString(raw.catalog_revision, "catalog_revision");
  if (!/^[0-9a-f]{64}$/.test(revision)) throw new Error("插件目录版本无效");
  const rawItems = raw.items;
  if (!Array.isArray(rawItems)) throw new Error("插件目录列表无效");
  const plugins = rawItems.map((value, index) => {
    const item = requireRecord(value, `plugins[${index}]`);
    const id = requireString(item.id, `plugins[${index}].id`);
    const pluginRevision = requireString(item.revision, `plugins[${index}].revision`);
    const moduleSha256 = requireDigest(item.module_sha256, `plugins[${index}].module_sha256`);
    const stylesheetSha256 = item.stylesheet_sha256 === null
      ? undefined
      : requireDigest(item.stylesheet_sha256, `plugins[${index}].stylesheet_sha256`);
    const slots = requireSlots(item.slots, index);
    const navigation = item.navigation === null
      ? undefined
      : parseNavigation(item.navigation, index);
    return {
      id,
      revision: pluginRevision,
      moduleUrl: webPluginAssetUrl(id, pluginRevision, "module", moduleSha256),
      stylesheetUrl: stylesheetSha256
        ? webPluginAssetUrl(id, pluginRevision, "stylesheet", stylesheetSha256)
        : undefined,
      navigation,
      slots,
    } satisfies MobilePluginCatalogItem;
  });
  if (new Set(plugins.map((plugin) => plugin.id)).size !== plugins.length) {
    throw new Error("插件目录包含重复 ID");
  }
  return { catalogRevision: revision, updating: false, plugins };
}

function webPluginAssetUrl(
  pluginId: string,
  pluginRevision: string,
  kind: "module" | "stylesheet",
  sha256: string,
): string {
  const query = new URLSearchParams({ plugin_id: pluginId, plugin_revision: pluginRevision, kind, sha256 });
  return `/api/chat/plugin-ui/asset?${query}`;
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} 无效`);
  return value as Record<string, unknown>;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) throw new Error(`${label} 无效`);
  return value;
}

function requireDigest(value: unknown, label: string): string {
  const digest = requireString(value, label);
  if (!/^[0-9a-f]{64}$/.test(digest)) throw new Error(`${label} 无效`);
  return digest;
}

function requireSlots(
  value: unknown,
  pluginIndex: number,
): Exclude<MobilePluginSlotName, "dashboard.main">[] {
  if (!Array.isArray(value)) throw new Error(`plugins[${pluginIndex}].slots 无效`);
  return value.map((slot, slotIndex) => {
    if (typeof slot !== "string" || !SLOT_NAMES.has(slot as Exclude<MobilePluginSlotName, "dashboard.main">)) {
      throw new Error(`plugins[${pluginIndex}].slots[${slotIndex}] 无效`);
    }
    return slot as Exclude<MobilePluginSlotName, "dashboard.main">;
  });
}

function parseNavigation(value: unknown, pluginIndex: number): MobilePluginCatalogItem["navigation"] {
  const navigation = requireRecord(value, `plugins[${pluginIndex}].navigation`);
  return {
    label: requireString(navigation.label, `plugins[${pluginIndex}].navigation.label`),
    description: requireString(navigation.description, `plugins[${pluginIndex}].navigation.description`),
  };
}

async function activateCatalog(next: MobilePluginCatalog): Promise<boolean> {
  const loaded = await Promise.all(next.plugins.map(async (plugin) => {
    const module = await withDeadline(
      import(/* @vite-ignore */ plugin.moduleUrl) as Promise<{ default?: unknown }>,
      MODULE_LOAD_TIMEOUT_MS,
      `插件界面加载超时: ${plugin.id}`,
    );
    return [plugin, parseDefinition(module.default, plugin)] as const;
  }));
  if (catalog.catalogRevision !== next.catalogRevision || catalog.updating) return false;

  const nextDefinitions = new Map<string, {
    revision: string;
    definition: MobilePluginDefinition;
  }>();
  const nextStyles = new Map<string, HTMLLinkElement>();
  for (const [plugin, definition] of loaded) {
    nextDefinitions.set(plugin.id, { revision: plugin.revision, definition });
    if (plugin.stylesheetUrl) {
      const node = document.createElement("link");
      node.rel = "stylesheet";
      node.href = plugin.stylesheetUrl;
      node.dataset.mobilePlugin = plugin.id;
      nextStyles.set(plugin.id, node);
    }
  }
  styleNodes.forEach((node) => node.remove());
  definitions.clear();
  nextDefinitions.forEach((definition, id) => definitions.set(id, definition));
  styleNodes.clear();
  nextStyles.forEach((node, id) => {
    document.head.appendChild(node);
    styleNodes.set(id, node);
  });
  emitChange();
  return true;
}

function withDeadline<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => { window.clearTimeout(timeout); resolve(value); },
      (error: unknown) => { window.clearTimeout(timeout); reject(error); },
    );
  });
}

function parseDefinition(value: unknown, plugin: MobilePluginCatalogItem): MobilePluginDefinition {
  if (!value || typeof value !== "object") {
    throw new Error(`插件界面必须默认导出定义对象: ${plugin.id}`);
  }
  const raw = value as { slots?: unknown; dashboard?: unknown };
  const slots = raw.slots ?? {};
  if (!slots || typeof slots !== "object" || Array.isArray(slots)) {
    throw new Error(`插件界面 slots 无效: ${plugin.id}`);
  }
  for (const [name, renderer] of Object.entries(slots)) {
    if (!SLOT_NAMES.has(name as Exclude<MobilePluginSlotName, "dashboard.main">)) {
      throw new Error(`插件界面 slot 无效: ${plugin.id}.${name}`);
    }
    if (!isRenderer(renderer)) throw new Error(`插件界面 renderer 无效: ${plugin.id}.${name}`);
  }
  const declaredSlots = Object.keys(slots).sort().join("|");
  if (declaredSlots !== [...plugin.slots].sort().join("|")) {
    throw new Error(`插件界面 slots 与 catalog 不一致: ${plugin.id}`);
  }
  const dashboard = raw.dashboard;
  if (dashboard !== undefined && !isRenderer(dashboard)) {
    throw new Error(`插件界面 dashboard renderer 无效: ${plugin.id}`);
  }
  if ((dashboard !== undefined) !== (plugin.navigation !== undefined)) {
    throw new Error(`插件界面 dashboard 与 catalog navigation 不一致: ${plugin.id}`);
  }
  return {
    slots: slots as MobilePluginDefinition["slots"],
    dashboard: dashboard as MobilePluginRenderer | undefined,
  };
}

function isRenderer(value: unknown): value is MobilePluginRenderer {
  return !!value && typeof value === "object" && typeof (value as { mount?: unknown }).mount === "function";
}

export function useMobilePluginDashboards(): MobilePluginDashboardEntry[] {
  useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  return catalog.plugins.flatMap((plugin) => plugin.navigation
    ? [{ id: plugin.id, ...plugin.navigation }]
    : []);
}

export function MobilePluginDashboard({ pluginId }: { pluginId: string }) {
  useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  const plugin = catalog.plugins.find((item) => item.id === pluginId);
  if (!plugin?.navigation) return null;
  if (catalog.error) return <div className="mobile-plugin-host mobile-plugin-host--error">{catalog.error}</div>;
  const loaded = definitions.get(pluginId);
  const definition = loaded?.revision === plugin.revision ? loaded.definition : undefined;
  if (!definition?.dashboard) {
    return <div className="mobile-plugin-host mobile-plugin-host--loading">正在加载插件界面…</div>;
  }
  return (
    <MountedPlugin
      pluginId={pluginId}
      pluginRevision={plugin.revision}
      renderer={definition.dashboard}
      context={{ slot: "dashboard.main" }}
    />
  );
}

export function receiveMobilePluginResult(response: MobilePluginResult) {
  const request = pendingQueries.get(response.requestId);
  if (!request) return;
  completePending(response.requestId, request);
  if (response.error) {
    request.reject(new Error(response.error));
    return;
  }
  try {
    const resultJson = response.resultJson ?? "{}";
    const result = JSON.parse(resultJson) as Record<string, unknown>;
    if (request.cacheKey && Object.values(result).some((value) => value !== null)) {
      immutableResults.set(request.cacheKey, resultJson);
    }
    request.resolve(result);
  } catch (error) {
    request.reject(error instanceof Error ? error : new Error("插件响应 JSON 无效"));
  }
}

export function MobilePluginSlot({
  name,
  sessionId,
  messageId,
  turnId,
  block,
}: {
  name: Exclude<MobilePluginSlotName, "dashboard.main">;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
}) {
  const version = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  const renderers = catalog.plugins.flatMap((plugin) => {
    const loaded = definitions.get(plugin.id);
    const renderer = loaded?.revision === plugin.revision
      ? loaded.definition.slots[name]
      : undefined;
    return renderer ? [{ plugin, renderer }] : [];
  });
  return renderers.length ? (
    <div className="mobile-plugin-slot" data-slot={name} data-version={version}>
      {renderers.map(({ plugin, renderer }) => (
        <ViewportMountedPlugin
          key={`${plugin.id}:${plugin.revision}:${name}`}
          pluginId={plugin.id}
          pluginRevision={plugin.revision}
          renderer={renderer}
          context={{ slot: name, sessionId, messageId, turnId, block }}
        />
      ))}
    </div>
  ) : null;
}

function ViewportMountedPlugin(props: React.ComponentProps<typeof MountedPlugin>) {
  const [visible, setVisible] = useState(false);
  const markerRef = React.useRef<HTMLDivElement>(null);
  useEffect(() => {
    const marker = markerRef.current;
    if (!marker || visible) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) setVisible(true);
    }, { rootMargin: "100% 0px" });
    observer.observe(marker);
    return () => observer.disconnect();
  }, [visible]);
  if (visible) return <MountedPlugin {...props} />;
  return <div ref={markerRef} className="mobile-plugin-host" data-plugin={props.pluginId} />;
}

function MountedPlugin({
  pluginId,
  pluginRevision,
  renderer,
  context,
}: {
  pluginId: string;
  pluginRevision: string;
  renderer: MobilePluginRenderer;
  context: Omit<MobilePluginContext, "query" | "capabilities">;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const ownerIdRef = React.useRef(createOwnerId());
  const { block, messageId, sessionId, slot, turnId } = context;
  const blockRevision = block === undefined ? undefined : JSON.stringify(block);
  const stableBlock = useMemo(
    () => blockRevision === undefined ? undefined : JSON.parse(blockRevision) as unknown,
    [blockRevision],
  );
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const ownerId = ownerIdRef.current;
    let cleanup: void | (() => void);
    try {
      cleanup = renderer.mount(host, {
        slot,
        sessionId,
        messageId,
        turnId,
        block: stableBlock,
        capabilities: {
          queryTransports: ["inline", "https"],
        },
        query(method, payload = {}, options = {}) {
          const requestId = createRequestId();
          return new Promise((resolve, reject) => {
            if (method.length < 1 || method.length > 256) {
              reject(new Error("插件方法名无效"));
              return;
            }
            let encoded: string;
            try {
              encoded = JSON.stringify(payload);
            } catch (error) {
              reject(error instanceof Error ? error : new Error("插件参数无法序列化"));
              return;
            }
            if (new TextEncoder().encode(encoded).byteLength > 64 * 1024) {
              reject(new Error("插件参数超过 64 KiB"));
              return;
            }
            const cacheKey = options.cache === "immutable"
              ? pluginQueryCacheKey(pluginId, pluginRevision, method, encoded, sessionId, turnId)
              : undefined;
            const cachedJson = cacheKey === undefined ? undefined : immutableResults.get(cacheKey);
            if (cachedJson !== undefined) {
              resolve(JSON.parse(cachedJson) as Record<string, unknown>);
              return;
            }
            const timeout = window.setTimeout(() => {
              const request = pendingQueries.get(requestId);
              if (!request) return;
              request.abort?.abort();
              window.AkashicNative?.cancelPluginUiOwner(request.ownerId);
              rejectOwnerPending(request.ownerId, "插件请求超时");
            }, 30_000);
            const abort = window.AkashicNative ? undefined : new AbortController();
            const request = {
              resolve,
              reject,
              ownerId,
              timeout,
              cacheKey,
              slot,
              started: false,
              abort,
              send: () => {
                if (window.AkashicNative) {
                  window.AkashicNative.queryPluginUi(
                    requestId,
                    ownerId,
                    slot,
                    sessionId ?? null,
                    turnId ?? null,
                    pluginId,
                    method,
                    encoded,
                    options.cache ?? "none",
                    options.transport ?? "inline",
                  );
                  return;
                }
                void queryWebPluginUi({
                  pluginId,
                  pluginRevision,
                  method,
                  payload,
                  slot,
                  sessionId,
                  turnId,
                  signal: abort!.signal,
                }).then(
                  (result) => receiveMobilePluginResult({ requestId, resultJson: JSON.stringify(result) }),
                  (error: unknown) => receiveMobilePluginResult({
                    requestId,
                    error: error instanceof Error ? error.message : "插件查询失败",
                  }),
                );
              },
            };
            try {
              pendingQueries.enqueue(requestId, request);
            } catch (error) {
              window.clearTimeout(timeout);
              reject(error instanceof Error ? error : new Error("插件请求无法入队"));
              return;
            }
            drainQueryQueue();
          });
        },
      });
    } catch (error) {
      host.textContent = error instanceof Error ? `插件界面错误：${error.message}` : "插件界面错误";
      host.classList.add("mobile-plugin-host--error");
    }
    return () => {
      window.AkashicNative?.cancelPluginUiOwner(ownerId);
      rejectOwnerPending(ownerId, "插件界面已卸载");
      try {
        cleanup?.();
      } catch (error) {
        console.error(`[plugin-ui] cleanup failed: ${pluginId}`, error);
      }
      host.replaceChildren();
    };
  }, [messageId, pluginId, pluginRevision, renderer, sessionId, slot, stableBlock, turnId]);
  return <div ref={hostRef} className="mobile-plugin-host" data-plugin={pluginId} />;
}

async function queryWebPluginUi({
  pluginId,
  pluginRevision,
  method,
  payload,
  slot,
  sessionId,
  turnId,
  signal,
}: {
  pluginId: string;
  pluginRevision: string;
  method: string;
  payload: Record<string, unknown>;
  slot: MobilePluginSlotName;
  sessionId?: string;
  turnId?: string;
  signal: AbortSignal;
}): Promise<Record<string, unknown>> {
  if (slot === "dashboard.main") throw new Error("Web 不开放插件独立面板");
  const response = await fetch("/api/chat/plugin-ui/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      plugin_id: pluginId,
      plugin_revision: pluginRevision,
      method,
      payload,
      slot,
      session_id: sessionId ?? null,
      turn_id: turnId ?? null,
    }),
    signal,
  });
  if (!response.ok) throw new Error(await webPluginError(response));
  return requireRecord(await response.json(), "插件响应");
}

async function webPluginError(response: Response): Promise<string> {
  const value = await response.json().catch(() => null);
  const detail = value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>).detail
    : undefined;
  return typeof detail === "string" ? detail : `插件请求失败 (${response.status})`;
}

function pluginQueryCacheKey(
  pluginId: string,
  pluginRevision: string,
  method: string,
  payloadJson: string,
  sessionId?: string,
  turnId?: string,
): string {
  return [pluginId, pluginRevision, method, sessionId ?? "", turnId ?? "", payloadJson].join("\n");
}

function rejectOwnerPending(ownerId: string, message: string) {
  const owned = pendingQueries.removeOwner(ownerId);
  for (const [, request] of owned) {
    window.clearTimeout(request.timeout);
    request.abort?.abort();
    request.reject(new Error(message));
  }
  drainQueryQueue();
}

function rejectAllPending(message: string) {
  const requests = pendingQueries.clear();
  for (const [, request] of requests) {
    window.clearTimeout(request.timeout);
    request.abort?.abort();
    request.reject(new Error(message));
  }
  drainQueryQueue();
}

function drainQueryQueue() {
  while (true) {
    const next = pendingQueries.startNext();
    if (!next) return;
    const [requestId, request] = next;
    try {
      request.send();
    } catch (error) {
      completePending(requestId, request, false);
      request.reject(error instanceof Error ? error : new Error("原生插件桥调用失败"));
    }
  }
}

function completePending(requestId: string, request: PendingQuery, shouldDrain = true) {
  if (pendingQueries.complete(requestId) !== request) {
    throw new Error("插件请求完成状态失配");
  }
  window.clearTimeout(request.timeout);
  if (shouldDrain) drainQueryQueue();
}

function isInteractiveSlot(slot: MobilePluginSlotName): boolean {
  return slot === "dashboard.main" || slot === "drawer.panel";
}

function createOwnerId(): string {
  return `owner:${createRequestId()}`;
}

function createRequestId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}
