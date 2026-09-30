import React, { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { createUuid as createRequestId } from "./browser-uuid.ts";
import { PluginUiQueryQueue } from "./plugin-ui-query-queue";
import { PluginUiResultCache } from "./plugin-ui-result-cache";

export type PluginUiSlotName =
  | "turn.before_reasoning"
  | "turn.before_tool"
  | "turn.after_answer"
  | "drawer.panel"
  | "dashboard.main";

export interface PluginUiContext {
  slot: PluginUiSlotName;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
  /** 挂载内的事实失效提示；不改变挂载身份。 */
  onInvalidate(callback: () => void): () => void;
  capabilities: {
    queryCacheModes?: readonly ("none" | "memory" | "immutable")[];
  };
  query(
    method: string,
    payload?: Record<string, unknown>,
    options?: PluginUiQueryOptions,
  ): Promise<Record<string, unknown>>;
}

export interface PluginUiQueryOptions {
  cache?: "none" | "memory" | "immutable";
}

export interface PluginUiRenderer {
  prefetch?(context: PluginUiContext): Promise<void>;
  mount(host: HTMLElement, context: PluginUiContext): void | (() => void);
}

export interface PluginUiDefinition {
  slots: Partial<Record<Exclude<PluginUiSlotName, "dashboard.main">, PluginUiRenderer>>;
  dashboard?: PluginUiRenderer;
}

export interface PluginUiCatalog {
  catalogRevision: string;
  updating: boolean;
  error?: string;
  plugins: PluginUiCatalogItem[];
}

export interface PluginUiCatalogItem {
  id: string;
  revision: string;
  moduleUrl: string;
  stylesheetUrl?: string;
  navigation?: {
    label: string;
    description: string;
  };
  slots: Exclude<PluginUiSlotName, "dashboard.main">[];
}

export interface PluginUiResult {
  requestId: string;
  resultJson?: string;
  error?: string;
}

export interface PluginUiDashboardEntry {
  id: string;
  label: string;
  description: string;
}

const definitions = new Map<string, {
  revision: string;
  definition: PluginUiDefinition;
}>();
const styleNodes = new Map<string, HTMLLinkElement>();
const listeners = new Set<() => void>();
interface PendingQuery {
  pluginId: string;
  method: string;
  sessionId?: string;
  messageId?: string;
  queuedAt: number;
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
  ownerId: string;
  timeout?: number;
  cacheKey?: string;
  sharedKey?: string;
  interactive: boolean;
  slot: PluginUiSlotName;
  started: boolean;
  abort: AbortController;
  send: () => void;
}

const pendingQueries = new PluginUiQueryQueue<PendingQuery>(
  (request) => request.interactive,
);
const immutableResults = new PluginUiResultCache();
const sharedQueries = new Map<string, {
  requestId: string;
  promise: Promise<Record<string, unknown>>;
  owners: Set<string>;
}>();
let catalog: PluginUiCatalog = {
  catalogRevision: "",
  updating: true,
  plugins: [],
};
let registryVersion = 0;
let activeRevision = "";
let activation: Promise<void> = Promise.resolve();
let staleCatalogRead: Promise<void> | null = null;
let staleRecoveryAttempted = false;
const quarantinedRevisions = new Map<string, Error>();
const MODULE_LOAD_TIMEOUT_MS = 5_000;
const SLOT_NAMES = new Set<Exclude<PluginUiSlotName, "dashboard.main">>([
  "turn.before_reasoning",
  "turn.before_tool",
  "turn.after_answer",
  "drawer.panel",
]);

function emitChange() {
  registryVersion += 1;
  listeners.forEach((listener) => listener());
}

export function receivePluginUiCatalog(next: PluginUiCatalog): Promise<void> {
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
  if (!response.ok) throw await webPluginError(response);
  await receivePluginUiCatalog(parseWebPluginCatalog(await response.json()));
}

function parseWebPluginCatalog(value: unknown): PluginUiCatalog {
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
    } satisfies PluginUiCatalogItem;
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
  // 远端对 asset URL 发 immutable 年缓存；dev 下插件资源可能被本地中间件替换，
  // 需要让 URL 每次加载都不同，否则浏览器直接命中远端旧缓存、本地改动不可见。
  if (import.meta.env.DEV) query.set("_dev", String(Date.now()));
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
): Exclude<PluginUiSlotName, "dashboard.main">[] {
  if (!Array.isArray(value)) throw new Error(`plugins[${pluginIndex}].slots 无效`);
  return value.map((slot, slotIndex) => {
    if (typeof slot !== "string" || !SLOT_NAMES.has(slot as Exclude<PluginUiSlotName, "dashboard.main">)) {
      throw new Error(`plugins[${pluginIndex}].slots[${slotIndex}] 无效`);
    }
    return slot as Exclude<PluginUiSlotName, "dashboard.main">;
  });
}

function parseNavigation(value: unknown, pluginIndex: number): PluginUiCatalogItem["navigation"] {
  const navigation = requireRecord(value, `plugins[${pluginIndex}].navigation`);
  return {
    label: requireString(navigation.label, `plugins[${pluginIndex}].navigation.label`),
    description: requireString(navigation.description, `plugins[${pluginIndex}].navigation.description`),
  };
}

async function activateCatalog(next: PluginUiCatalog): Promise<boolean> {
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
    definition: PluginUiDefinition;
  }>();
  const nextStyles = new Map<string, HTMLLinkElement>();
  for (const [plugin, definition] of loaded) {
    nextDefinitions.set(plugin.id, { revision: plugin.revision, definition });
    if (plugin.stylesheetUrl) {
      const node = document.createElement("link");
      node.rel = "stylesheet";
      node.href = plugin.stylesheetUrl;
      node.dataset.pluginUi = plugin.id;
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

function parseDefinition(value: unknown, plugin: PluginUiCatalogItem): PluginUiDefinition {
  if (!value || typeof value !== "object") {
    throw new Error(`插件界面必须默认导出定义对象: ${plugin.id}`);
  }
  const raw = value as { slots?: unknown; dashboard?: unknown };
  const slots = raw.slots ?? {};
  if (!slots || typeof slots !== "object" || Array.isArray(slots)) {
    throw new Error(`插件界面 slots 无效: ${plugin.id}`);
  }
  for (const [name, renderer] of Object.entries(slots)) {
    if (!SLOT_NAMES.has(name as Exclude<PluginUiSlotName, "dashboard.main">)) {
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
    slots: slots as PluginUiDefinition["slots"],
    dashboard: dashboard as PluginUiRenderer | undefined,
  };
}

function isRenderer(value: unknown): value is PluginUiRenderer {
  if (!value || typeof value !== "object") return false;
  const renderer = value as { mount?: unknown; prefetch?: unknown };
  return typeof renderer.mount === "function"
    && (renderer.prefetch === undefined || typeof renderer.prefetch === "function");
}

/** 宿主按插件名找能力；调用时保留目录中的完整 name@marketplace 身份。 */
function hostPlugins(pluginName: string): PluginUiCatalogItem[] {
  return catalog.updating ? [] : catalog.plugins.filter((plugin) => plugin.id.split("@")[0] === pluginName);
}

/** 宿主自身的导航按插件名组合能力；目录变化时返回新版本号以便重读。 */
export function usePluginUiCatalogVersion(): { version: number; installed: (pluginName: string) => boolean } {
  const version = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  return {
    version,
    installed: (pluginName) => hostPlugins(pluginName).length > 0,
  };
}

/** 宿主侧栏直接调用插件查询；插件缺失或目录更新中都作为普通错误返回。 */
export async function queryHostPlugin(
  pluginName: string,
  method: string,
  payload: Record<string, unknown> = {},
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  const matches = hostPlugins(pluginName);
  if (matches.length > 1) throw new Error(`存在多个同名插件，无法选择: ${matches.map((plugin) => plugin.id).join(", ")}`);
  const plugin = matches[0];
  if (!plugin) throw new Error(`插件未安装或正在更新: ${pluginName}`);
  return queryWebPluginUi({
    pluginId: plugin.id, pluginRevision: plugin.revision, method, payload, slot: "drawer.panel",
    signal: signal ?? new AbortController().signal,
  });
}

export function usePluginUiDashboards(): PluginUiDashboardEntry[] {
  useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  return catalog.plugins.flatMap((plugin) => plugin.navigation
    ? [{ id: plugin.id, ...plugin.navigation }]
    : []);
}

export function PluginUiDashboard({ pluginId }: { pluginId: string }) {
  useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  const plugin = catalog.plugins.find((item) => item.id === pluginId);
  if (!plugin?.navigation) return null;
  if (catalog.error) return <div className="plugin-ui-host plugin-ui-host--error">{catalog.error}</div>;
  const loaded = definitions.get(pluginId);
  const definition = loaded?.revision === plugin.revision ? loaded.definition : undefined;
  if (!definition?.dashboard) {
    return <div className="plugin-ui-host plugin-ui-host--loading">正在加载插件界面…</div>;
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

export function receivePluginUiResult(response: PluginUiResult) {
  const request = pendingQueries.get(response.requestId);
  if (!request) return;
  tracePluginQuery(response.requestId, request, response.error ? "failed" : "received");
  completePending(response.requestId, request);
  if (response.error) {
    request.reject(new Error(response.error));
    return;
  }
  try {
    const resultJson = response.resultJson ?? "{}";
    const result = JSON.parse(resultJson) as Record<string, unknown>;
    if (
      request.cacheKey
      && result.pending !== true
      && Object.values(result).some((value) => value !== null)
    ) {
      immutableResults.set(request.cacheKey, resultJson);
    }
    request.resolve(result);
  } catch (error) {
    request.reject(error instanceof Error ? error : new Error("插件响应 JSON 无效"));
  }
}

export function PluginUiSlot({
  name,
  sessionId,
  messageId,
  turnId,
  block,
  refreshToken,
  prefetch = false,
}: {
  name: Exclude<PluginUiSlotName, "dashboard.main">;
  prefetch?: boolean;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
  refreshToken?: string | number;
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
    return renderer && (!prefetch || renderer.prefetch) ? [{ plugin, renderer }] : [];
  });
  // 目录加载失败时保留本页已经挂载的卡片；新页面不会执行旧模块。
  const previous = React.useRef<typeof renderers>([]);
  const hadRenderer = React.useRef(false);
  const contextKey = JSON.stringify([name, prefetch, sessionId, messageId, turnId, block]);
  const previousContext = React.useRef(contextKey);
  const waiting = catalog.updating || !!catalog.error || catalog.plugins.some(plugin =>
    plugin.slots.includes(name) && definitions.get(plugin.id)?.revision !== plugin.revision);
  const keepPrevious = waiting && previousContext.current === contextKey && previous.current.length > 0
    && previous.current.every(({plugin}) => catalog.plugins.some(item => item.id === plugin.id));
  const shown = keepPrevious ? previous.current : renderers;
  if (!keepPrevious) { previous.current = renderers; previousContext.current = contextKey; }
  if (shown.length) hadRenderer.current = true;
  const notice = waiting ? catalog.error ? `插件界面暂不可用：${catalog.error}` : "插件界面正在更新…"
    : !shown.length && hadRenderer.current ? "本页插件界面已卸载或不再提供此展示。" : "";
  return shown.length || notice ? (
    <div className={prefetch ? "plugin-ui-prefetch" : "plugin-ui-slot"} data-slot={name} data-version={version}>
      {notice && !prefetch && <p role="status">{notice} <button type="button" onClick={() => window.location.reload()}>刷新页面</button></p>}
      {shown.map(({ plugin, renderer }) => (
        <ViewportMountedPlugin
          key={`${plugin.id}:${plugin.revision}:${name}`}
          pluginId={plugin.id}
          pluginRevision={plugin.revision}
          renderer={renderer}
          prefetch={prefetch}
          refreshToken={refreshToken}
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
  return <div ref={markerRef} className="plugin-ui-host" data-plugin={props.pluginId} />;
}

function MountedPlugin({
  pluginId,
  pluginRevision,
  renderer,
  context,
  refreshToken,
  prefetch = false,
}: {
  pluginId: string;
  pluginRevision: string;
  renderer: PluginUiRenderer;
  prefetch?: boolean;
  context: Omit<PluginUiContext, "query" | "capabilities" | "onInvalidate">;
  refreshToken?: string | number;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const ownerIdRef = React.useRef(createOwnerId());
  const invalidatorsRef = React.useRef<Set<() => void> | null>(null);
  const previousRefreshToken = React.useRef(refreshToken);
  const { block, messageId, sessionId, slot, turnId } = context;
  const blockRevision = block === undefined ? undefined : JSON.stringify(block);
  const stableBlock = useMemo(
    () => blockRevision === undefined ? undefined : JSON.parse(blockRevision) as unknown,
    [blockRevision],
  );
  // 先通知仍存活的挂载；身份切换已执行 cleanup，新挂载只需自己的首次读取。
  useEffect(() => {
    if (Object.is(previousRefreshToken.current, refreshToken)) return;
    previousRefreshToken.current = refreshToken;
    for (const callback of invalidatorsRef.current ?? []) {
      try { callback(); }
      catch (error) { console.error(`[plugin-ui] invalidation failed: ${pluginId}`, error); }
    }
  }, [pluginId, refreshToken]);
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const ownerId = ownerIdRef.current;
    const invalidators = new Set<() => void>();
    invalidatorsRef.current = invalidators;
    let cleanup: void | (() => void);
    try {
      const queryContext: PluginUiContext = {
        slot, sessionId, messageId, turnId, block: stableBlock,
        onInvalidate: (callback) => {
          invalidators.add(callback);
          return () => { invalidators.delete(callback); };
        },
        capabilities: { queryCacheModes: ["none", "memory", "immutable"] },
        query: (method, payload = {}, options = {}) => queryPlugin({
          pluginId, pluginRevision, ownerId, slot, sessionId, messageId, turnId, method, payload, options, prefetch,
        }),
      };
      if (prefetch) {
        void renderer.prefetch!(queryContext).catch((error: unknown) => {
          console.debug(`[plugin-ui] prefetch failed: ${pluginId}`, error);
        });
      } else {
        cleanup = renderer.mount(host, queryContext);
      }
    } catch (error) {
      host.textContent = error instanceof Error ? `插件界面错误：${error.message}` : "插件界面错误";
      host.classList.add("plugin-ui-host--error");
    }
    return () => {
      invalidators.clear();
      invalidatorsRef.current = null;
      releaseQueryOwner(ownerId);
      try {
        cleanup?.();
      } catch (error) {
        console.error(`[plugin-ui] cleanup failed: ${pluginId}`, error);
      }
      host.replaceChildren();
    };
  }, [messageId, pluginId, pluginRevision, renderer, sessionId, slot, stableBlock, turnId, prefetch]);
  return <div ref={hostRef} className="plugin-ui-host" data-plugin={pluginId} />;
}

/** 页面缓存查询拥有独立请求，折叠只释放订阅，不打断已经发出的读取。 */
function queryPlugin({ pluginId, pluginRevision, ownerId, slot, sessionId, messageId, turnId, method, payload, options, prefetch }: {
  pluginId: string; pluginRevision: string; ownerId: string; slot: PluginUiSlotName;
  sessionId?: string; messageId?: string; turnId?: string; method: string; payload: Record<string, unknown>;
  options: PluginUiQueryOptions; prefetch: boolean;
}): Promise<Record<string, unknown>> {
  // 1. 已读卡片可暂留；旧上下文不能发送请求或读取新版本缓存。
  const current = catalog.plugins.find(plugin => plugin.id === pluginId);
  if (catalog.updating || catalog.error || current?.revision !== pluginRevision) {
    return Promise.reject(new WebPluginError(409, "插件界面正在更新或暂不可用，请刷新页面。", "plugin_ui_stale_revision"));
  }
  // 参数只在插件调用边界校验；缓存沿用原有插件、消息与版本身份。
  if (method.length < 1 || method.length > 256) return Promise.reject(new Error("插件方法名无效"));
  let encoded: string;
  try {
    encoded = JSON.stringify(payload);
  } catch (error) {
    return Promise.reject(error instanceof Error ? error : new Error("插件参数无法序列化"));
  }
  if (new TextEncoder().encode(encoded).byteLength > 64 * 1024) {
    return Promise.reject(new Error("插件参数超过 64 KiB"));
  }
  const cacheKey = options.cache === "immutable" || options.cache === "memory"
    ? pluginQueryCacheKey(pluginId, pluginRevision, method, encoded, sessionId, turnId) : undefined;
  const cachedJson = cacheKey === undefined ? undefined : immutableResults.get(cacheKey);
  if (cachedJson !== undefined) return Promise.resolve(JSON.parse(cachedJson) as Record<string, unknown>);
  const sharedKey = options.cache === "memory" ? cacheKey : undefined;
  const shared = sharedKey === undefined ? undefined : sharedQueries.get(sharedKey);
  if (shared) {
    shared.owners.add(ownerId);
    const request = pendingQueries.get(shared.requestId)!;
    if (!prefetch && !request.started) request.interactive = true;
    drainQueryQueue();
    return shared.promise;
  }

  // 2. 同一内存缓存键共享一个有界请求。
  const requestId = createRequestId();
  const requestOwnerId = sharedKey === undefined ? ownerId : createOwnerId();
  const promise = new Promise<Record<string, unknown>>((resolve, reject) => {
    const abort = new AbortController();
    const request: PendingQuery = {
      resolve, reject, ownerId: requestOwnerId, cacheKey, sharedKey, slot, started: false, abort,
      pluginId, method, sessionId, messageId, queuedAt: performance.now(),
      interactive: isInteractiveSlot(slot) || (sharedKey !== undefined && !prefetch),
      send: () => {
        tracePluginQuery(requestId, request, "sent");
        request.timeout = window.setTimeout(() => {
          tracePluginQuery(requestId, request, "timeout");
          rejectOwnerPending(requestOwnerId, "插件请求超时");
        }, 30_000);
        void queryWebPluginUi({ pluginId, pluginRevision, method, payload, slot, sessionId, turnId,
          signal: abort.signal }).then(
          (result) => receivePluginUiResult({ requestId, resultJson: JSON.stringify(result) }),
          (error: unknown) => {
            const pending = pendingQueries.get(requestId);
            if (!pending) return;
            tracePluginQuery(requestId, pending, "failed");
            completePending(requestId, pending);
            pending.reject(error instanceof Error ? error : new Error("插件查询失败"));
          },
        );
      },
    };
    try {
      pendingQueries.enqueue(requestId, request);
      tracePluginQuery(requestId, request, "queued");
    } catch (error) {
      reject(error instanceof Error ? error : new Error("插件请求无法入队"));
    }
  });
  if (sharedKey !== undefined && pendingQueries.get(requestId)) {
    sharedQueries.set(sharedKey, { requestId, promise, owners: new Set([ownerId]) });
  }
  drainQueryQueue();
  return promise;
}

/** 未发出的孤立预取直接撤销；在途读取最多继续到现有传输期限。 */
function releaseQueryOwner(ownerId: string) {
  for (const shared of sharedQueries.values()) {
    shared.owners.delete(ownerId);
    const request = pendingQueries.get(shared.requestId)!;
    if (shared.owners.size === 0 && !request.started) {
      completePending(shared.requestId, request, false);
      request.reject(new Error("插件界面已卸载"));
    }
  }
  rejectOwnerPending(ownerId, "插件界面已卸载");
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
  slot: PluginUiSlotName;
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
  if (!response.ok) {
    const error = await webPluginError(response);
    if (response.status === 409 && error.code === "plugin_ui_stale_revision") {
      await refreshStaleCatalog();
    }
    throw error;
  }
  if (!staleCatalogRead && !catalog.updating && !catalog.error
      && catalog.plugins.some(item => item.id === pluginId && item.revision === pluginRevision)) {
    staleRecoveryAttempted = false;
  }
  return requireRecord(await response.json(), "插件响应");
}

class WebPluginError extends Error {
  constructor(readonly status: number, message: string, readonly code?: string) { super(message); }
}

/** 同一恢复周期只读取一次正式目录，不重放旧模块查询。 */
async function refreshStaleCatalog(): Promise<void> {
  if (staleCatalogRead) return staleCatalogRead;
  if (staleRecoveryAttempted) {
    throw new WebPluginError(409, "插件界面仍未恢复，请刷新页面。", "plugin_ui_stale_revision");
  }
  staleRecoveryAttempted = true;
  const read = loadWebPluginCatalog(AbortSignal.timeout(10_000));
  staleCatalogRead = read;
  try { await read; }
  catch (error) {
    throw new WebPluginError(409, `插件界面更新失败：${error instanceof Error ? error.message : String(error)}。请刷新页面。`, "plugin_ui_stale_revision");
  }
  finally { if (staleCatalogRead === read) staleCatalogRead = null; }
}

async function webPluginError(response: Response): Promise<WebPluginError> {
  const value = await response.json().catch(() => null);
  const detail = value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>).detail : undefined;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const body = detail as Record<string, unknown>;
    if (typeof body.code === "string" && typeof body.message === "string") {
      return new WebPluginError(response.status, body.message, body.code);
    }
  }
  return new WebPluginError(response.status, typeof detail === "string" ? detail : `插件请求失败 (${response.status})`);
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
    if (request.sharedKey) sharedQueries.delete(request.sharedKey);
    window.clearTimeout(request.timeout);
    request.abort.abort();
    request.reject(new Error(message));
  }
  drainQueryQueue();
}

function rejectAllPending(message: string) {
  const requests = pendingQueries.clear();
  sharedQueries.clear();
  for (const [, request] of requests) {
    window.clearTimeout(request.timeout);
    request.abort.abort();
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
      request.reject(error instanceof Error ? error : new Error("插件请求失败"));
    }
  }
}

function completePending(requestId: string, request: PendingQuery, shouldDrain = true) {
  if (pendingQueries.complete(requestId) !== request) {
    throw new Error("插件请求完成状态失配");
  }
  if (request.sharedKey) sharedQueries.delete(request.sharedKey);
  window.clearTimeout(request.timeout);
  if (shouldDrain) drainQueryQueue();
}

function isInteractiveSlot(slot: PluginUiSlotName): boolean {
  return slot === "dashboard.main" || slot === "drawer.panel";
}

function createOwnerId(): string {
  return `owner:${createRequestId()}`;
}

/** 复用原生已有的结构日志入口，只记录请求身份和阶段耗时。 */
function tracePluginQuery(requestId: string, request: PendingQuery, phase: string) {
  console.log(`[akashic-trace] ${JSON.stringify({
    event: `webui.plugin_query.${phase}`, request_id: requestId, owner_id: request.ownerId,
    plugin_id: request.pluginId, method: request.method, slot: request.slot,
    session_id: request.sessionId, message_id: request.messageId, wall_ms: Date.now(),
    elapsed_ms: Math.round((performance.now() - request.queuedAt) * 1000) / 1000,
  })}`);
}
