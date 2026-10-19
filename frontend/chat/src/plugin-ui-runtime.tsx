import React, { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { createUuid as createRequestId } from "./browser-uuid.ts";
import { PluginUiQueryQueue } from "./plugin-ui-query-queue";
import { PluginUiResultCache } from "./plugin-ui-result-cache";
import { composerControls } from "./composer-contributions";
import type { ScopeFields } from "./scope-fields";
import { openInWorkbench, subscribeWorkbenchIntents, workbenchAccepts } from "./shell-workbench";

import type { ComposerControls, ComposerSuggestProvider, NavigationControls, NavigationForm, NavigationFormControls, NavigationGroupsProvider, NavigationNotice, PluginUiDefinition, PluginUiHostContext, PluginUiMountSlotName, PluginUiQueryOptions, PluginUiSlotFacts, PluginUiSlotName, ScopeFieldControls } from "@akashic/chat-ui-v1";

// 插槽合同由 @akashic/chat-ui-v1 发布，宿主与插件共用同一份类型。
export type { ComposerSuggestProvider, ComposerSuggestion, PluginUiContext, PluginUiDefinition, PluginUiQueryOptions, PluginUiRenderer, PluginUiSlotName } from "@akashic/chat-ui-v1";

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
  slots: PluginUiSlotName[];
}

export interface PluginUiResult {
  requestId: string;
  resultJson?: string;
  error?: string;
}

const definitions = new Map<string, {
  revision: string;
  definition: PluginUiDefinition;
}>();
const styleNodes = new Map<string, HTMLLinkElement>();
const listeners = new Set<() => void>();
const shownListeners = new Set<() => void>();
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
const SLOT_NAMES = new Set<PluginUiSlotName>([
  "turn.before_reasoning",
  "turn.after_answer",
  "drawer.panel",
  "tool.result",
  "message.attachment",
  "chat.notice",
  "header.end",
  "composer.toolbar",
  "composer.status",
  "composer.suggest",
  "scope.fields",
  "scope.badge",
  "navigation.groups",
]);

/** 目录或模块变化时通知；宿主内的数据型扩展点据此对齐自己的缓存。 */
export function subscribePluginUiCatalog(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

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
    return {
      id,
      revision: pluginRevision,
      moduleUrl: webPluginAssetUrl(id, pluginRevision, "module", moduleSha256),
      stylesheetUrl: stylesheetSha256
        ? webPluginAssetUrl(id, pluginRevision, "stylesheet", stylesheetSha256)
        : undefined,
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
): PluginUiSlotName[] {
  if (!Array.isArray(value)) throw new Error(`plugins[${pluginIndex}].slots 无效`);
  return value.map((slot, slotIndex) => {
    if (typeof slot !== "string" || !SLOT_NAMES.has(slot as PluginUiSlotName)) {
      throw new Error(`plugins[${pluginIndex}].slots[${slotIndex}] 无效`);
    }
    return slot as PluginUiSlotName;
  });
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
  const retired = new Set([...definitions].filter(([id, { revision }]) => nextDefinitions.get(id)?.revision !== revision)
    .map(([id, { revision }]) => JSON.stringify([id, revision])));
  definitions.clear();
  nextDefinitions.forEach((definition, id) => definitions.set(id, definition));
  retirePushes(retired);
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
  const raw = value as { slots?: unknown };
  const slots = raw.slots ?? {};
  if (!slots || typeof slots !== "object" || Array.isArray(slots)) {
    throw new Error(`插件界面 slots 无效: ${plugin.id}`);
  }
  for (const [name, renderer] of Object.entries(slots)) {
    if (!SLOT_NAMES.has(name as PluginUiSlotName)) {
      throw new Error(`插件界面 slot 无效: ${plugin.id}.${name}`);
    }
    const valid = name === "composer.suggest" ? isSuggestProvider(renderer)
      : name === "navigation.groups" ? isGroupsProvider(renderer) : isRenderer(renderer);
    if (!valid) {
      throw new Error(`插件界面 renderer 无效: ${plugin.id}.${name}`);
    }
  }
  const declaredSlots = Object.keys(slots).sort().join("|");
  if (declaredSlots !== [...plugin.slots].sort().join("|")) {
    throw new Error(`插件界面 slots 与 catalog 不一致: ${plugin.id}`);
  }
  return { slots: slots as PluginUiDefinition["slots"] };
}

function isRenderer(value: unknown): value is HostRenderer {
  if (!value || typeof value !== "object") return false;
  const renderer = value as { mount?: unknown; prefetch?: unknown; accepts?: unknown };
  return typeof renderer.mount === "function"
    && (renderer.accepts === undefined || typeof renderer.accepts === "function")
    && (renderer.prefetch === undefined || typeof renderer.prefetch === "function");
}

function isSuggestProvider(value: unknown): value is ComposerSuggestProvider {
  if (!value || typeof value !== "object") return false;
  const provider = value as Partial<Record<keyof ComposerSuggestProvider, unknown>>;
  return typeof provider.label === "string" && typeof provider.empty === "string" && typeof provider.load === "function"
    && (provider.prefix === "/" || provider.prefix === "@") && Array.isArray(provider.sigils)
    && provider.sigils.every((sigil) => sigil === "/" || sigil === "@");
}

function isGroupsProvider(value: unknown): value is NavigationGroupsProvider {
  if (!value || typeof value !== "object") return false;
  const provider = value as Partial<Record<keyof NavigationGroupsProvider, unknown>>;
  const form = (item: unknown) => item === undefined || (typeof item === "object" && item !== null
    && typeof (item as { title?: unknown }).title === "string" && typeof (item as { mount?: unknown }).mount === "function");
  return typeof provider.dimension === "string" && typeof provider.label === "string" && typeof provider.load === "function"
    && form(provider.create) && (provider.actions === undefined || typeof provider.actions === "function")
    && (provider.notice === undefined || isRenderer(provider.notice));
}

/** 已加载的侧栏分组数据源，按目录顺序；宿主不按插件名挑选。 */
export interface NavigationGroupSource {
  /** 插件身份加版本；目录更新后变化。 */
  key: string;
  pluginId: string;
  pluginRevision: string;
  provider: NavigationGroupsProvider;
  context(signal: AbortSignal): Pick<PluginUiHostContext, "query">;
}

export function navigationGroupSources(): NavigationGroupSource[] {
  if (catalog.updating) return [];
  return catalog.plugins.flatMap((plugin) => {
    const loaded = definitions.get(plugin.id);
    const provider = loaded?.revision === plugin.revision ? loaded.definition.slots["navigation.groups"] : undefined;
    return provider ? [groupSource(plugin, provider)] : [];
  });
}

function groupSource(plugin: PluginUiCatalogItem, provider: NavigationGroupsProvider): NavigationGroupSource {
  return { key: `${plugin.id}:${plugin.revision}`, pluginId: plugin.id, pluginRevision: plugin.revision, provider,
    context: (signal) => ({
      query: (method, payload = {}) => queryWebPluginUi({
        pluginId: plugin.id, pluginRevision: plugin.revision, method, payload, slot: "navigation.groups", signal,
      }),
    }) };
}

/** 在宿主外框里挂分组数据源的表单或提示：表单配表单控制，提示配导航能力，类型上不能混用。 */
export function NavigationGroupMount({ source, renderer, extra, scope, refreshToken }: {
  source: NavigationGroupSource;
  /** 变化时通知插件重新读取宿主状态（例如表单 busy）。 */
  refreshToken?: string | number;
} & (
  | { renderer: NavigationForm; extra: { form: NavigationFormControls }; scope: { dimension: string; value?: string } }
  | { renderer: NavigationNotice; extra: { navigation: NavigationControls }; scope: { dimension: string } }
)) {
  return <MountedPlugin pluginId={source.pluginId} pluginRevision={source.pluginRevision} renderer={renderer as HostRenderer}
    extra={extra} refreshToken={refreshToken} context={{ slot: "navigation.groups", scope }} />;
}

export interface ComposerSuggestSource {
  /** 插件身份加版本；目录更新后变化。 */
  key: string;
  provider: ComposerSuggestProvider;
  /** 每次加载一个上下文；宿主撤下数据源或超时时经 signal 中止它的查询。 */
  context(signal: AbortSignal): Pick<PluginUiHostContext, "query">;
}

/** 已加载的输入框建议数据源，按目录顺序；宿主不按插件名挑选。 */
export function composerSuggestSources(): ComposerSuggestSource[] {
  if (catalog.updating) return [];
  return catalog.plugins.flatMap((plugin) => {
    const loaded = definitions.get(plugin.id);
    const provider = loaded?.revision === plugin.revision ? loaded.definition.slots["composer.suggest"] : undefined;
    return provider ? [{ key: `${plugin.id}:${plugin.revision}`, provider, context: (signal) => ({
      query: (method, payload = {}) => queryWebPluginUi({
        pluginId: plugin.id, pluginRevision: plugin.revision, method, payload, slot: "composer.suggest", signal,
      }),
    }) }] : [];
  });
}

/** 宿主据目录变化重读插件数据源；宿主不按插件名挑选插件。 */
export function usePluginUiCatalogVersion(): { version: number; catalogRevision: string } {
  const version = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  return { version, catalogRevision: catalog.catalogRevision };
}

// 插件推送按“插件|revision|会话”记住最新一帧，晚挂载的界面先拿到当前状态；换代后旧代的帧（包括它的 null）
// 碰不到新代界面。新代帧可能先于新目录到达，所以目录激活时只清掉被这次激活确认退役的 revision，其余留着。
const pushed = new Map<string, Record<string, unknown>>();
const pushListeners = new Map<string, Set<(value: Record<string, unknown> | null) => void>>();

function pushKey(pluginId: string, revision: string, sessionId: string): string {
  return JSON.stringify([pluginId, revision, sessionId]);
}

// 结束帧是尽力送达的：同一插件与会话只留当前 revision 和最新到达的一个未确认 revision，丢了 null 也不会越攒越多。
function capPushes(pluginId: string, latest: string, sessionId: string) {
  const current = definitions.get(pluginId)?.revision;
  for (const key of pushed.keys()) {
    const [id, revision, session] = JSON.parse(key) as [string, string, string];
    if (id === pluginId && session === sessionId && revision !== latest && revision !== current) pushed.delete(key);
  }
}

function retirePushes(retired: ReadonlySet<string>) {
  for (const key of pushed.keys()) {
    const [pluginId, revision] = JSON.parse(key) as [string, string, string];
    if (retired.has(JSON.stringify([pluginId, revision]))) pushed.delete(key);
  }
}

function deliverPush(key: string, value: Record<string, unknown> | null) {
  for (const listener of pushListeners.get(key) ?? []) {
    try { listener(value); }
    catch (error) { console.error(`[plugin-ui] push listener failed: ${key}`, error); }
  }
}

/** WebSocket 上的插件推送帧；不是该帧时返回 false，交给其他解析器。 */
export function receivePluginUiEvent(value: unknown): boolean {
  if (typeof value !== "object" || value === null || (value as { type?: unknown }).type !== "plugin_ui.event") return false;
  const frame = value as Record<string, unknown>;
  const state = frame.value;
  if (typeof frame.plugin_id !== "string" || typeof frame.plugin_revision !== "string" || typeof frame.session_id !== "string"
    || (state !== null && (typeof state !== "object" || Array.isArray(state)))) throw new Error("插件推送帧无效");
  const key = pushKey(frame.plugin_id, frame.plugin_revision, frame.session_id);
  // null 表示流已结束，直接忘掉这个键，长连接跨会话不会越攒越多。
  if (state === null) pushed.delete(key);
  else {
    pushed.set(key, state as Record<string, unknown>);
    capPushes(frame.plugin_id, frame.plugin_revision, frame.session_id);
  }
  deliverPush(key, state as Record<string, unknown> | null);
  return true;
}

/** 连接结束时全部推送状态失效；重连后的基线帧再装回。 */
export function resetPluginUiEvents() {
  const keys = [...pushed.keys()];
  pushed.clear();
  keys.forEach((key) => deliverPush(key, null));
}

function subscribePush(pluginId: string, revision: string, sessionId: string | undefined, callback: (value: Record<string, unknown> | null) => void): () => void {
  if (!sessionId) return () => undefined;
  const key = pushKey(pluginId, revision, sessionId);
  const listeners = pushListeners.get(key) ?? new Set();
  pushListeners.set(key, listeners);
  listeners.add(callback);
  const latest = pushed.get(key);
  if (latest) callback(latest);
  return () => {
    listeners.delete(callback);
    if (!listeners.size) pushListeners.delete(key);
  };
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

type InlineSlotName = PluginUiMountSlotName;

// 页面重新可见的三种来源合成一个信号：标签切回、窗口获得焦点、宿主重新显示聊天（经 ChatHandle.shown）。
export function notifyShown(): void {
  for (const listener of [...shownListeners]) {
    try { listener(); }
    catch (error) { console.error("[plugin-ui] shown callback failed", error); }
  }
}

if (typeof window !== "undefined") {
  window.addEventListener("focus", notifyShown);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") notifyShown(); });
}

// 一次挂载登记的"重新可见"回调；挂载卸载时一并撤销，卸载后不再回调。
function shownSubscriptions(isLive: () => boolean): { onShown(callback: () => void): () => void; dispose(): void } {
  const owned = new Set<() => void>();
  return {
    onShown(callback) {
      const listener = () => { if (isLive()) callback(); };
      owned.add(listener);
      shownListeners.add(listener);
      return () => { owned.delete(listener); shownListeners.delete(listener); };
    },
    dispose() { owned.forEach((listener) => shownListeners.delete(listener)); owned.clear(); },
  };
}

// 宿主内部组装挂载上下文用的松散形状；各挂载点必须给出的事实由 PluginUiSlot 的属性类型保证。
type HostMountContext = PluginUiHostContext & {
  slot: InlineSlotName | "navigation.groups";
  sessionId?: string;
  messageId?: string;
  block?: unknown;
  scope?: { dimension: string; value?: string };
  composer?: ComposerControls;
  fields?: ScopeFieldControls;
  form?: NavigationFormControls;
  navigation?: NavigationControls;
};
type HostMountFacts = Pick<HostMountContext, "slot" | "sessionId" | "messageId" | "block" | "scope">;

interface HostRenderer {
  accepts?(block: unknown): boolean;
  prefetch?(context: HostMountContext): Promise<void>;
  mount(host: HTMLElement, context: HostMountContext): void | (() => void);
}

// 挂载点按插槽给出的事实：插件合同里的事实去掉宿主自己构造的输入框与表单控制。
type HostSlotFacts<S extends InlineSlotName> = Omit<PluginUiSlotFacts[S], "composer" | "fields">
  & (S extends "scope.fields" ? { fields: ScopeFields } : unknown);

interface SlotMatch {
  plugin: PluginUiCatalogItem;
  renderer: HostRenderer;
  error?: string;
}

function useRegistryVersion(): number {
  return useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
}

// 目录或模块尚未就绪时，已挂载的展示保持原样。
function slotWaiting(name: InlineSlotName): boolean {
  return catalog.updating || !!catalog.error || catalog.plugins.some(plugin =>
    plugin.slots.includes(name) && definitions.get(plugin.id)?.revision !== plugin.revision);
}

// 选出已加载且认领该 block 的 renderer；accepts 抛错只成为该插件的局部错误。
// 只属于某个 owner 的插槽（工具结果）只问该插件；没有 owner 时谁也不认领。
const OWNED_SLOTS = new Set<InlineSlotName>(["tool.result"]);

function slotMatches(name: InlineSlotName, block: unknown, prefetch: boolean, owner?: string): SlotMatch[] {
  if (OWNED_SLOTS.has(name) && !owner) return [];
  return catalog.plugins.flatMap((plugin) => {
    if (owner !== undefined && plugin.id !== owner) return [];
    const loaded = definitions.get(plugin.id);
    const renderer = loaded?.revision === plugin.revision
      ? loaded.definition.slots[name] as HostRenderer | undefined
      : undefined;
    let accepted = true;
    let error: string | undefined;
    try { accepted = !renderer?.accepts || renderer.accepts(block); }
    catch (cause) { error = cause instanceof Error ? cause.message : String(cause); }
    return renderer && accepted && (!prefetch || renderer.prefetch) ? [{ plugin, renderer, error }] : [];
  });
}

/** 宿主据此决定附件交给插件插槽还是普通附件条；调用方需随目录版本重新渲染。 */
export function pluginSlotAccepts(name: InlineSlotName, block: unknown, owner?: string): boolean {
  return slotMatches(name, block, false, owner).length > 0;
}

export function PluginUiSlot<S extends InlineSlotName>(props: {
  name: S;
  /** 只挂这个插件的渲染器；tool.result 必须提供。 */
  owner?: string;
  prefetch?: boolean;
  refreshToken?: string | number;
  /** 宿主另有降级展示时不显示目录更新和卸载提示。 */
  quiet?: boolean;
  fallback?: React.ReactNode;
} & HostSlotFacts<S>) {
  const { name, owner, refreshToken, prefetch = false, quiet = false, fallback = null } = props;
  const { sessionId, messageId, block, scope, fields } = props as unknown as Omit<HostMountFacts, "slot"> & { fields?: ScopeFields };
  const version = useRegistryVersion();
  const renderers = slotMatches(name, block, prefetch, owner);
  // 目录加载失败时保留本页已经挂载的卡片；新页面不会执行旧模块。
  const previous = React.useRef<typeof renderers>([]);
  const hadRenderer = React.useRef(false);
  const contextKey = JSON.stringify([name, prefetch, sessionId, messageId, block, scope]);
  const previousContext = React.useRef(contextKey);
  const waiting = slotWaiting(name);
  const keepPrevious = waiting && previousContext.current === contextKey && previous.current.length > 0
    && previous.current.every(({plugin}) => catalog.plugins.some(item => item.id === plugin.id));
  const shown = keepPrevious ? previous.current : renderers;
  if (!keepPrevious) { previous.current = renderers; previousContext.current = contextKey; }
  if (shown.length) hadRenderer.current = true;
  const notice = quiet ? "" : waiting ? catalog.error ? `插件界面暂不可用：${catalog.error}` : "插件界面正在更新…"
    : !shown.length && hadRenderer.current ? "本页插件界面已卸载或不再提供此展示。" : "";
  return shown.length || notice ? (
    <div className={prefetch ? "plugin-ui-prefetch" : "plugin-ui-slot"} data-slot={name} data-version={version}>
      {notice && !prefetch && <p role="status">{notice} <button type="button" onClick={() => window.location.reload()}>刷新页面</button></p>}
      {shown.map(({ plugin, renderer, error }) => error ? (
        <div key={plugin.id}><p role="alert">插件界面错误：{error}</p>{fallback}</div>
      ) : (
        <ViewportMountedPlugin
          key={`${plugin.id}:${plugin.revision}:${name}`}
          pluginId={plugin.id}
          pluginRevision={plugin.revision}
          renderer={renderer}
          fallback={fallback}
          prefetch={prefetch}
          refreshToken={refreshToken}
          fields={fields}
          context={{ slot: name, sessionId, messageId, block, scope }}
        />
      ))}
    </div>
  ) : fallback;
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
  fallback,
  fields,
  extra,
}: {
  fallback?: React.ReactNode;
  pluginId: string;
  pluginRevision: string;
  renderer: HostRenderer;
  prefetch?: boolean;
  fields?: ScopeFields;
  /** 宿主外框给这次挂载的额外能力（分组表单与提示）；身份变化即重新挂载。 */
  extra?: Pick<HostMountContext, "form" | "navigation">;
  context: HostMountFacts;
  refreshToken?: string | number;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const ownerIdRef = React.useRef(createOwnerId());
  const invalidatorsRef = React.useRef<Set<() => void> | null>(null);
  const previousRefreshToken = React.useRef(refreshToken);
  const { block, messageId, sessionId, slot } = context;
  const dimension = context.scope?.dimension;
  const scopeValue = context.scope?.value;
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
    let live = true;
    const stopPushes = new Set<() => void>();
    const shown = shownSubscriptions(() => live);
    const composer = slot === "composer.toolbar" ? composerControls() : null;
    const scopeFields = slot === "scope.fields" && fields ? fields.controls() : null;
    setError(null);
    const stopIntents = subscribeWorkbenchIntents(() => {
      for (const callback of invalidators) {
        try { callback(); }
        catch (error) { console.error(`[plugin-ui] invalidation failed: ${pluginId}`, error); }
      }
    });
    try {
      const queryContext: HostMountContext = {
        slot, sessionId, messageId, block: stableBlock,
        onInvalidate: (callback) => {
          invalidators.add(callback);
          return () => { invalidators.delete(callback); };
        },
        reportError: (message) => { if (live) setError(message); },
        onShown: shown.onShown,
        subscribe: (callback) => {
          const stop = subscribePush(pluginId, pluginRevision, sessionId, (value) => { if (live) callback(value); });
          stopPushes.add(stop);
          return () => { stop(); stopPushes.delete(stop); };
        },
        capabilities: { queryCacheModes: ["none", "memory", "immutable"] },
        query: (method, payload = {}, options = {}) => queryPlugin({
          pluginId, pluginRevision, ownerId, slot, sessionId, messageId, method, payload, options, prefetch,
        }),
        workbench: {
          accepts: workbenchAccepts,
          open: (name, payload) => live ? openInWorkbench(name, sessionId ?? "", payload) : Promise.resolve("invalid"),
        },
        ...(composer ? { composer: composer.controls } : {}),
        ...(dimension ? { scope: { dimension, ...(scopeValue ? { value: scopeValue } : {}) } } : {}),
        ...(scopeFields ? { fields: scopeFields.controls } : {}),
        ...extra,
      };
      if (prefetch) {
        void renderer.prefetch!(queryContext).catch((error: unknown) => {
          console.debug(`[plugin-ui] prefetch failed: ${pluginId}`, error);
        });
      } else {
        cleanup = renderer.mount(host, queryContext);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
    return () => {
      live = false;
      stopPushes.forEach((stop) => stop());
      shown.dispose();
      stopIntents();
      invalidators.clear();
      invalidatorsRef.current = null;
      releaseQueryOwner(ownerId);
      composer?.release();
      scopeFields?.release();
      try {
        cleanup?.();
      } catch (error) {
        console.error(`[plugin-ui] cleanup failed: ${pluginId}`, error);
      }
      host.replaceChildren();
    };
  }, [messageId, pluginId, pluginRevision, renderer, sessionId, slot, stableBlock, prefetch, dimension, scopeValue, fields, extra]);
  return <><div ref={hostRef} className="plugin-ui-host" data-plugin={pluginId} hidden={error !== null} />
    {error !== null && <><p role="status">交互内容暂不可用：{error}</p>{fallback}</>}</>;
}

/** 页面缓存查询拥有独立请求，折叠只释放订阅，不打断已经发出的读取。 */
function queryPlugin({ pluginId, pluginRevision, ownerId, slot, sessionId, messageId, method, payload, options, prefetch }: {
  pluginId: string; pluginRevision: string; ownerId: string; slot: PluginUiSlotName;
  sessionId?: string; messageId?: string; method: string; payload: Record<string, unknown>;
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
    ? pluginQueryCacheKey(pluginId, pluginRevision, method, encoded, sessionId) : undefined;
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
      interactive: sharedKey !== undefined && !prefetch,
      send: () => {
        tracePluginQuery(requestId, request, "sent");
        request.timeout = window.setTimeout(() => {
          tracePluginQuery(requestId, request, "timeout");
          rejectOwnerPending(requestOwnerId, "插件请求超时");
        }, 30_000);
        void queryWebPluginUi({ pluginId, pluginRevision, method, payload, slot, sessionId,
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
  signal,
}: {
  pluginId: string;
  pluginRevision: string;
  method: string;
  payload: Record<string, unknown>;
  slot: PluginUiSlotName;
  sessionId?: string;
  signal: AbortSignal;
}): Promise<Record<string, unknown>> {
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
): string {
  return [pluginId, pluginRevision, method, sessionId ?? "", payloadJson].join("\n");
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
