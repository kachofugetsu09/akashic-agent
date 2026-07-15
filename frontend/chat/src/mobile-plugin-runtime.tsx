import React, { useEffect, useMemo, useSyncExternalStore } from "react";

export type MobilePluginSlotName =
  | "turn.before_reasoning"
  | "turn.before_tool"
  | "turn.after_answer"
  | "drawer.panel";

export interface MobilePluginContext {
  slot: MobilePluginSlotName;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
  request(method: string, payload?: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export interface MobilePluginRenderer {
  mount(host: HTMLElement, context: MobilePluginContext): void | (() => void);
}

export interface MobilePluginDefinition {
  slots: Partial<Record<MobilePluginSlotName, MobilePluginRenderer>>;
}

export interface MobilePluginAsset {
  id: string;
  revision: string;
  sha256: string;
  module: string;
  stylesheet: string;
}

const definitions = new Map<string, MobilePluginDefinition>();
const styleUrls = new Map<string, { url: string; node: HTMLLinkElement }>();
const listeners = new Set<() => void>();
const pending = new Map<string, {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
  owner: symbol;
  timeout: number;
}>();
let registryVersion = 0;
let assetQueue: Promise<void> = Promise.resolve();
let queuedSignature: string | null = null;
let activeSignature: string | null = null;
const quarantinedSignatures = new Map<string, Error>();
const MODULE_LOAD_TIMEOUT_MS = 5_000;
const SLOT_NAMES = new Set<MobilePluginSlotName>([
  "turn.before_reasoning",
  "turn.before_tool",
  "turn.after_answer",
  "drawer.panel",
]);

function emitChange() {
  registryVersion += 1;
  listeners.forEach((listener) => listener());
}

export function receiveMobilePluginAssets(assets: MobilePluginAsset[]): Promise<void> {
  const signature = assets.map((asset) => `${asset.id}:${asset.sha256}`).join("|");
  if (signature === activeSignature) return Promise.resolve();
  if (signature === queuedSignature) return assetQueue;
  const quarantined = quarantinedSignatures.get(signature);
  if (quarantined) return Promise.reject(quarantined);
  queuedSignature = signature;
  assetQueue = assetQueue.catch(() => undefined).then(async () => {
    try {
      await activateMobilePluginAssets(assets);
      activeSignature = signature;
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error("移动插件加载失败");
      quarantinedSignatures.set(signature, normalized);
      throw normalized;
    } finally {
      if (queuedSignature === signature) queuedSignature = null;
    }
  });
  return assetQueue;
}

async function activateMobilePluginAssets(assets: MobilePluginAsset[]) {
  const nextDefinitions = new Map<string, MobilePluginDefinition>();
  const nextStyles = new Map<string, { url: string; node: HTMLLinkElement }>();
  try {
    for (const asset of assets) {
      if (asset.stylesheet) {
        const url = URL.createObjectURL(new Blob([asset.stylesheet], { type: "text/css" }));
        const node = document.createElement("link");
        node.rel = "stylesheet";
        node.href = url;
        node.dataset.mobilePlugin = asset.id;
        nextStyles.set(asset.id, { url, node });
      }
      const sourceUrl = URL.createObjectURL(new Blob([asset.module], { type: "text/javascript" }));
      try {
        const module = await withDeadline(
          import(/* @vite-ignore */ sourceUrl) as Promise<{ default?: unknown }>,
          MODULE_LOAD_TIMEOUT_MS,
          `移动插件加载超时: ${asset.id}`,
        );
        nextDefinitions.set(asset.id, parseDefinition(module.default, asset.id));
      } finally {
        URL.revokeObjectURL(sourceUrl);
      }
    }
  } catch (error) {
    nextStyles.forEach((style) => URL.revokeObjectURL(style.url));
    throw error;
  }
  styleUrls.forEach((style) => {
    style.node.remove();
    URL.revokeObjectURL(style.url);
  });
  definitions.clear();
  nextDefinitions.forEach((definition, id) => definitions.set(id, definition));
  styleUrls.clear();
  nextStyles.forEach((style, id) => {
    document.head.appendChild(style.node);
    styleUrls.set(id, style);
  });
  emitChange();
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

function parseDefinition(value: unknown, pluginId: string): MobilePluginDefinition {
  if (!value || typeof value !== "object") {
    throw new Error(`移动插件必须默认导出定义对象: ${pluginId}`);
  }
  const slots = (value as { slots?: unknown }).slots;
  if (!slots || typeof slots !== "object") {
    throw new Error(`移动插件缺少 slots: ${pluginId}`);
  }
  for (const [name, renderer] of Object.entries(slots)) {
    if (!SLOT_NAMES.has(name as MobilePluginSlotName)) {
      throw new Error(`移动插件 slot 无效: ${pluginId}.${name}`);
    }
    if (!renderer || typeof renderer !== "object" ||
      typeof (renderer as { mount?: unknown }).mount !== "function") {
      throw new Error(`移动插件 renderer 无效: ${pluginId}.${name}`);
    }
  }
  return { slots: slots as MobilePluginDefinition["slots"] };
}

export function settleMobilePluginResponses(
  responses: { requestId: string; resultJson?: string; error?: string }[],
) {
  for (const response of responses) {
    const request = pending.get(response.requestId);
    if (!request) continue;
    pending.delete(response.requestId);
    window.clearTimeout(request.timeout);
    if (response.error) request.reject(new Error(response.error));
    else {
      try {
        request.resolve(JSON.parse(response.resultJson ?? "{}") as Record<string, unknown>);
      } catch (error) {
        request.reject(error instanceof Error ? error : new Error("插件响应 JSON 无效"));
      }
    }
  }
  if (responses.length) {
    const requestIds = responses.map((response) => response.requestId);
    for (let offset = 0; offset < requestIds.length; offset += 256) {
      window.AkashicNative?.acknowledgePluginUiResponses(
        JSON.stringify(requestIds.slice(offset, offset + 256)),
      );
    }
  }
}

export function MobilePluginSlot({
  name,
  sessionId,
  messageId,
  turnId,
  block,
}: {
  name: MobilePluginSlotName;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
}) {
  const version = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
    () => registryVersion,
  );
  const renderers = Array.from(definitions.entries())
    .flatMap(([pluginId, plugin]) => plugin.slots[name] ? [{ pluginId, renderer: plugin.slots[name] }] : []);
  return renderers.length ? (
    <div className="mobile-plugin-slot" data-slot={name} data-version={version}>
      {renderers.map(({ pluginId, renderer }) => (
        <MountedPlugin
          key={`${pluginId}:${name}`}
          pluginId={pluginId}
          renderer={renderer!}
          context={{ slot: name, sessionId, messageId, turnId, block }}
        />
      ))}
    </div>
  ) : null;
}

function MountedPlugin({
  pluginId,
  renderer,
  context,
}: {
  pluginId: string;
  renderer: MobilePluginRenderer;
  context: Omit<MobilePluginContext, "request">;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const ownerRef = React.useRef(Symbol(pluginId));
  const { block, messageId, sessionId, slot, turnId } = context;
  const blockRevision = block === undefined ? undefined : JSON.stringify(block);
  const stableBlock = useMemo(
    () => blockRevision === undefined ? undefined : JSON.parse(blockRevision) as unknown,
    [blockRevision],
  );
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const owner = ownerRef.current;
    let cleanup: void | (() => void);
    try {
      cleanup = renderer.mount(host, {
        slot,
        sessionId,
        messageId,
        turnId,
        block: stableBlock,
        request(method, payload = {}) {
          const requestId = createRequestId();
          return new Promise((resolve, reject) => {
            if (!window.AkashicNative) {
              reject(new Error("原生插件桥未连接"));
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
            const timeout = window.setTimeout(() => {
              pending.delete(requestId);
              reject(new Error("插件请求超时"));
            }, 30_000);
            pending.set(requestId, { resolve, reject, owner, timeout });
            try {
              window.AkashicNative.callPluginUi(
                requestId,
                sessionId ?? null,
                turnId ?? null,
                pluginId,
                method,
                encoded,
              );
            } catch (error) {
              window.clearTimeout(timeout);
              pending.delete(requestId);
              reject(error instanceof Error ? error : new Error("原生插件桥调用失败"));
            }
          });
        },
      });
    } catch (error) {
      host.textContent = error instanceof Error ? `插件界面错误：${error.message}` : "插件界面错误";
      host.classList.add("mobile-plugin-host--error");
    }
    return () => {
      for (const [requestId, request] of pending) {
        if (request.owner !== owner) continue;
        window.clearTimeout(request.timeout);
        request.reject(new Error("插件界面已卸载"));
        pending.delete(requestId);
      }
      try {
        cleanup?.();
      } catch (error) {
        console.error(`[mobile] plugin cleanup failed: ${pluginId}`, error);
      }
      host.replaceChildren();
    };
  }, [messageId, pluginId, renderer, sessionId, slot, stableBlock, turnId]);
  return <div ref={hostRef} className="mobile-plugin-host" data-plugin={pluginId} />;
}

function createRequestId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}
