import type {
  WebEntry,
  WebEntryView,
  WebHostContextV1,
  WebMountCardinality,
  WebMountRegistration,
  WebUiDisposer,
} from "@akashic/web-ui-v1";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

type Disposer = WebUiDisposer;
type MountCardinality = WebMountCardinality;

interface WebModulePayload {
  pluginId: string;
  generationId: string;
  module: string;
  moduleSha256: string;
  moduleBytes: number;
  stylesheet: string;
  stylesheetSha256: string | null;
  stylesheetBytes: number;
  requires: string[];
  provides: string[];
  contractDigests: Record<string, string>;
  contractSha256: string;
}

interface WebUiBootstrap {
  schemaVersion: 1;
  snapshotId: string;
  catalogId: string;
  modules: WebModulePayload[];
}

interface WebModuleExports {
  activate(ctx: WebHostContextV1): Disposer;
}

interface MountedEntry {
  readonly entry: WebEntry;
  readonly owner: ModuleActivation;
  readonly childMountIds: string[];
}

interface MountNode {
  readonly id: string;
  readonly cardinality: MountCardinality;
  readonly parentEntry: MountedEntry | null;
  readonly entries: Map<string, MountedEntry>;
}

interface Injection {
  readonly mountId: string;
  readonly owner: ModuleActivation;
  readonly connect: (mount: WebMountRegistration) => Disposer;
  connection: Disposer | null;
}

interface ModuleActivation {
  readonly module: WebModulePayload;
  readonly effects: Disposer[];
  admitting: boolean;
  disposed: boolean;
}

export interface WebHostSession {
  close(): void;
}

class BrowserCatalogSession implements WebHostSession {
  readonly errors = new Map<string, Error>();
  private readonly mounts = new Map<string, MountNode>();
  private readonly injections: Injection[] = [];
  private readonly activations: ModuleActivation[] = [];
  private readonly renderEffects: Disposer[] = [];
  private admissionOpen = true;
  private closing = false;
  private closed = false;
  private staleCatalog = false;
  private staleNotice: HTMLElement | null = null;

  constructor(readonly bootstrap: WebUiBootstrap) {
    this.mounts.set("web.root.v1", {
      id: "web.root.v1",
      cardinality: "single",
      parentEntry: null,
      entries: new Map(),
    });
  }

  async checkCurrent(): Promise<boolean> {
    this.requireOpen();
    const response = await fetch("/api/chat/web-ui/state", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) return false;
    const state = await response.json() as unknown;
    if (!isRecord(state)
      || typeof state.snapshotId !== "string"
      || typeof state.catalogId !== "string") return false;
    if (state.snapshotId === this.bootstrap.snapshotId
      && state.catalogId === this.bootstrap.catalogId) return true;
    this.markStale();
    return false;
  }

  private markStale(): void {
    this.staleCatalog = true;
    if (!this.staleNotice) {
      const notice = document.createElement("div");
      notice.className = "web-host-stale";
      notice.setAttribute("role", "status");
      notice.textContent = "界面已更新，请刷新页面。";
      document.body.prepend(notice);
      this.staleNotice = notice;
    }
  }

  async activateModules(): Promise<void> {
    for (const module of activationOrder(this.bootstrap.modules)) {
      await this.activateModule(module);
    }
    this.flushInjections();
    this.rejectUnresolvedInjections();
    this.verifyContractUse();
    this.admissionOpen = false;
  }

  renderRoot(host: HTMLElement): Disposer {
    this.requireOpen();
    disposeReverse(this.renderEffects);
    const root = this.mounts.get("web.root.v1");
    const entry = root ? this.sortedEntries(root)[0] : undefined;
    host.replaceChildren();
    if (!entry) {
      const empty = document.createElement("p");
      empty.className = this.errors.size ? "web-host-entry-error" : "web-host-empty";
      if (this.errors.size) empty.setAttribute("role", "alert");
      empty.textContent = this.errors.size
        ? `Web 界面加载失败：${[...this.errors.keys()].join("、")}`
        : "没有插件提供 Web 界面。";
      host.appendChild(empty);
      return () => host.replaceChildren();
    }
    const dispose = this.renderEntry(entry, host);
    this.renderEffects.push(dispose);
    return dispose;
  }

  close(): void {
    if (this.closed || this.closing) return;
    this.closing = true;
    this.admissionOpen = false;
    disposeReverse(this.renderEffects);
    for (const activation of [...this.activations].reverse()) {
      activation.disposed = true;
      disposeReverse(activation.effects);
    }
    this.injections.length = 0;
    this.mounts.clear();
    this.staleNotice?.remove();
    this.staleNotice = null;
    this.closed = true;
    this.closing = false;
  }

  private async activateModule(module: WebModulePayload): Promise<void> {
    const activation: ModuleActivation = {
      module,
      effects: [],
      admitting: false,
      disposed: false,
    };
    this.activations.push(activation);
    try {
      verifyModuleAssets(module);
      if (module.stylesheetSha256 !== null) {
        activation.effects.push(installStyle(module));
      }
      const exports = await importModule(module.module);
      if (typeof exports.activate !== "function") {
        throw new Error("web module does not export activate(ctx)");
      }
      activation.admitting = true;
      const disposer = exports.activate(this.contextFor(activation));
      activation.admitting = false;
      if (typeof disposer !== "function") {
        throw new Error("web module activate(ctx) must return a disposer");
      }
      if (!activation.effects.includes(disposer)) activation.effects.push(once(disposer));
      this.flushInjections();
    } catch (reason) {
      activation.admitting = false;
      activation.disposed = true;
      disposeReverse(activation.effects);
      this.removeOwner(activation);
      const error = asError(reason);
      console.error(`[web-host] failed to activate ${module.pluginId}`, error);
      this.errors.set(module.pluginId, error);
    }
  }

  private contextFor(owner: ModuleActivation): WebHostContextV1 {
    return {
      http: {
        request: (path, init) => this.request(owner, path, init),
      },
      ui: {
        inject: (mountId, connect) => this.inject(owner, mountId, connect),
      },
    };
  }

  private request(
    owner: ModuleActivation,
    path: string,
    init: RequestInit = {},
  ): Promise<Response> {
    if (this.closed) throw new Error("web catalog session is closed");
    if (owner.disposed) throw new Error("web module is disposed");
    if (this.staleCatalog) throw new Error("web catalog is stale");
    const url = new URL(path, window.location.origin);
    if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/dashboard/")) {
      throw new Error("web modules may only call their own dashboard API");
    }
    const headers = new Headers(init.headers);
    headers.set("X-Akashic-Web-Snapshot", this.bootstrap.snapshotId);
    headers.set("X-Akashic-Web-Catalog", this.bootstrap.catalogId);
    headers.set("X-Akashic-Web-Module", owner.module.pluginId);
    headers.set("X-Akashic-Web-Generation", owner.module.generationId);
    headers.set("X-Akasic-CSRF", "1");
    return fetch(`${url.pathname}${url.search}`, { ...init, headers }).then((response) => {
      if (!this.closed && response.headers.get("X-Akashic-Web-Stale") === "1") this.markStale();
      return response;
    });
  }

  private inject(
    owner: ModuleActivation,
    mountId: string,
    connect: (mount: WebMountRegistration) => Disposer,
  ): Disposer {
    this.requireAdmission();
    if (!owner.admitting) throw new Error("web registration must be synchronous");
    requireMountId(mountId, "mount");
    if (typeof connect !== "function") throw new Error("mount injector must be a function");
    const injection: Injection = { mountId, owner, connect, connection: null };
    this.injections.push(injection);
    const dispose = once(() => {
      injection.connection?.();
      injection.connection = null;
      const index = this.injections.indexOf(injection);
      if (index >= 0) this.injections.splice(index, 1);
    });
    owner.effects.push(dispose);
    this.connectInjection(injection);
    return dispose;
  }

  private connectInjection(injection: Injection): void {
    if (injection.connection || injection.owner.disposed) return;
    const mount = this.mounts.get(injection.mountId);
    if (!mount) return;
    const previousAdmission = injection.owner.admitting;
    injection.owner.admitting = true;
    let disposer: Disposer;
    try {
      disposer = injection.connect({
        register: (entry) => this.registerEntry(injection.owner, mount, entry),
      });
    } finally {
      injection.owner.admitting = previousAdmission;
    }
    if (typeof disposer !== "function") {
      throw new Error(`mount injector ${injection.mountId} must return a disposer`);
    }
    injection.connection = injection.owner.effects.includes(disposer) ? disposer : once(disposer);
  }

  private flushInjections(): void {
    let connected = -1;
    while (connected !== this.connectedInjectionCount()) {
      connected = this.connectedInjectionCount();
      for (const injection of [...this.injections]) {
        try {
          this.connectInjection(injection);
        } catch (reason) {
          injection.owner.disposed = true;
          disposeReverse(injection.owner.effects);
          this.removeOwner(injection.owner);
          this.errors.set(injection.owner.module.pluginId, asError(reason));
        }
      }
    }
  }

  private connectedInjectionCount(): number {
    return this.injections.filter((item) => item.connection !== null).length;
  }

  private rejectUnresolvedInjections(): void {
    for (const injection of [...this.injections]) {
      if (injection.connection !== null || injection.owner.disposed) continue;
      injection.owner.disposed = true;
      disposeReverse(injection.owner.effects);
      this.removeOwner(injection.owner);
      this.errors.set(
        injection.owner.module.pluginId,
        new Error(`mount is unavailable: ${injection.mountId}`),
      );
    }
  }

  private verifyContractUse(): void {
    for (const owner of this.activations) {
      if (owner.disposed) continue;
      const requires = this.injections
        .filter((item) => item.owner === owner)
        .map((item) => item.mountId)
        .sort();
      const provides = [...this.mounts.values()]
        .filter((mount) => mount.parentEntry?.owner === owner)
        .map((mount) => mount.id)
        .sort();
      if (sameStrings(requires, [...owner.module.requires].sort())
        && sameStrings(provides, [...owner.module.provides].sort())) continue;
      owner.disposed = true;
      disposeReverse(owner.effects);
      this.removeOwner(owner);
      this.errors.set(
        owner.module.pluginId,
        new Error("web module contract does not match its declared mounts"),
      );
    }
  }

  private registerEntry(owner: ModuleActivation, mount: MountNode, raw: WebEntry): Disposer {
    this.requireAdmission();
    if (!owner.admitting) throw new Error("web registration must be synchronous");
    if (!raw || typeof raw !== "object") throw new Error("mount entry must be an object");
    requireEntryId(raw.id);
    if (raw.render !== undefined && typeof raw.render !== "function") {
      throw new Error(`entry ${raw.id} has invalid render`);
    }
    if (mount.entries.has(raw.id)) throw new Error(`duplicate entry ${mount.id}:${raw.id}`);
    if (mount.cardinality === "single" && mount.entries.size > 0) {
      throw new Error(`single mount ${mount.id} already has an entry`);
    }
    if (raw.order !== undefined && !Number.isFinite(raw.order)) {
      throw new Error(`entry ${raw.id} has invalid order`);
    }
    if (raw.children !== undefined && !Array.isArray(raw.children)) {
      throw new Error(`entry ${raw.id} children must be an array`);
    }
    const definitions = raw.children ?? [];
    const childIds = new Set<string>();
    for (const child of definitions) {
      if (!child || typeof child !== "object") throw new Error("child mount must be an object");
      requireMountId(child.id, "child mount");
      if (child.cardinality !== "single" && child.cardinality !== "list") {
        throw new Error(`mount ${child.id} has invalid cardinality`);
      }
      if (childIds.has(child.id) || this.mounts.has(child.id)) {
        throw new Error(`duplicate mount ${child.id}`);
      }
      childIds.add(child.id);
    }
    const childMountIds: string[] = [];
    const mounted: MountedEntry = { entry: raw, owner, childMountIds };
    for (const child of definitions) {
      this.mounts.set(child.id, {
        id: child.id,
        cardinality: child.cardinality,
        parentEntry: mounted,
        entries: new Map(),
      });
      childMountIds.push(child.id);
    }
    mount.entries.set(raw.id, mounted);
    const dispose = once(() => this.removeEntry(mount, mounted));
    owner.effects.push(dispose);
    return dispose;
  }

  private removeEntry(mount: MountNode, mounted: MountedEntry): void {
    for (const mountId of [...mounted.childMountIds].reverse()) {
      this.removeMount(mountId);
    }
    if (mount.entries.get(mounted.entry.id) === mounted) {
      mount.entries.delete(mounted.entry.id);
    }
  }

  private removeMount(mountId: string): void {
    const mount = this.mounts.get(mountId);
    if (!mount) return;
    for (const entry of [...mount.entries.values()].reverse()) this.removeEntry(mount, entry);
    this.mounts.delete(mountId);
    for (const injection of [...this.injections]) {
      if (injection.mountId === mountId) {
        injection.connection?.();
        injection.connection = null;
      }
    }
  }

  private removeOwner(owner: ModuleActivation): void {
    for (const mount of [...this.mounts.values()]) {
      for (const entry of [...mount.entries.values()]) {
        if (entry.owner === owner) this.removeEntry(mount, entry);
      }
    }
  }

  private sortedEntries(mount: MountNode): MountedEntry[] {
    return [...mount.entries.values()].sort((left, right) =>
      (left.entry.order ?? 0) - (right.entry.order ?? 0)
      || left.entry.id.localeCompare(right.entry.id),
    );
  }

  private renderEntry(mounted: MountedEntry, host: HTMLElement, props?: unknown): Disposer {
    if (!mounted.entry.render) throw new Error(`entry ${mounted.entry.id} cannot be rendered`);
    const releaseStyle = installEntryStyle(mounted, host);
    const childIds = new Set(mounted.childMountIds);
    const childRenderEffects: Disposer[] = [];
    const trackChildEffect = (dispose: Disposer): Disposer => {
      const tracked = once(() => {
        dispose();
        const index = childRenderEffects.indexOf(tracked);
        if (index >= 0) childRenderEffects.splice(index, 1);
      });
      childRenderEffects.push(tracked);
      return tracked;
    };
    const view: WebEntryView = {
      child: (mountId) => {
        if (!childIds.has(mountId)) throw new Error(`entry does not own mount ${mountId}`);
        const mount = this.mounts.get(mountId);
        if (!mount) throw new Error(`mount is unavailable: ${mountId}`);
        const entries = this.sortedEntries(mount);
        return {
          entries: entries.map((item) => item.entry),
          render: (entryId, target, childProps) => {
            const child = entries.find((item) => item.entry.id === entryId);
            if (!child) throw new Error(`entry is unavailable: ${mountId}:${entryId}`);
            target.replaceChildren();
            return trackChildEffect(this.renderEntry(child, target, childProps));
          },
          style: (entryId, target) => {
            const child = entries.find((item) => item.entry.id === entryId);
            if (!child) throw new Error(`entry is unavailable: ${mountId}:${entryId}`);
            return trackChildEffect(installEntryStyle(child, target));
          },
        };
      },
    };
    let pluginDispose: void | Disposer;
    try {
      pluginDispose = mounted.entry.render(host, view, props);
      if (pluginDispose !== undefined && typeof pluginDispose !== "function") {
        throw new Error(`entry ${mounted.entry.id} render returned an invalid disposer`);
      }
    } catch (reason) {
      host.replaceChildren();
      const error = document.createElement("p");
      error.className = "web-host-entry-error";
      error.setAttribute("role", "alert");
      error.textContent = `界面模块加载失败：${asError(reason).message}`;
      host.appendChild(error);
      pluginDispose = undefined;
    }
    return once(() => {
      disposeReverse(childRenderEffects);
      pluginDispose?.();
      host.replaceChildren();
      releaseStyle();
    });
  }

  private requireAdmission(): void {
    this.requireOpen();
    if (!this.admissionOpen) throw new Error("web mount admission is closed");
  }

  private requireOpen(): void {
    if (this.closed || this.closing) throw new Error("web catalog session is closed");
  }
}

function installEntryStyle(mounted: MountedEntry, host: HTMLElement): Disposer {
  const entryId = mounted.entry.id;
  const moduleId = mounted.owner.module.pluginId;
  const styleScope = styleScopeId(mounted.owner.module);
  if (host.hasAttribute("data-akashic-style")) {
    throw new Error("entry host already has a stylesheet owner");
  }
  host.dataset.akashicEntry = entryId;
  host.dataset.akashicModule = moduleId;
  host.dataset.akashicStyle = styleScope;
  return once(() => {
    if (host.dataset.akashicEntry === entryId) delete host.dataset.akashicEntry;
    if (host.dataset.akashicModule === moduleId) delete host.dataset.akashicModule;
    if (host.dataset.akashicStyle === styleScope) delete host.dataset.akashicStyle;
  });
}

/** Load providers before consumers so inherited CSS follows the mount tree. */
function activationOrder(modules: WebModulePayload[]): WebModulePayload[] {
  const provider = new Map<string, WebModulePayload>();
  for (const module of modules) {
    for (const mountId of module.provides) {
      if (provider.has(mountId)) throw new Error(`duplicate Web mount provider: ${mountId}`);
      provider.set(mountId, module);
    }
  }
  const pending = new Set(modules);
  const ordered: WebModulePayload[] = [];
  while (pending.size) {
    const ready = modules.filter((module) => pending.has(module) && module.requires.every((mountId) => {
      const dependency = provider.get(mountId);
      return dependency === undefined || !pending.has(dependency);
    }));
    if (!ready.length) throw new Error("Web UI mount dependencies contain a cycle");
    for (const module of ready) {
      pending.delete(module);
      ordered.push(module);
    }
  }
  return ordered;
}

export async function startWebHost(host: HTMLElement): Promise<WebHostSession> {
  const response = await fetch("/api/chat/web-ui/bootstrap", {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Web UI bootstrap failed: ${response.status}`);
  const bootstrap = parseBootstrap(await response.json());
  host.dataset.akashicCatalog = bootstrap.catalogId;
  const session = new BrowserCatalogSession(bootstrap);
  await session.activateModules();
  session.renderRoot(host);
  const checkCurrent = (): void => {
    void session.checkCurrent().catch((error) => {
      console.warn("[web-host] catalog state unavailable", error);
    });
  };
  window.addEventListener("focus", checkCurrent);
  document.addEventListener("visibilitychange", checkCurrent);
  const close = session.close.bind(session);
  session.close = once(() => {
    window.removeEventListener("focus", checkCurrent);
    document.removeEventListener("visibilitychange", checkCurrent);
    close();
  });
  return session;
}

function parseBootstrap(value: unknown): WebUiBootstrap {
  if (!isRecord(value) || value.schemaVersion !== 1 || typeof value.snapshotId !== "string"
    || !value.snapshotId || typeof value.catalogId !== "string" || !/^[0-9a-f]{64}$/.test(value.catalogId)
    || !Array.isArray(value.modules)) {
    throw new Error("Web UI bootstrap is invalid");
  }
  const modules = value.modules.map((raw, index): WebModulePayload => {
    if (!isRecord(raw)
      || typeof raw.pluginId !== "string" || !raw.pluginId
      || typeof raw.generationId !== "string" || !raw.generationId
      || typeof raw.module !== "string"
      || typeof raw.moduleSha256 !== "string" || !/^[0-9a-f]{64}$/.test(raw.moduleSha256)
      || typeof raw.moduleBytes !== "number" || !Number.isSafeInteger(raw.moduleBytes) || raw.moduleBytes < 1
      || typeof raw.stylesheet !== "string"
      || (raw.stylesheetSha256 !== null
        && (typeof raw.stylesheetSha256 !== "string" || !/^[0-9a-f]{64}$/.test(raw.stylesheetSha256)))
      || typeof raw.stylesheetBytes !== "number"
      || !Number.isSafeInteger(raw.stylesheetBytes)
      || raw.stylesheetBytes < 0
      || !stringList(raw.requires) || !stringList(raw.provides)
      || !digestRecord(raw.contractDigests)
      || typeof raw.contractSha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(raw.contractSha256)) {
      throw new Error(`Web UI module ${index} is invalid`);
    }
    return raw as unknown as WebModulePayload;
  });
  if (new Set(modules.map((item) => item.pluginId)).size !== modules.length) {
    throw new Error("Web UI bootstrap contains duplicate modules");
  }
  return { schemaVersion: 1, snapshotId: value.snapshotId, catalogId: value.catalogId, modules };
}

function verifyAsset(content: string, bytes: number, digest: string): void {
  const encoded = new TextEncoder().encode(content);
  if (encoded.byteLength !== bytes) throw new Error("Web UI asset byte count mismatch");
  const actual = bytesToHex(sha256(encoded));
  if (actual !== digest) throw new Error("Web UI asset digest mismatch");
}

function verifyModuleAssets(module: WebModulePayload): void {
  verifyAsset(module.module, module.moduleBytes, module.moduleSha256);
  if (module.stylesheetSha256 !== null) {
    verifyAsset(module.stylesheet, module.stylesheetBytes, module.stylesheetSha256);
  } else if (module.stylesheet || module.stylesheetBytes !== 0) {
    throw new Error("stylesheet descriptor is inconsistent");
  }
  const contract = JSON.stringify({
    contractDigests: module.contractDigests,
    provides: module.provides,
    requires: module.requires,
  });
  verifyAsset(contract, new TextEncoder().encode(contract).byteLength, module.contractSha256);
}

function digestRecord(value: unknown): value is Record<string, string> {
  return isRecord(value) && Object.entries(value).every(([contract, digest]) => (
    contract.length > 0 && contract === contract.trim()
      && typeof digest === "string" && /^[0-9a-f]{64}$/.test(digest)
  ));
}

async function importModule(source: string): Promise<WebModuleExports> {
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  try {
    return await import(/* @vite-ignore */ url) as WebModuleExports;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function installStyle(module: WebModulePayload): Disposer {
  const style = document.createElement("style");
  style.dataset.akashicModuleStyle = module.pluginId;
  style.textContent = `@scope ([data-akashic-style="${styleScopeId(module)}"]) {\n${module.stylesheet}\n}`;
  document.head.appendChild(style);
  return once(() => style.remove());
}

function styleScopeId(module: WebModulePayload): string {
  return module.stylesheetSha256 ?? module.moduleSha256;
}

function disposeReverse(disposers: Disposer[]): void {
  for (const dispose of [...disposers].reverse()) {
    try { dispose(); } catch (error) { console.error("[web-host] cleanup failed", error); }
  }
  disposers.length = 0;
}

function once(dispose: Disposer): Disposer {
  let active = true;
  return () => {
    if (!active) return;
    active = false;
    dispose();
  };
}

function requireMountId(value: string, label: string): void {
  if (typeof value !== "string" || !/^[a-z][a-z0-9.-]*\.v[1-9][0-9]*$/.test(value)) {
    throw new Error(`${label} id is invalid: ${String(value)}`);
  }
}

function requireEntryId(value: string): void {
  if (typeof value !== "string" || !/^[a-z][a-z0-9.-]*$/.test(value)) {
    throw new Error(`entry id is invalid: ${String(value)}`);
  }
}

function asError(reason: unknown): Error {
  return reason instanceof Error ? reason : new Error(String(reason));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringList(value: unknown): value is string[] {
  return Array.isArray(value)
    && new Set(value).size === value.length
    && value.every((item) => typeof item === "string"
      && /^[a-z][a-z0-9.-]*\.v[1-9][0-9]*$/.test(item));
}

function sameStrings(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}
