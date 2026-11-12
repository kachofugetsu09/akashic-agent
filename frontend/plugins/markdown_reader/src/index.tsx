import { micromark } from "micromark";
import { gfm, gfmHtml } from "micromark-extension-gfm";
import type { ConversationIntent, ConversationIntentStatus, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { FileSyntax } from "./syntax";
import "./style.css";

type Entry = { name: string; path: string; kind?: "directory" | "file" | "other" };
type RootPreset = { id: string; label: string; path: string };
type Result = {
  status: string;
  path?: string;
  text?: string;
  bytes?: number;
  mime_type?: string;
  mtime_ms?: number;
  truncated?: boolean;
  tail?: boolean;
  message?: string;
  error?: string;
  root?: string | null;
  roots?: RootPreset[];
  open_file?: string;
  parent?: string | null;
  after?: string | null;
  items?: Entry[];
};

const API = "/api/dashboard/markdown_reader";
const PINS_KEY = "akashic.files.pins.v1";
const SPLIT_WIDTH_PX = 480;
const MAX_PINS = 8;
const IMPORT_LIMIT_BYTES = 50 * 1024 * 1024;
const FOLDER_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>';

const STATUS_MESSAGES: Record<string, string> = {
  not_found: "服务端路径不存在",
  not_directory: "该路径不是目录",
  permission_denied: "没有读取权限",
  not_file: "只能预览普通文件",
  too_large: "文件过大，请下载后查看",
  invalid_text: "文件不是 UTF-8 文本",
  invalid_path: "路径格式无效",
  offline: "执行主机离线",
  io_error: "磁盘读取失败",
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text = "", className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function btn(text: string, action: () => void, className = ""): HTMLButtonElement {
  const node = el("button", text, className);
  node.type = "button";
  node.onclick = action;
  return node;
}

function toggle(text: string, action: () => void, className = ""): HTMLButtonElement {
  const node = btn(text, action, className);
  node.setAttribute("aria-pressed", "false");
  return node;
}

function basename(path: string): string {
  const trimmed = path.length > 1 ? path.replace(/\/+$/, "") : path;
  return trimmed.slice(trimmed.lastIndexOf("/") + 1) || trimmed;
}

function dirnameOf(path: string): string {
  const idx = path.lastIndexOf("/");
  if (idx <= 0) return "/";
  return path.slice(0, idx);
}

function normalizePosix(path: string): string {
  const absolute = path.startsWith("/");
  const stack: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") stack.pop();
    else stack.push(part);
  }
  return (absolute ? "/" : "") + stack.join("/") || (absolute ? "/" : ".");
}

function isMarkdown(path: string): boolean {
  return /\.(md|markdown|mdown)$/i.test(path);
}

function formatBytes(bytes: number | undefined): string {
  if (bytes === undefined || !Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function readPins(): string[] {
  try {
    const raw: unknown = JSON.parse(window.localStorage.getItem(PINS_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((item): item is string => typeof item === "string" && item.startsWith("/")).slice(0, MAX_PINS) : [];
  } catch {
    return [];
  }
}

function writePins(pins: readonly string[]): void {
  try {
    window.localStorage.setItem(PINS_KEY, JSON.stringify(pins.slice(0, MAX_PINS)));
  } catch {
    // 无痕或存储满时只在内存生效。
  }
}

async function copyString(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // 非 HTTPS 下回退临时选区复制。
  }
  try {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "true");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

class Reader {
  root = el("section", "", "md-reader");
  private name = el("span", "", "md-name");
  private meta = el("span", "", "md-meta");
  private body = el("div", "", "md-body");
  private nav = el("nav", "", "md-nav");
  private topNavRow = el("div", "", "md-nav-top");
  private rootsRow = el("div", "", "md-roots");
  private pathToggle = toggle("路径", () => this.togglePathBox(), "md-chip-btn");
  private pinBtn = toggle("收藏", () => this.togglePin(), "md-chip-btn");
  private pathBox = el("div", "", "md-path-box");
  private form = el("form", "", "md-open");
  private input = el("input");
  private crumbBar = el("div", "", "md-crumbs");
  private filterInput = el("input", "", "md-filter");
  private hiddenToggle = toggle("隐藏项", () => this.toggleHidden(), "md-chip-btn");
  private navStatusRow = el("div", "", "md-note-row md-nav-status");
  private navNote = el("p", "", "md-note");
  private navRetryBtn = btn("重试", () => this.retryLast(), "md-inline-btn");
  private list = el("div", "", "md-list");
  private outlinePane = el("nav", "", "md-outline");
  private stage = el("div", "", "md-stage");
  private noteRow = el("div", "", "md-note-row");
  private note = el("p", "", "md-note");
  private retryBtn = btn("重试", () => this.retryLast(), "md-inline-btn");
  private banner = el("div", "", "md-banner");
  private article = el("article", "", "md-article");
  private filesToggle = toggle("目录", () => this.toggleFiles());
  private outlineToggle = toggle("大纲", () => this.toggleOutline());
  private sourceToggle = toggle("源码", () => this.toggleSource());
  private quoteBtn = btn("引用", () => this.quoteSelection());
  private copyBtn = btn("复制路径", () => void this.copyPath());
  private downloadBtn = btn("下载", () => void this.downloadCurrent(), "md-download");
  private refreshBtn = btn("刷新", () => void this.refreshCurrent());

  private session: string | null;
  private path = "";
  private directory = "";
  private workdir: string | null | undefined;
  private roots: RootPreset[] = [];
  private pins: string[] = readPins();
  private items: Entry[] = [];
  private afterCursor: string | null = null;
  private parentDir: string | null = null;
  private showHidden = false;
  private navOpen = true;
  private pathBoxOpen = false;
  private outlineOpen = false;
  private source = false;
  private downloading = false;
  private docResult: Result | null = null;
  private selectedLines: { start: number; end: number } | null = null;
  private pendingLine: number | null = null;
  private retryAction: (() => void) | null = null;
  private filesAbort: AbortController | null = null;
  private docAbort: AbortController | null = null;
  private blobUrls: string[] = [];
  private generation = 0;
  private navigation = 0;
  private copyTimer = 0;
  private quoteTimer = 0;
  private targetTimer = 0;
  private disposed = false;
  private readonly syntax = new FileSyntax();
  private stops: WebUiDisposer[];

  constructor(private ctx: WebHostContextV1, private view: ConversationTabView) {
    this.session = view.session;
    this.build();
    this.article.addEventListener("click", (event) => this.follow(event));
    let subscribing = true;
    this.stops = [
      view.onSessionChange((session) => {
        if (session === this.session && this.directory) return;
        this.session = session;
        this.workdir = undefined;
        if (!this.directory) void this.files("");
        else void this.refreshSessionRoots();
      }),
      view.onIntent(async (intent) => this.handleIntent(intent)),
      view.onActiveChange((active) => {
        if (active && !subscribing && !this.directory && !this.filesAbort) {
          void this.files("");
        }
      }),
    ];
    subscribing = false;
    if (!this.directory && !this.filesAbort) {
      void this.files("");
    }
  }

  private build(): void {
    const bar = el("div", "", "md-bar");
    const titleWrap = el("div", "", "md-title-wrap");
    titleWrap.append(this.filesToggle, this.name, this.meta);
    const tools = el("div", "", "md-tools");
    tools.append(this.sourceToggle, this.outlineToggle, this.quoteBtn, this.copyBtn, this.downloadBtn, this.refreshBtn);
    bar.append(titleWrap, tools);

    const navActions = el("div", "", "md-nav-actions");
    this.pathToggle.title = "输入指定服务端目录或文件绝对路径";
    navActions.append(this.pathToggle, this.pinBtn);
    this.topNavRow.append(this.rootsRow, navActions);

    this.input.placeholder = "服务端路径（如 /srv/... 或 ~/...）";
    this.input.setAttribute("aria-label", "服务端目录或文件路径");
    this.input.autocomplete = "off";
    this.input.spellcheck = false;
    const goBtn = el("button", "前往");
    goBtn.type = "submit";
    this.form.append(this.input, goBtn);
    this.form.onsubmit = (event) => {
      event.preventDefault();
      const target = this.input.value.trim();
      if (target) void this.files(target);
    };
    const serverHint = el("p", "访问运行 Akashic 的服务端（执行主机）文件，非当前浏览器本机磁盘。", "md-server-hint");
    this.pathBox.hidden = true;
    this.pathBox.append(this.form, serverHint);

    const filterRow = el("div", "", "md-filter-row");
    this.filterInput.type = "search";
    this.filterInput.placeholder = "筛选当前目录";
    this.filterInput.setAttribute("aria-label", "筛选当前目录");
    this.filterInput.autocomplete = "off";
    this.filterInput.spellcheck = false;
    this.filterInput.oninput = () => this.renderEntries();
    filterRow.append(this.filterInput, this.hiddenToggle);

    this.navNote.setAttribute("role", "status");
    this.navNote.setAttribute("aria-live", "polite");
    this.navRetryBtn.hidden = true;
    this.navStatusRow.hidden = true;
    this.navStatusRow.append(this.navNote, this.navRetryBtn);

    this.nav.setAttribute("aria-label", "服务端目录浏览");
    this.nav.append(this.topNavRow, this.pathBox, this.crumbBar, filterRow, this.navStatusRow, this.list);

    this.outlinePane.setAttribute("aria-label", "文档大纲");
    this.outlinePane.hidden = true;

    this.note.setAttribute("role", "status");
    this.note.setAttribute("aria-live", "polite");
    this.retryBtn.hidden = true;
    this.noteRow.hidden = true;
    this.noteRow.append(this.note, this.retryBtn);
    this.banner.hidden = true;

    this.article.tabIndex = 0;
    this.article.setAttribute("aria-label", "文件内容预览");
    this.article.append(el("p", "选择文件预览，或点「路径」输入任意服务端目录。", "md-empty-file"));
    this.stage.append(this.outlinePane, this.noteRow, this.banner, this.article);
    this.body.append(this.nav, this.stage);
    this.root.append(bar, this.body);
    this.syncChrome();
  }

  private isNarrow(): boolean {
    const width = this.root.clientWidth;
    return width > 0 && width < SPLIT_WIDTH_PX;
  }

  private clearBlobs(): void {
    for (const url of this.blobUrls) URL.revokeObjectURL(url);
    this.blobUrls = [];
  }

  private syncChrome(): void {
    this.root.dataset.navOpen = String(this.navOpen);
    this.root.dataset.hasFile = String(Boolean(this.path && this.docResult));
    this.nav.hidden = !this.navOpen;
    this.pathBox.hidden = !this.pathBoxOpen;
    this.outlinePane.hidden = !this.outlineOpen;
    this.filesToggle.setAttribute("aria-pressed", String(this.navOpen));
    this.pathToggle.setAttribute("aria-pressed", String(this.pathBoxOpen));
    this.outlineToggle.setAttribute("aria-pressed", String(this.outlineOpen));
    this.sourceToggle.setAttribute("aria-pressed", String(this.source));
    this.hiddenToggle.setAttribute("aria-pressed", String(this.showHidden));

    const hasFile = Boolean(this.path && this.docResult);
    const md = hasFile && isMarkdown(this.path) && this.docResult?.status === "available" && typeof this.docResult.text === "string";
    const hasText = hasFile && this.docResult?.status === "available" && typeof this.docResult.text === "string";
    this.sourceToggle.hidden = !md;
    this.outlineToggle.hidden = !md;
    if (!md && this.outlineOpen) {
      this.outlineOpen = false;
      this.outlinePane.hidden = true;
    }
    this.quoteBtn.hidden = !(hasText && this.session);
    this.copyBtn.hidden = !this.path && !this.directory;
    const canDownload = hasFile && (this.docResult?.bytes ?? 0) <= IMPORT_LIMIT_BYTES
      && (this.docResult?.status === "available" || this.docResult?.status === "binary");
    this.downloadBtn.hidden = !canDownload;
    this.downloadBtn.disabled = this.downloading;
    this.downloadBtn.textContent = this.downloading ? "下载中…" : "下载";

    const pinned = Boolean(this.directory && this.pins.includes(this.directory));
    this.pinBtn.hidden = !this.directory;
    this.pinBtn.textContent = pinned ? "★ 已收藏" : "☆ 收藏";
    this.pinBtn.setAttribute("aria-pressed", String(pinned));
    this.renderRoots();
  }

  private rawUrl(path: string, download: boolean): string {
    const query = new URLSearchParams({ path, ...(this.session ? { session: this.session } : {}), ...(download ? { download: "1" } : {}) });
    return `${API}/raw?${query}`;
  }

  // 窄屏目录打开时舞台被收起，目录请求的状态提示需要同时在目录区可见。
  private setStatus(text: string, isError = false, retry: (() => void) | null = null, scope: "nav" | "stage" = "stage"): void {
    this.retryAction = retry;
    if (scope === "nav") {
      this.navNote.textContent = text;
      this.navStatusRow.dataset.tone = isError ? "error" : "";
      this.navRetryBtn.hidden = !retry;
      this.navStatusRow.hidden = !text;
      if (!text) {
        this.noteRow.hidden = true;
      }
      return;
    }
    this.note.textContent = text;
    this.noteRow.dataset.tone = isError ? "error" : "";
    this.retryBtn.hidden = !retry;
    this.noteRow.hidden = !text;
  }

  private retryLast(): void {
    const run = this.retryAction;
    if (run) run();
  }

  private toggleFiles(): void {
    this.navOpen = !this.navOpen;
    this.syncChrome();
  }

  private togglePathBox(): void {
    this.pathBoxOpen = !this.pathBoxOpen;
    this.syncChrome();
    if (this.pathBoxOpen) {
      this.input.value = this.directory || this.path;
      this.input.focus();
      this.input.select();
    }
  }

  private toggleOutline(): void {
    if (this.isNarrow() && this.navOpen) this.navOpen = false;
    this.outlineOpen = !this.outlineOpen;
    if (this.outlineOpen) this.buildOutline();
    this.syncChrome();
  }

  private toggleSource(): void {
    if (this.isNarrow() && this.navOpen) this.navOpen = false;
    this.source = !this.source;
    this.syncChrome();
    this.renderDocument();
    if (this.outlineOpen) this.buildOutline();
  }

  private toggleHidden(): void {
    this.showHidden = !this.showHidden;
    this.syncChrome();
    void this.files(this.directory);
  }

  private togglePin(): void {
    if (!this.directory) return;
    if (this.pins.includes(this.directory)) {
      this.pins = this.pins.filter((item) => item !== this.directory);
    } else {
      this.pins = [this.directory, ...this.pins.filter((item) => item !== this.directory)].slice(0, MAX_PINS);
    }
    writePins(this.pins);
    this.syncChrome();
  }

  private renderRoots(): void {
    const presets: RootPreset[] = [...this.roots];
    const seen = new Set(presets.map((item) => item.path));
    for (const pin of this.pins) {
      if (!seen.has(pin)) {
        seen.add(pin);
        presets.push({ id: `pin:${pin}`, label: `★ ${basename(pin)}`, path: pin });
      }
    }
    this.rootsRow.replaceChildren();
    for (const preset of presets) {
      const chip = btn(preset.label, () => void this.files(preset.path), "md-root-chip");
      chip.title = preset.path;
      if (this.directory === preset.path) chip.setAttribute("aria-current", "true");
      this.rootsRow.append(chip);
    }
  }

  private renderCrumbs(): void {
    this.crumbBar.replaceChildren();
    if (!this.directory) {
      this.crumbBar.hidden = true;
      return;
    }
    this.crumbBar.hidden = false;
    const rootBtn = btn("/", () => void this.files("/"), "md-crumb");
    rootBtn.title = "/";
    if (this.directory === "/") rootBtn.setAttribute("aria-current", "location");
    this.crumbBar.append(rootBtn);
    const parts = this.directory.split("/").filter(Boolean);
    let acc = "";
    parts.forEach((part, index) => {
      acc += `/${part}`;
      const target = acc;
      if (index > 0) this.crumbBar.append(el("span", "/", "md-crumb-sep"));
      const item = btn(part, () => void this.files(target), "md-crumb");
      item.title = target;
      if (index === parts.length - 1) item.setAttribute("aria-current", "location");
      this.crumbBar.append(item);
    });
    requestAnimationFrame(() => {
      this.crumbBar.scrollLeft = this.crumbBar.scrollWidth;
    });
  }

  private async handleIntent(intent: ConversationIntent): Promise<ConversationIntentStatus> {
    if (intent.name !== "workspace.document.open.v1" && intent.name !== "workspace.file.open.v1") return "invalid";
    const payload = intent.payload as { path?: unknown; line?: unknown } | null;
    if (!payload || typeof payload.path !== "string" || !payload.path || payload.path.length > 4096
      || /[\x00-\x1f]/.test(payload.path) || /^[a-z][a-z0-9+.-]*:/i.test(payload.path) || payload.path.startsWith("//")) {
      return "invalid";
    }
    const line = typeof payload.line === "number" && Number.isInteger(payload.line) && payload.line > 0 ? payload.line : null;
    await this.openOrBrowse(payload.path, line);
    return "handled";
  }

  private async openOrBrowse(targetPath: string, line: number | null): Promise<void> {
    if (targetPath.endsWith("/") && targetPath.length > 1) {
      this.navOpen = true;
      this.syncChrome();
      await this.files(targetPath);
      return;
    }
    const opened = await this.open(targetPath, { line, allowDirectoryFallback: true });
    if (opened === "directory") {
      this.navOpen = true;
      this.syncChrome();
      await this.files(targetPath);
    }
  }

  private async refreshSessionRoots(): Promise<void> {
    try {
      const result = await this.requestJson("files", { path: this.directory, ...(this.showHidden ? { hidden: "1" } : {}) });
      if (this.disposed) return;
      const nextWorkdir = result.root ?? null;
      const prevWorkdir = this.workdir;
      this.workdir = nextWorkdir;
      this.roots = this.normalizeRoots(result);
      if (nextWorkdir && nextWorkdir !== prevWorkdir && nextWorkdir !== this.directory) {
        void this.files(nextWorkdir);
        return;
      }
      this.syncChrome();
    } catch {
      // 后台刷新根目录列表失败不打断当前阅读。
    }
  }

  private normalizeRoots(result: Result): RootPreset[] {
    if (Array.isArray(result.roots) && result.roots.length) return result.roots;
    if (typeof result.root === "string" && result.root) return [{ id: "workdir", label: "会话目录", path: result.root }];
    return this.roots;
  }

  private async requestJson(route: string, params: Record<string, string>, signal?: AbortSignal): Promise<Result> {
    const query = new URLSearchParams({ ...(this.session ? { session: this.session } : {}), ...params });
    const response = await this.ctx.http.request(`${API}/${route}?${query}`, signal ? { signal } : undefined);
    let data: Result | null = null;
    try {
      data = await response.json() as Result;
    } catch {
      if (!response.ok) throw new Error(`服务端响应异常（${response.status}）`);
      throw new Error("服务端返回格式无效");
    }
    if (!response.ok) {
      throw new Error(data?.message || data?.error || `请求失败（${response.status}）`);
    }
    return data;
  }

  private async files(path = "", after = ""): Promise<void> {
    const navId = ++this.navigation;
    this.filesAbort?.abort();
    const abort = this.filesAbort = new AbortController();
    this.view.setIndicator?.({ busy: true, detail: "读取目录" });
    this.setStatus(`正在读取服务端目录${path ? ` ${basename(path)}` : ""}…`, false, null, "nav");
    try {
      const result = await this.requestJson("files", {
        path,
        ...(after ? { after } : {}),
        ...(this.showHidden ? { hidden: "1" } : {}),
      }, abort.signal);
      if (this.disposed || navId !== this.navigation) return;
      this.workdir = result.root ?? null;
      this.roots = this.normalizeRoots(result);
      if (result.status !== "available") {
        const msg = result.message || `${STATUS_MESSAGES[result.status] || "目录读取失败"}${result.path ? `：${result.path}` : ""}`;
        this.pathBoxOpen = true;
        this.setStatus(msg, true, () => void this.files(path, after), "nav");
        this.syncChrome();
        return;
      }
      const resolvedDir = result.path || path;
      const dirChanged = resolvedDir !== this.directory;
      this.directory = resolvedDir;
      this.parentDir = result.parent ?? (resolvedDir !== "/" ? dirnameOf(resolvedDir) : null);
      this.afterCursor = result.after ?? null;
      if (after && !dirChanged) {
        this.items = [...this.items, ...(result.items ?? [])];
      } else {
        this.items = result.items ?? [];
        if (dirChanged) this.filterInput.value = "";
      }
      this.input.value = resolvedDir;
      if (!this.path) {
        this.name.textContent = basename(resolvedDir);
        this.name.title = resolvedDir;
        this.meta.textContent = "";
      }
      this.setStatus("", false, null, "nav");
      this.renderCrumbs();
      this.renderEntries();
      this.syncChrome();
      if (result.open_file) {
        this.pathBoxOpen = false;
        this.syncChrome();
        await this.open(result.open_file, {});
      }
    } catch (error) {
      if (abort.signal.aborted || this.disposed || navId !== this.navigation) return;
      this.pathBoxOpen = true;
      this.setStatus(error instanceof Error ? error.message : String(error), true, () => void this.files(path, after), "nav");
      this.syncChrome();
    } finally {
      if (this.filesAbort === abort) {
        this.filesAbort = null;
        if (!this.docAbort) this.view.setIndicator?.(null);
      }
    }
  }

  private renderEntries(): void {
    this.list.replaceChildren();
    const needle = this.filterInput.value.trim().toLowerCase();
    if (!needle && this.parentDir && this.parentDir !== this.directory) {
      const targetParent = this.parentDir;
      const up = btn("↑ 上一级", () => void this.files(targetParent), "md-entry");
      up.dataset.kind = "parent";
      up.title = targetParent;
      this.list.append(up);
    }
    const filtered = needle
      ? this.items.filter((item) => item.name.toLowerCase().includes(needle))
      : this.items;
    for (const item of filtered) {
      const kind = item.kind ?? (item.name.endsWith("/") ? "directory" : "file");
      const row = btn(item.name, () => {
        if (kind === "directory") {
          void this.files(item.path);
          return;
        }
        if (kind === "other") return;
        if (this.isNarrow()) {
          this.navOpen = false;
          this.syncChrome();
        }
        void this.open(item.path, {});
      }, "md-entry");
      row.title = item.path;
      row.dataset.kind = kind;
      if (kind === "other") {
        row.disabled = true;
        row.append(el("span", "特殊", "md-entry-badge"));
      }
      if (this.path && item.path === this.path) {
        row.setAttribute("aria-current", "true");
      }
      this.list.append(row);
    }
    if (!filtered.length) {
      this.list.append(el("p", needle ? "没有匹配的条目。" : "此目录为空。", "md-empty-list"));
    }
    if (!needle && this.afterCursor) {
      const cursor = this.afterCursor;
      const more = btn("加载更多", () => void this.files(this.directory, cursor), "md-entry");
      more.dataset.more = "true";
      this.list.append(more);
    }
  }

  private async open(
    path: string,
    options: { tail?: boolean; line?: number | null; fragment?: string; allowDirectoryFallback?: boolean },
  ): Promise<"opened" | "directory" | "failed"> {
    if (!path) return "failed";
    const gen = ++this.generation;
    this.docAbort?.abort();
    const abort = this.docAbort = new AbortController();
    this.syntax.cancel();
    if (this.isNarrow()) {
      this.navOpen = false;
      this.syncChrome();
    }
    this.view.setIndicator?.({ busy: true, detail: basename(path) });
    this.setStatus(`正在读取 ${basename(path)}…`, false, null, "stage");
    try {
      const result = await this.requestJson("document", {
        path,
        ...(options.tail ? { tail: "1" } : {}),
      }, abort.signal);
      if (this.disposed || gen !== this.generation) return "failed";
      if (result.status === "not_file" && options.allowDirectoryFallback) {
        this.setStatus("", false, null, "stage");
        return "directory";
      }
      if (result.status !== "available" && result.status !== "binary") {
        const msg = result.message || `${STATUS_MESSAGES[result.status] || "读取失败"}：${path}`;
        this.setStatus(msg, true, () => void this.open(path, options), "stage");
        return "failed";
      }
      const resolvedPath = result.path || path;
      const fileChanged = resolvedPath !== this.path;
      this.path = resolvedPath;
      this.docResult = result;
      this.selectedLines = null;
      this.pendingLine = options.line ?? null;
      if (fileChanged) {
        this.source = Boolean(options.line && isMarkdown(resolvedPath));
      }
      this.name.textContent = basename(resolvedPath);
      this.name.title = resolvedPath;
      this.meta.textContent = formatBytes(result.bytes);
      this.view.setTitle?.(`文件 · ${basename(resolvedPath)}`.slice(0, 80));
      this.setStatus("", false, null, "stage");
      this.renderBanner(result);
      this.renderDocument(gen, abort.signal);
      this.renderEntries();
      this.syncChrome();
      if (options.fragment) this.jump(options.fragment);
      if (this.outlineOpen) this.buildOutline();
      const parent = dirnameOf(resolvedPath);
      if (resolvedPath.startsWith("/") && parent !== this.directory && !this.filesAbort) {
        void this.syncParentDirectory(parent);
      }
      return "opened";
    } catch (error) {
      if (abort.signal.aborted || this.disposed || gen !== this.generation) return "failed";
      this.setStatus(error instanceof Error ? error.message : String(error), true, () => void this.open(path, options), "stage");
      return "failed";
    } finally {
      if (this.docAbort === abort) {
        this.docAbort = null;
        if (!this.filesAbort) this.view.setIndicator?.(null);
      }
    }
  }

  private async syncParentDirectory(parent: string): Promise<void> {
    const navId = ++this.navigation;
    try {
      const result = await this.requestJson("files", { path: parent, ...(this.showHidden ? { hidden: "1" } : {}) });
      if (this.disposed || navId !== this.navigation || result.status !== "available") return;
      this.workdir = result.root ?? null;
      this.roots = this.normalizeRoots(result);
      this.directory = result.path || parent;
      this.parentDir = result.parent ?? (this.directory !== "/" ? dirnameOf(this.directory) : null);
      this.afterCursor = result.after ?? null;
      this.items = result.items ?? [];
      this.input.value = this.directory;
      this.renderCrumbs();
      this.renderEntries();
      this.syncChrome();
    } catch {
      // 附带目录同步失败不影响当前已打开的文件。
    }
  }

  // Dashboard 路由受插件 generation 头保护，图片与下载经 ctx.http.request 取回 Blob 再挂到页面。
  private attachImage(img: HTMLImageElement, targetPath: string, gen: number, signal?: AbortSignal): void {
    img.alt = basename(targetPath);
    void this.ctx.http.request(this.rawUrl(targetPath, false), signal ? { signal } : undefined)
      .then(async (res) => {
        if (!res.ok) throw new Error(`图片读取失败（${res.status}）`);
        return res.blob();
      })
      .then((blob) => {
        if (this.disposed || gen !== this.generation || !img.isConnected) return;
        const url = URL.createObjectURL(blob);
        this.blobUrls.push(url);
        img.src = url;
      })
      .catch(() => {
        if (this.disposed || gen !== this.generation || !img.isConnected) return;
        img.replaceWith(el("span", `[图片未能读取：${basename(targetPath)}]`, "md-image-note"));
      });
  }

  private async downloadCurrent(): Promise<void> {
    if (!this.path || this.downloading) return;
    const targetPath = this.path;
    this.downloading = true;
    this.syncChrome();
    try {
      const response = await this.ctx.http.request(this.rawUrl(targetPath, true));
      if (!response.ok) throw new Error(`下载失败（${response.status}）`);
      const blob = await response.blob();
      if (this.disposed) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = basename(targetPath);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (error) {
      if (!this.disposed) {
        this.setStatus(error instanceof Error ? error.message : "下载失败", true, () => void this.downloadCurrent(), "stage");
      }
    } finally {
      this.downloading = false;
      if (!this.disposed) this.syncChrome();
    }
  }

  private renderBanner(result: Result): void {
    this.banner.replaceChildren();
    if (!result.truncated || !this.path) {
      this.banner.hidden = true;
      return;
    }
    this.banner.hidden = false;
    const sizeText = formatBytes(result.bytes);
    const copy = result.tail
      ? `文件较大${sizeText ? `（共 ${sizeText}）` : ""}，当前显示末尾片段`
      : `文件较大${sizeText ? `（共 ${sizeText}）` : ""}，当前仅显示前 256 KB`;
    const switchBtn = btn(
      result.tail ? "查看开头" : "查看末尾",
      () => void this.open(this.path, { tail: !result.tail }),
      "md-inline-btn",
    );
    this.banner.append(el("span", copy), switchBtn);
  }

  private renderDocument(gen = this.generation, signal?: AbortSignal): void {
    this.syntax.cancel();
    this.clearBlobs();
    this.article.replaceChildren();
    const result = this.docResult;
    if (!result || !this.path) return;

    if (result.mime_type) {
      const box = el("div", "", "md-media-preview");
      const img = el("img");
      this.attachImage(img, this.path, gen, signal);
      const caption = el("p", `${this.path}${result.bytes !== undefined ? ` · ${formatBytes(result.bytes)}` : ""}`, "md-media-caption");
      box.append(img, caption);
      this.article.append(box);
      return;
    }

    if (result.status === "binary") {
      const card = el("div", "", "md-binary-card");
      card.append(
        el("strong", `${basename(this.path)} 是二进制文件`),
        el("p", `路径：${this.path}`),
        el("p", `大小：${formatBytes(result.bytes) || "未知"} · 为避免阻塞 IO，不直接加载二进制正文。`),
      );
      const actions = el("div", "", "md-binary-actions");
      actions.append(btn("复制服务端路径", () => void this.copyPath(), "md-inline-btn"));
      if ((result.bytes ?? 0) <= IMPORT_LIMIT_BYTES) {
        actions.append(btn("下载到本地", () => void this.downloadCurrent(), "md-inline-btn"));
      }
      card.append(actions);
      this.article.append(card);
      return;
    }

    const text = result.text ?? "";
    if (!text) {
      this.article.append(el("p", "这是一个空文件。", "md-empty-file"));
      return;
    }

    if (isMarkdown(this.path) && !this.source) {
      this.renderMarkdown(text, gen, signal);
      return;
    }
    this.renderCode(text);
  }

  private renderMarkdown(text: string, gen: number, signal?: AbortSignal): void {
    const template = document.createElement("template");
    template.innerHTML = micromark(text, { extensions: [gfm()], htmlExtensions: [gfmHtml()] });
    const dir = dirnameOf(this.path);
    const pendingImages: Array<{ img: HTMLImageElement; target: string }> = [];
    template.content.querySelectorAll("img").forEach((image) => {
      const src = (image.getAttribute("src") || "").trim();
      image.removeAttribute("src");
      if (src && !/^[a-z][a-z0-9+.-]*:/i.test(src) && !src.startsWith("//") && /\.(png|jpe?g|webp|gif)$/i.test(src.split("?")[0])) {
        const joined = src.startsWith("/") ? src : `${dir === "/" ? "" : dir}/${src}`;
        pendingImages.push({ img: image, target: normalizePosix(joined) });
      } else {
        image.replaceWith(el("span", `[图片：${image.alt || "外部或不支持的资源"}；已省略]`, "md-image-note"));
      }
    });
    const prose = el("div", "", "md-prose");
    prose.append(template.content);
    this.decorateProse(prose);
    this.article.append(prose);
    for (const item of pendingImages) {
      this.attachImage(item.img, item.target, gen, signal);
    }
  }

  private renderCode(text: string): void {
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
    const pre = el("div", "", "md-source md-code-view");
    pre.setAttribute("role", "region");
    pre.setAttribute("aria-label", `${basename(this.path)} 代码内容`);
    lines.forEach((lineText, idx) => {
      const lineNo = idx + 1;
      const row = el("div", "", "md-code-row");
      row.dataset.line = String(lineNo);
      const gutter = el("button", String(lineNo), "md-code-ln");
      gutter.type = "button";
      gutter.title = "点击选择行，Shift+点击选择范围后可引用到对话";
      gutter.onclick = (event) => this.selectLine(lineNo, event.shiftKey);
      const code = el("span", lineText || " ", "md-code-text");
      row.append(gutter, code);
      pre.append(row);
    });
    this.article.append(pre);
    this.article.scrollTop = 0;
    this.syntax.highlight(this.path, lines, pre);
    if (this.pendingLine !== null) {
      const targetLine = this.pendingLine;
      this.pendingLine = null;
      this.revealLine(targetLine);
    }
  }

  private selectLine(lineNo: number, extend: boolean): void {
    if (extend && this.selectedLines) {
      const start = Math.min(this.selectedLines.start, lineNo);
      const end = Math.max(this.selectedLines.start, lineNo);
      this.selectedLines = { start, end };
    } else if (this.selectedLines && this.selectedLines.start === lineNo && this.selectedLines.end === lineNo) {
      this.selectedLines = null;
    } else {
      this.selectedLines = { start: lineNo, end: lineNo };
    }
    const range = this.selectedLines;
    this.article.querySelectorAll<HTMLElement>(".md-code-row").forEach((row) => {
      const n = Number(row.dataset.line);
      row.classList.toggle("is-selected", Boolean(range && n >= range.start && n <= range.end));
    });
    this.quoteBtn.textContent = range
      ? (range.start === range.end ? `引用 L${range.start}` : `引用 L${range.start}-${range.end}`)
      : "引用";
  }

  private revealLine(lineNo: number): void {
    const row = this.article.querySelector<HTMLElement>(`.md-code-row[data-line="${lineNo}"]`);
    if (!row) return;
    row.scrollIntoView({ block: "center" });
    row.classList.add("is-target");
    window.clearTimeout(this.targetTimer);
    this.targetTimer = window.setTimeout(() => row.classList.remove("is-target"), 1600);
  }

  private quoteSelection(): void {
    if (!this.session || !this.path || typeof this.docResult?.text !== "string") return;
    const browserSelection = window.getSelection()?.toString().trim() ?? "";
    const allLines = this.docResult.text.replace(/\r\n/g, "\n").split("\n");
    let snippet = "";
    let sourceLabel = `文件 · ${this.path}`;
    if (this.selectedLines) {
      const { start, end } = this.selectedLines;
      snippet = allLines.slice(start - 1, Math.min(end, start + 120)).join("\n");
      sourceLabel = `文件 · ${this.path}:${start}${end > start ? `-${end}` : ""}`;
    } else if (browserSelection) {
      snippet = browserSelection.slice(0, 4000);
    } else {
      snippet = allLines.slice(0, 40).join("\n");
      if (allLines.length > 40) sourceLabel = `文件 · ${this.path}:1-40`;
    }
    if (!snippet) return;
    const status = this.view.quote?.({
      session: this.session,
      source: sourceLabel.slice(0, 200),
      text: `\`\`\`\n${snippet.slice(0, 7800)}\n\`\`\``,
    });
    if (status === "inserted") {
      this.quoteBtn.textContent = "已引用";
      window.clearTimeout(this.quoteTimer);
      this.quoteTimer = window.setTimeout(() => {
        this.quoteBtn.textContent = this.selectedLines
          ? (this.selectedLines.start === this.selectedLines.end
            ? `引用 L${this.selectedLines.start}`
            : `引用 L${this.selectedLines.start}-${this.selectedLines.end}`)
          : "引用";
      }, 1500);
    }
  }

  private async copyPath(): Promise<void> {
    const target = this.path || this.directory;
    if (!target) return;
    const ok = await copyString(target);
    this.copyBtn.textContent = ok ? "已复制" : "复制失败";
    window.clearTimeout(this.copyTimer);
    this.copyTimer = window.setTimeout(() => {
      this.copyBtn.textContent = "复制路径";
    }, 1500);
  }

  private async refreshCurrent(): Promise<void> {
    if (this.directory) await this.files(this.directory);
    if (this.path) await this.open(this.path, { tail: Boolean(this.docResult?.tail) });
  }

  private follow(event: MouseEvent): void {
    const link = (event.target as Element).closest("a");
    if (!link) return;
    const href = link.getAttribute("href") || "";
    if (href.startsWith("#")) {
      event.preventDefault();
      this.jump(href.slice(1));
      return;
    }
    if (/^(https?|mailto):/i.test(href)) return;
    event.preventDefault();
    if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return;
    const [raw, fragment] = href.split("#");
    let relative: string;
    try {
      relative = decodeURIComponent(raw);
    } catch {
      this.setStatus("链接路径编码无效。", true, null, "stage");
      return;
    }
    const base = dirnameOf(this.path);
    const nextPath = relative.startsWith("/") ? relative : normalizePosix(`${base === "/" ? "" : base}/${relative}`);
    void this.open(nextPath, { fragment, allowDirectoryFallback: true }).then((res) => {
      if (res === "directory") {
        this.navOpen = true;
        this.syncChrome();
        void this.files(nextPath);
      }
    });
  }

  private decorateProse(container: HTMLElement): void {
    const slugs = new Map<string, number>();
    container.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach((heading) => {
      if (heading.id) return;
      const slug = (heading.textContent || "section").toLowerCase()
        .replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s+/g, "-");
      const count = slugs.get(slug) || 0;
      slugs.set(slug, count + 1);
      heading.id = `md-${count ? `${slug}-${count}` : slug}`;
    });
    container.querySelectorAll("a").forEach((link) => {
      if (/^https?:/i.test(link.getAttribute("href") || "")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    });
    container.querySelectorAll("table").forEach((table) => {
      const wrapper = el("div", "", "md-table");
      wrapper.tabIndex = 0;
      table.replaceWith(wrapper);
      wrapper.append(table);
    });
    container.querySelectorAll("input").forEach((input) => {
      input.disabled = true;
    });
  }

  private jump(fragment: string): void {
    let id: string;
    try {
      id = decodeURIComponent(fragment);
    } catch {
      return;
    }
    const nodes = [...this.article.querySelectorAll<HTMLElement>("[id]")];
    const heading = nodes.find((node) => node.id === `md-${id}`) ?? nodes.find((node) => node.id === id);
    heading?.scrollIntoView({ block: "start" });
  }

  private buildOutline(): void {
    this.outlinePane.replaceChildren();
    if (this.source) {
      this.outlinePane.append(el("p", "切回阅读模式查看文档大纲。"));
      return;
    }
    this.article.querySelectorAll<HTMLElement>(".md-prose :is(h1,h2,h3,h4,h5,h6)").forEach((heading) => {
      const row = btn(heading.textContent || "标题", () => {
        heading.scrollIntoView({ block: "start" });
        this.outlineOpen = false;
        this.syncChrome();
        heading.tabIndex = -1;
        heading.focus({ preventScroll: true });
      });
      row.style.paddingInlineStart = `${8 + (Number(heading.tagName.slice(1)) - 1) * 12}px`;
      this.outlinePane.append(row);
    });
    if (!this.outlinePane.childElementCount) this.outlinePane.append(el("p", "文档没有标题。"));
  }

  dispose(): void {
    this.disposed = true;
    this.generation += 1;
    this.navigation += 1;
    this.filesAbort?.abort();
    this.docAbort?.abort();
    this.syntax.cancel();
    this.clearBlobs();
    window.clearTimeout(this.copyTimer);
    window.clearTimeout(this.quoteTimer);
    window.clearTimeout(this.targetTimer);
    this.stops.forEach((stop) => stop());
  }
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "markdown-reader",
    accepts: ["workspace.document.open.v1", "workspace.file.open.v1"],
    label: "文件",
    order: 25,
    header: true,
    icon: FOLDER_ICON,
    shortcut: "Mod+Shift+E",
    hint: "浏览服务端工作区、会话目录或指定路径文件",
    render(host, _entry, raw) {
      const reader = new Reader(ctx, raw as ConversationTabView);
      host.replaceChildren(reader.root);
      return () => {
        reader.dispose();
        host.replaceChildren();
      };
    },
  }));
}
