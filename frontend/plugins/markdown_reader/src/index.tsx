import { micromark } from "micromark";
import { gfm, gfmHtml } from "micromark-extension-gfm";
import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

type Entry = { name: string; path: string };
type Result = {
  status: string;
  path: string;
  text?: string;
  message?: string;
  root?: string;
  parent?: string;
  after?: string;
  items?: Entry[];
};

const errors: Record<string, string> = {
  not_found: "文件不存在",
  not_directory: "目录不可用",
  permission_denied: "没有读取权限",
  not_file: "只能打开普通文件",
  too_large: "文档超过 1 MiB，请在编辑器中打开",
  invalid_text: "需要 UTF-8 编码的 Markdown 文件",
  offline: "执行主机离线",
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text = "", className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function button(text: string, action: () => void): HTMLButtonElement {
  const node = el("button", text);
  node.type = "button";
  node.onclick = action;
  return node;
}

function toggle(text: string, action: () => void): HTMLButtonElement {
  const node = button(text, action);
  node.setAttribute("aria-pressed", "false");
  return node;
}

function basename(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1) || path;
}

class Reader {
  root = el("section", "", "md-reader");
  private name = el("span", "", "md-name");
  private form = el("form", "", "md-open");
  private input = el("input");
  private nav = el("nav", "", "md-nav");
  private list = el("div");
  private listed = "";
  private article = el("article");
  private note = el("p", "", "md-note");
  private filesToggle = toggle("文件", () => this.toggleFiles());
  private outlineToggle = toggle("目录", () => this.toggleOutline());
  private sourceToggle = toggle("源码", () => this.toggleSource());
  private session: string | null;
  private active: boolean;
  private path = "";
  private directory = "";
  private workdir: string | null | undefined;
  private navMode: "files" | "outline" = "files";
  private text: string | null = null;
  private source = false;
  private generation = 0;
  private navigation = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;
  private loading = false;
  private stops: WebUiDisposer[];

  constructor(private ctx: WebHostContextV1, view: ConversationTabView) {
    this.session = view.session;
    this.active = view.active;
    this.build();
    this.article.addEventListener("click", (event) => this.follow(event));
    this.stops = [
      view.onSessionChange((session) => {
        this.session = session;
        this.reset();
        void this.files();
      }),
      view.onIntent(async (intent) => {
        if (intent.name !== "workspace.document.open.v1") return "invalid";
        const payload = intent.payload as { path?: unknown } | null;
        if (!payload || typeof payload.path !== "string" || !payload.path || payload.path.length > 4096
          || /[\x00-\x1f]/.test(payload.path) || !/\.(md|markdown|mdown)$/i.test(payload.path)
          || /^[a-z][a-z0-9+.-]*:/i.test(payload.path) || payload.path.startsWith("//")) return "invalid";
        await this.open(payload.path);
        return "handled";
      }),
      view.onActiveChange((active) => {
        this.active = active;
        clearTimeout(this.timer);
        if (active) {
          void this.files(this.directory);
          this.schedule();
        }
      }),
    ];
    void this.files();
    this.schedule();
  }

  // 顶栏一行放文档名与三个视图开关；路径输入只在文件面板里出现。
  private build(): void {
    const bar = el("div", "", "md-bar");
    const tools = el("div", "", "md-tools");
    tools.append(this.filesToggle, this.outlineToggle, this.sourceToggle);
    bar.append(this.name, tools);
    this.input.placeholder = "输入绝对路径打开";
    this.input.setAttribute("aria-label", "Markdown 文件路径");
    this.input.autocomplete = "off";
    const open = el("button", "打开");
    open.type = "submit";
    this.form.append(this.input, open);
    this.form.onsubmit = (event) => {
      event.preventDefault();
      void this.open(this.input.value.trim());
    };
    this.note.setAttribute("role", "status");
    this.nav.setAttribute("aria-label", "文档导航");
    this.nav.hidden = true;
    this.article.tabIndex = 0;
    this.article.setAttribute("aria-label", "Markdown 正文");
    this.root.append(bar, this.nav, this.note, this.article);
  }

  // 开关的按下态只反映当前展开的面板与正文模式。
  private syncTools(): void {
    const files = !this.nav.hidden && this.navMode === "files";
    const outline = !this.nav.hidden && this.navMode === "outline";
    this.filesToggle.setAttribute("aria-pressed", String(files));
    this.outlineToggle.setAttribute("aria-pressed", String(outline));
    this.sourceToggle.setAttribute("aria-pressed", String(this.source));
  }

  private toggleFiles(): void {
    const opening = this.nav.hidden || this.navMode !== "files";
    this.navMode = "files";
    if (!opening) {
      this.nav.hidden = true;
      this.syncTools();
      return;
    }
    this.showOpener();
    void this.files(this.directory);
  }

  private toggleOutline(): void {
    if (!this.nav.hidden && this.navMode === "outline") {
      this.nav.hidden = true;
      this.syncTools();
      return;
    }
    this.outline();
  }

  private toggleSource(): void {
    this.source = !this.source;
    this.syncTools();
    this.render();
    if (!this.nav.hidden && this.navMode === "outline") this.outline();
  }

  // 正文里的锚点与相对 Markdown 链接留在阅读器内打开，其余链接交给浏览器。
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
    if (/^[a-z][a-z0-9+.-]*:/i.test(href) || !/\.(md|markdown|mdown)(?:#.*)?$/i.test(href)) {
      this.note.textContent = "阅读器只打开 Markdown 链接和网页链接。";
      return;
    }
    const [raw, fragment] = href.split("#");
    let relative: string;
    try {
      relative = decodeURIComponent(raw);
    } catch {
      this.note.textContent = "链接路径编码无效。";
      return;
    }
    const base = this.path.slice(0, this.path.lastIndexOf("/") + 1);
    void this.open(relative.startsWith("/") ? relative : base + relative, false, fragment);
  }

  private reset(): void {
    this.generation++;
    this.navigation++;
    this.path = "";
    this.directory = "";
    this.text = null;
    this.workdir = undefined;
    this.input.value = "";
    this.name.textContent = "";
    this.name.title = "";
    this.article.replaceChildren();
    this.nav.replaceChildren();
    this.list.replaceChildren();
    this.listed = "";
    this.loading = false;
  }

  private async get(route: string, params: Record<string, string>): Promise<Result> {
    if (!this.session) throw new Error("先选择一个会话，再打开文档。");
    const query = new URLSearchParams({ session: this.session, ...params });
    const response = await this.ctx.http.request(`/api/dashboard/markdown_reader/${route}?${query}`);
    if (!response.ok) throw new Error(`读取失败（${response.status}），请检查路径后重试。`);
    const result = await response.json() as Result;
    if (result.status !== "available") {
      throw new Error(result.message || `${errors[result.status] || "读取失败"}。请检查路径后重试。`);
    }
    return result;
  }

  // 1. 只采用最新一次浏览的结果；2. 工作目录换了就整体重置；3. 大纲模式下只更新状态不重画。
  private async files(path = "", after = "", background = false): Promise<void> {
    const generation = this.generation;
    const navigation = background ? this.navigation : ++this.navigation;
    try {
      const result = await this.get("files", { path, after });
      if (this.disposed || generation !== this.generation || navigation !== this.navigation) return;
      if (this.workdir !== undefined && result.root !== this.workdir) {
        this.reset();
        void this.files();
        return;
      }
      this.workdir = result.root ?? null;
      this.directory = result.path;
      if (this.navMode === "outline" && this.path) return;
      this.listFiles(result, after);
      if (!this.path) {
        this.nav.hidden = false;
        this.syncTools();
        this.note.textContent = result.items?.length
          ? "选择文档，或输入绝对路径。"
          : "此目录没有 Markdown 文档，可进入子目录或输入绝对路径。";
      }
    } catch (error) {
      if (generation === this.generation && navigation === this.navigation) {
        this.note.textContent = String((error as Error).message);
        if (!this.path) this.showOpener();
      }
    }
  }

  // 没有可列的目录时仍给出路径输入，绝对路径是唯一入口。
  private showOpener(): void {
    if (this.navMode === "outline") return;
    if (this.nav.firstChild !== this.form) this.nav.replaceChildren(this.form, this.list);
    this.nav.hidden = false;
    this.syncTools();
  }

  // 输入框常驻面板顶部，轮询只重画下面的列表；结果没变时不动 DOM，保留焦点与悬停。
  private listFiles(result: Result, after: string): void {
    if (this.nav.firstChild !== this.form) this.nav.replaceChildren(this.form, this.list);
    const key = JSON.stringify([result.path, result.items, result.after]);
    if (after) {
      this.list.querySelector("[data-more]")?.remove();
    } else {
      if (key === this.listed) return;
      this.list.replaceChildren(el("div", result.path, "md-directory"));
      if (result.parent && result.path !== result.root) {
        const parent = button("上一级", () => void this.files(result.parent));
        parent.dataset.kind = "parent";
        this.list.append(parent);
      }
    }
    if (!after) this.listed = key;
    for (const item of result.items || []) {
      const directory = item.name.endsWith("/");
      const entry = button(item.name, () => {
        if (directory) {
          void this.files(item.path);
          return;
        }
        this.nav.hidden = true;
        this.syncTools();
        void this.open(item.path);
      });
      entry.title = item.path;
      entry.dataset.kind = directory ? "directory" : "file";
      this.list.append(entry);
    }
    if (result.after) {
      const more = button("加载更多", () => void this.files(result.path, result.after));
      more.dataset.more = "true";
      this.list.append(more);
    }
  }

  // 刷新只在内容或路径变化时重画，并保留滚动位置。
  private async open(path: string, refresh = false, fragment?: string): Promise<void> {
    if (!path || (refresh && this.loading)) return;
    const generation = ++this.generation;
    this.loading = true;
    if (!refresh) this.note.textContent = "正在读取…";
    try {
      const result = await this.get("document", { path });
      if (this.disposed || generation !== this.generation) return;
      const changed = result.text !== this.text || result.path !== this.path;
      this.path = result.path;
      if (!refresh) this.input.value = result.path;
      this.name.textContent = basename(result.path);
      this.name.title = result.path;
      this.text = result.text ?? "";
      this.note.textContent = this.text ? "" : "这是一个空文档。";
      if (!refresh && this.navMode === "files") {
        this.nav.hidden = true;
        this.syncTools();
      }
      if (changed) {
        const position = refresh ? this.article.scrollTop : 0;
        this.render();
        this.article.scrollTop = position;
      }
      if (fragment) this.jump(fragment);
      if (changed && this.navMode === "outline" && !this.nav.hidden) this.outline();
    } catch (error) {
      if (generation === this.generation) {
        this.note.textContent = (refresh ? "更新失败，保留当前内容。" : "") + (error as Error).message;
      }
    } finally {
      if (generation === this.generation) this.loading = false;
    }
  }

  private render(): void {
    if (this.text === null) return;
    if (this.source) {
      this.article.replaceChildren(el("pre", this.text, "md-source"));
      return;
    }
    // micromark escapes raw HTML and rejects dangerous protocols by default. Never enable dangerous HTML.
    const template = document.createElement("template");
    template.innerHTML = micromark(this.text, { extensions: [gfm()], htmlExtensions: [gfmHtml()] });
    // Images are removed in the inert template, before any network request can start.
    template.content.querySelectorAll("img").forEach((image) => {
      image.replaceWith(el("span", `[图片：${image.alt || "未命名"}；暂不加载]`, "md-image-note"));
    });
    this.article.replaceChildren(template.content);
    this.decorate();
  }

  // 标题补稳定 id，外链新开页，表格套横滚框，任务项只读。
  private decorate(): void {
    const slugs = new Map<string, number>();
    this.article.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach((heading) => {
      if (heading.id) return;
      const slug = (heading.textContent || "section").toLowerCase()
        .replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s+/g, "-");
      const count = slugs.get(slug) || 0;
      slugs.set(slug, count + 1);
      heading.id = `md-${count ? `${slug}-${count}` : slug}`;
    });
    this.article.querySelectorAll("a").forEach((link) => {
      if (/^https?:/i.test(link.getAttribute("href") || "")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    });
    this.article.querySelectorAll("table").forEach((table) => {
      const wrapper = el("div", "", "md-table");
      wrapper.tabIndex = 0;
      table.replaceWith(wrapper);
      wrapper.append(table);
    });
    this.article.querySelectorAll("input").forEach((input) => {
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

  private outline(): void {
    this.navMode = "outline";
    this.listed = "";
    this.nav.replaceChildren();
    this.nav.hidden = false;
    this.syncTools();
    if (this.source) {
      this.nav.append(el("p", "切回阅读模式查看目录。"));
      return;
    }
    this.article.querySelectorAll<HTMLElement>("h1,h2,h3,h4,h5,h6").forEach((heading) => {
      const row = button(heading.textContent || "标题", () => {
        heading.scrollIntoView({ block: "start" });
        this.nav.hidden = true;
        this.syncTools();
        heading.tabIndex = -1;
        heading.focus({ preventScroll: true });
      });
      row.style.paddingInlineStart = `${8 + (Number(heading.tagName.slice(1)) - 1) * 12}px`;
      this.nav.append(row);
    });
    if (!this.nav.childElementCount) this.nav.append(el("p", "文档没有标题。"));
  }

  // 标签可见时每 5 秒对照一次磁盘；页面隐藏时跳过请求。
  private schedule(): void {
    clearTimeout(this.timer);
    if (!this.active || this.disposed) return;
    this.timer = setTimeout(async () => {
      if (!document.hidden) {
        await this.files(this.directory, "", true);
        if (this.path) await this.open(this.path, true);
      }
      this.schedule();
    }, 5000);
  }

  dispose(): void {
    this.disposed = true;
    this.generation++;
    this.navigation++;
    clearTimeout(this.timer);
    this.stops.forEach((stop) => stop());
  }
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "markdown-reader",
    accepts: ["workspace.document.open.v1"],
    label: "文档",
    order: 35,
    hint: "阅读 Markdown，支持绝对路径",
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
