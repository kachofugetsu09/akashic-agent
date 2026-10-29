import { micromark } from "micromark";
import { gfm, gfmHtml } from "micromark-extension-gfm";
import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

type Result = { status: string; path: string; text?: string; message?: string; root?: string; parent?: string; after?: string; items?: { name: string; path: string }[] };
const errors: Record<string, string> = { not_found: "文件不存在", not_directory: "目录不可用", permission_denied: "没有读取权限", not_file: "只能打开普通文件", too_large: "文档超过 1 MiB，请在编辑器中打开", invalid_text: "需要 UTF-8 编码的 Markdown 文件", offline: "执行主机离线" };
function el<K extends keyof HTMLElementTagNameMap>(tag: K, text = ""): HTMLElementTagNameMap[K] { const node = document.createElement(tag); node.textContent = text; return node; }
function button(text: string, action: () => void) { const node = el("button", text); node.type = "button"; node.onclick = action; return node; }

class Reader {
  root = el("section");
  private bar = el("form");
  private input = el("input");
  private nav = el("nav");
  private article = el("article");
  private note = el("p");
  private title = el("div");
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
    this.session = view.session; this.active = view.active;
    this.root.className = "md-reader";
    this.input.placeholder = "文档路径，支持绝对路径"; this.input.setAttribute("aria-label", "Markdown 文件路径");
    this.input.autocomplete = "off";
    const open = el("button", "打开"); open.type = "submit";
    this.bar.append(this.input, open);
    this.bar.onsubmit = (event) => { event.preventDefault(); void this.open(this.input.value.trim()); };
    const tools = el("div"); tools.className = "md-tools";
    const files = button("文件", () => { this.navMode = "files"; this.nav.hidden = !this.nav.hidden; if (!this.nav.hidden) void this.files(this.directory); });
    const outline = button("目录", () => this.outline());
    const source = button("源码", () => { this.source = !this.source; source.textContent = this.source ? "阅读" : "源码"; source.setAttribute("aria-pressed", String(this.source)); this.render(); });
    tools.append(files, outline, source, button("刷新", () => { void this.files(this.directory); if (this.path) void this.open(this.path, true); }));
    this.title.className = "md-path"; this.note.className = "md-note"; this.note.setAttribute("role", "status");
    this.nav.setAttribute("aria-label", "文档导航"); this.nav.hidden = true;
    this.article.tabIndex = 0; this.article.setAttribute("aria-label", "Markdown 正文");
    this.root.append(this.bar, tools, this.title, this.note, this.nav, this.article);
    this.article.addEventListener("click", (event) => {
      const link = (event.target as Element).closest("a"); if (!link) return;
      const href = link.getAttribute("href") || "";
      if (href.startsWith("#")) { event.preventDefault(); document.getElementById(href.slice(1))?.scrollIntoView({ block: "start" }); }
      else if (!/^[a-z][a-z0-9+.-]*:/i.test(href) && /\.(md|markdown|mdown)(?:#.*)?$/i.test(href)) {
        event.preventDefault(); const [relative] = href.split("#");
        void this.open(relative.startsWith("/") ? relative : this.path.slice(0, this.path.lastIndexOf("/") + 1) + decodeURIComponent(relative));
      }
    });
    this.stops = [view.onSessionChange((session) => { this.session = session; this.reset(); void this.files(); }), view.onActiveChange((active) => { this.active = active; clearTimeout(this.timer); if (active) { void this.files(this.directory); this.schedule(); } })];
    void this.files(); this.schedule();
  }
  private reset() { this.generation++; this.navigation++; this.path = ""; this.directory = ""; this.text = null; this.workdir = undefined; this.input.value = ""; this.title.textContent = ""; this.article.replaceChildren(); this.nav.replaceChildren(); this.loading = false; }
  private async get(route: string, params: Record<string, string>): Promise<Result> {
    if (!this.session) throw new Error("先选择一个会话，再打开文档。");
    const response = await this.ctx.http.request(`/api/dashboard/markdown_reader/${route}?${new URLSearchParams({ session: this.session, ...params })}`);
    if (!response.ok) throw new Error(`读取失败（${response.status}），请检查路径后重试。`);
    const result = await response.json() as Result;
    if (result.status !== "available") throw new Error(result.message || `${errors[result.status] || "读取失败"}。请检查路径后重试。`);
    return result;
  }
  private async files(path = "", after = "") {
    const generation = this.generation, navigation = ++this.navigation;
    try {
      const result = await this.get("files", { path, after });
      if (this.disposed || generation !== this.generation || navigation !== this.navigation) return;
      if (this.workdir !== undefined && result.root !== this.workdir) { this.reset(); void this.files(); return; }
      this.workdir = result.root ?? null; this.directory = result.path;
      if (this.navMode === "outline" && this.path) return;
      if (!after) this.nav.replaceChildren();
      if (!after && result.parent && result.path !== result.root) this.nav.append(button("上一级", () => void this.files(result.parent)));
      for (const item of result.items || []) {
        const entry = button(item.name, () => { if (item.name.endsWith("/")) void this.files(item.path); else { this.nav.hidden = true; void this.open(item.path); } });
        entry.title = item.path; this.nav.append(entry);
      }
      if (after) this.nav.querySelector("[data-more]")?.remove();
      if (result.after) { const more = button("加载更多", () => void this.files(result.path, result.after)); more.dataset.more = "true"; this.nav.append(more); }
      if (!this.path) { this.nav.hidden = false; this.note.textContent = result.items?.length ? "选择文档，或输入绝对路径。" : "此目录没有 Markdown 文档，可进入子目录或输入绝对路径。"; }
    } catch (error) { if (generation === this.generation && navigation === this.navigation) this.note.textContent = String((error as Error).message); }
  }
  private async open(path: string, refresh = false) {
    if (!path || (refresh && this.loading)) return;
    const generation = ++this.generation; this.loading = true;
    if (!refresh) this.note.textContent = "正在读取…";
    try {
      const result = await this.get("document", { path });
      if (this.disposed || generation !== this.generation) return;
      const changed = result.text !== this.text || result.path !== this.path;
      this.path = result.path; this.input.value = result.path; this.title.textContent = result.path; this.title.title = result.path;
      this.text = result.text ?? ""; this.note.textContent = this.text ? "" : "这是一个空文档。";
      if (changed) { const position = refresh ? this.article.scrollTop : 0; this.render(); this.article.scrollTop = position; }
    } catch (error) { if (generation === this.generation) this.note.textContent = (refresh ? "更新失败，保留当前内容。" : "") + (error as Error).message; }
    finally { if (generation === this.generation) this.loading = false; }
  }
  private render() {
    if (this.text === null) return;
    if (this.source) { const pre = el("pre", this.text); pre.className = "md-source"; this.article.replaceChildren(pre); return; }
    // micromark escapes raw HTML and rejects dangerous protocols by default. Never enable dangerous HTML.
    const template = document.createElement("template");
    template.innerHTML = micromark(this.text, { extensions: [gfm()], htmlExtensions: [gfmHtml()] });
    template.content.querySelectorAll("img").forEach((image) => {
      const label = el("span", `[图片：${image.alt || "未命名"}；暂不加载]`); label.className = "md-image-note"; image.replaceWith(label);
    });
    this.article.replaceChildren(template.content);
    const slugs = new Map<string, number>();
    this.article.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach((heading) => {
      const slug = (heading.textContent || "section").toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s+/g, "-");
      const count = slugs.get(slug) || 0; slugs.set(slug, count + 1);
      heading.id = count ? `${slug}-${count}` : slug;
    });
    this.article.querySelectorAll("a").forEach((link) => { if (/^https?:/i.test(link.getAttribute("href") || "")) { link.target = "_blank"; link.rel = "noopener noreferrer"; } });
    // Images were removed in the inert template, before any network requests can start.
    this.article.querySelectorAll("table").forEach((table) => { const wrapper = el("div"); wrapper.className = "md-table"; wrapper.tabIndex = 0; table.replaceWith(wrapper); wrapper.append(table); });
    this.article.querySelectorAll("input").forEach((input) => { input.disabled = true; });
  }
  private outline() {
    this.navMode = "outline";
    this.nav.replaceChildren(); this.nav.hidden = false;
    if (this.source) { this.nav.append(el("p", "切回阅读模式查看目录。")); return; }
    this.article.querySelectorAll<HTMLElement>("h1,h2,h3,h4,h5,h6").forEach((heading) => {
      const row = button(heading.textContent || "标题", () => { heading.scrollIntoView({ block: "start" }); this.nav.hidden = true; heading.tabIndex = -1; heading.focus({ preventScroll: true }); });
      row.style.paddingInlineStart = `${8 + (Number(heading.tagName.slice(1)) - 1) * 12}px`; this.nav.append(row);
    });
    if (!this.nav.childElementCount) this.nav.append(el("p", "文档没有标题。"));
  }
  private schedule() { clearTimeout(this.timer); if (!this.active || this.disposed) return; this.timer = setTimeout(async () => { if (!document.hidden) { await this.files(this.directory); if (this.path) await this.open(this.path, true); } this.schedule(); }, 5000); }
  dispose() { this.disposed = true; this.generation++; this.navigation++; clearTimeout(this.timer); this.stops.forEach((stop) => stop()); }
}
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "markdown-reader", label: "文档", order: 35, hint: "阅读 Markdown，支持绝对路径",
    render(host, _entry, raw) { const reader = new Reader(ctx, raw as ConversationTabView); host.replaceChildren(reader.root); return () => { reader.dispose(); host.replaceChildren(); }; },
  }));
}
