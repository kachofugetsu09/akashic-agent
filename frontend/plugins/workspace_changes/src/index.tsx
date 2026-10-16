import type { ConversationIntent, ConversationIntentStatus, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

const API = "/api/dashboard/workspace_changes";
const POLL_MS = 3000;
const FRESH_MS = 8000;
/** workspace.changes.open.v1 的 payload：{ path: string }，先在“未提交”里找这个文件，再找相对默认分支（这一轮可能已经提交）。 */
const OPEN_INTENT = "workspace.changes.open.v1";
/** workspace.file.open.v1（对话回复里点文件路径）：{ path, line? }，先找未提交，再找相对默认分支；只写了文件名时按结尾匹配。 */
const FILE_INTENT = "workspace.file.open.v1";
const STATUS_LABELS = { A: "新增", M: "修改", D: "删除", T: "类型变化" } as const;

interface FileStatus { path: string; status: keyof typeof STATUS_LABELS; added: number | null; removed: number | null }
interface Commit { sha: string; short: string; subject: string; time: string }
interface ScopeResult {
  status: "ok" | "unavailable";
  message?: string;
  base_ref?: string | null;
  base_unknown?: boolean;
  commits?: Commit[];
  scope?: string;
  before?: string;
  after?: string;
  files?: FileStatus[];
  truncated?: boolean;
  submodules?: boolean;
  skipped?: string[];
}
interface Row { kind: "meta" | "hunk" | "add" | "del" | "ctx"; text: string; line: number | null }

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function button(className: string, text: string, onClick: () => void): HTMLButtonElement {
  const node = element("button", className, text);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function counts(file: FileStatus): HTMLElement {
  const node = element("span", "changes-counts");
  if (file.added === null) node.append(element("span", "changes-binary", "二进制"));
  else node.append(element("span", "changes-add", `+${file.added}`), element("span", "changes-del", `−${file.removed}`));
  return node;
}

function totals(files: FileStatus[]): string {
  const added = files.reduce((sum, item) => sum + (item.added ?? 0), 0);
  const removed = files.reduce((sum, item) => sum + (item.removed ?? 0), 0);
  return `${files.length} 个文件 +${added} −${removed}`;
}

// 按目录分组，根目录在前，其余按名字排；组内按文件名排。
function byDirectory(files: FileStatus[]): [string, FileStatus[]][] {
  const groups = new Map<string, FileStatus[]>();
  for (const file of files) {
    const dir = file.path.slice(0, file.path.lastIndexOf("/") + 1);
    groups.set(dir, [...(groups.get(dir) ?? []), file]);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([dir, items]) => [dir, items.sort((a, b) => a.path.localeCompare(b.path))]);
}

// 单文件 patch 拆成行：文件头不显示，hunk 头给出起始行号，之后按新文件行号编号（删除行用旧文件行号）。
// 文件头只在 hunk 之外认：hunk 里删掉的 "-- 注释" 写出来就是 "--- 注释"，不能当文件头吞掉。
function parsePatch(patch: string): Row[] {
  const rows: Row[] = [];
  let left = 0;
  let right = 0;
  let inHunk = false;
  for (const text of patch.split("\n")) {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(text);
    if (text.startsWith("diff --git ")) inHunk = false;
    if (hunk) {
      [left, right] = [Number(hunk[1]), Number(hunk[2])];
      inHunk = true;
      rows.push({ kind: "hunk", text, line: null });
    } else if (!inHunk) {
      if (/^(new|deleted) file mode|^Binary files/.test(text)) rows.push({ kind: "meta", text, line: null });
    } else if (text.startsWith("+")) rows.push({ kind: "add", text, line: right++ });
    else if (text.startsWith("-")) rows.push({ kind: "del", text, line: left++ });
    else if (text.startsWith(" ")) rows.push({ kind: "ctx", text, line: (left++, right++) });
    else if (text) rows.push({ kind: "meta", text, line: null });
  }
  return rows;
}

// 改动标签：工作目录相对 git 的改动，范围可选（未提交 / 相对默认分支 / 单个提交）。左栏文件按目录分组，右栏只看一个文件；
// 可见时轮询，刚变的文件带圆点。点一行写评论，放进对话输入框，由用户决定何时发送。
class ChangesTab {
  readonly root = element("div", "changes-root");
  private readonly scopePicker = element("select", "changes-scope");
  private readonly total = element("span", "changes-total");
  private readonly note = element("p", "changes-note");
  private readonly filter = element("input", "changes-filter");
  private readonly list = element("div", "changes-files");
  private readonly pane = element("section", "changes-pane");
  private scope = "wt";
  private result: ScopeResult | null = null;
  private selected: string | null = null;
  private target: { path: string; line: number } | null = null;
  private shownPatch = "";
  private shownText: string | null = null;
  private readonly seen = new Map<string, string>();
  private readonly fresh = new Map<string, number>();
  private generation = 0;
  private viewVersion = 0;
  private patchRequest = 0;
  private timer = 0;
  private freshTimer = 0;

  constructor(private readonly ctx: WebHostContextV1, private readonly view: ConversationTabView, private session: string | null) {
    this.scopePicker.setAttribute("aria-label", "改动范围");
    this.scopePicker.addEventListener("change", () => this.setScope(this.scopePicker.value));
    this.filter.type = "search";
    this.filter.placeholder = "筛选文件";
    this.filter.setAttribute("aria-label", "筛选文件");
    this.filter.addEventListener("input", () => this.renderList());
    this.list.setAttribute("role", "listbox");
    this.list.setAttribute("aria-label", "改动的文件");
    this.pane.setAttribute("aria-live", "polite");
    const bar = element("div", "changes-bar");
    bar.append(this.scopePicker, this.total);
    const side = element("nav", "changes-side");
    side.append(this.filter, this.list);
    const split = element("div", "changes-split");
    split.append(side, this.pane);
    this.note.hidden = true;
    this.root.append(bar, this.note, split);
    this.root.addEventListener("keydown", (event) => this.onKey(event));
  }

  setSession(session: string | null): void {
    this.viewVersion += 1;
    this.session = session;
    this.scope = "wt";
    this.selected = null;
    this.seen.clear();
    this.fresh.clear();
    this.target = null;
    this.message("读取改动中…");
    void this.load(true);
  }

  // 重新可见时按新一轮比较：不可见期间没有轮询，那段时间变过的文件不算“刚刚变化”。
  setActive(active: boolean): void {
    window.clearTimeout(this.timer);
    if (active) void this.load(true);
  }

  // 意图：切到“未提交”并选中这个文件；它已经提交或还原时如实说明。
  async open(intent: ConversationIntent): Promise<ConversationIntentStatus> {
    const payload = intent.payload as { path?: unknown; line?: unknown } | null;
    if (!payload || typeof payload.path !== "string" || !payload.path) return "invalid";
    const line = Number.isInteger(payload.line) && (payload.line as number) > 0 ? payload.line as number : null;
    const want = payload.path.replace(/^\.\//, "");
    const viewVersion = ++this.viewVersion;
    this.target = null;
    this.message("读取改动中…");
    for (const scope of ["wt", "base"]) {
      this.scope = scope;
      this.selected = want;
      await this.load(true);
      if (viewVersion !== this.viewVersion) return "invalid";
      const found = this.findFile(want);
      // 行号只认找到的这个文件：列表里没有时 render 会退回第一个文件，不能把行号耗在它身上。
      if (!found) continue;
      this.target = line === null ? null : { path: found, line };
      this.select(found, false);
      return "handled";
    }
    // 都没找到：回到默认的“未提交”，再说明原因。
    if (this.scope !== "wt") {
      this.scope = "wt";
      await this.load(true);
      if (viewVersion !== this.viewVersion) return "invalid";
    }
    this.showNote(`${want} 相对 HEAD 和 ${this.result?.base_ref ?? "默认分支"} 都没有改动，可能已经还原。`);
    return "handled";
  }

  // 精确路径优先；回复里常只写文件名或半截路径，结尾唯一匹配时也算。
  private findFile(path: string): string | null {
    const files = this.result?.files ?? [];
    if (files.some((file) => file.path === path)) return path;
    const tail = files.filter((file) => file.path.endsWith(`/${path}`));
    return tail.length === 1 ? tail[0].path : null;
  }

  dispose(): void {
    this.generation += 1;
    this.viewVersion += 1;
    this.patchRequest += 1;
    window.clearTimeout(this.timer);
    window.clearTimeout(this.freshTimer);
  }

  private setScope(scope: string): void {
    this.viewVersion += 1;
    this.scope = scope;
    this.selected = null;
    this.seen.clear();
    this.target = null;
    this.message("读取改动中…");
    void this.load(true);
  }

  // 1. 取文件列表；未提交与本分支两种范围在可见时每 3 秒再取一次，单个提交不会变，不轮询。
  private async load(reset: boolean): Promise<void> {
    const generation = ++this.generation;
    window.clearTimeout(this.timer);
    if (!this.session) return this.message("发送消息后，这里会显示工作目录相对 git 的改动。");
    try {
      const response = await this.get("scope", { scope: this.scope });
      const result = (await response.json()) as ScopeResult;
      if (generation !== this.generation) return;
      if (result.status !== "ok") return this.message(result.message ?? "无法读取改动");
      this.track(result.files ?? [], reset);
      this.result = result;
      this.render();
    } catch (error) {
      if (generation === this.generation) this.message(error instanceof Error ? error.message : String(error), true);
    }
    if (generation === this.generation && this.view.active && (this.scope === "wt" || this.scope === "base"))
      this.timer = window.setTimeout(() => void this.load(false), POLL_MS);
  }

  // 2. 刚变的文件：与上一次列表相比新出现或行数、状态变了的，标记 8 秒。换范围、换会话后的第一次不算。
  private track(files: FileStatus[], reset: boolean): void {
    const now = Date.now();
    const next = new Map(files.map((file) => [file.path, `${file.status}:${file.added}:${file.removed}`]));
    if (!reset && this.seen.size) for (const [path, sign] of next) if (this.seen.get(path) !== sign) this.fresh.set(path, now);
    this.seen.clear();
    for (const [path, sign] of next) this.seen.set(path, sign);
    this.expireFresh();
  }

  // 圆点到时自己消失：最早到期的那个到期时重画左栏。
  private expireFresh(): void {
    const now = Date.now();
    for (const [path, at] of this.fresh) if (now - at >= FRESH_MS || !this.seen.has(path)) this.fresh.delete(path);
    window.clearTimeout(this.freshTimer);
    if (!this.fresh.size) return;
    this.freshTimer = window.setTimeout(() => {
      this.expireFresh();
      this.renderList();
    }, Math.min(...this.fresh.values()) + FRESH_MS - now);
  }

  private render(): void {
    const result = this.result!;
    const files = result.files ?? [];
    this.renderScopes(result);
    this.total.textContent = files.length ? totals(files) : "";
    const notes = [
      result.base_unknown && this.scope === "base" ? "算不出这个分支相对默认分支的基线，只显示未提交的改动。" : "",
      result.truncated ? "改动过多，只列出前 2000 个文件。" : "",
      result.submodules ? "子模块内部的改动不统计。" : "",
      result.skipped?.length ? `${result.skipped.length} 个超过 1 MiB 的未跟踪文件没有计入：${result.skipped.slice(0, 3).join("、")}${result.skipped.length > 3 ? " 等" : ""}。` : "",
    ].filter(Boolean);
    this.note.textContent = notes.join(" ");
    this.note.hidden = !notes.length;
    if (!files.some((file) => file.path === this.selected)) this.selected = byDirectory(files)[0]?.[1][0]?.path ?? null;
    this.renderList();
    void this.renderPatch();
  }

  private renderScopes(result: ScopeResult): void {
    const option = (value: string, label: string) => Object.assign(element("option", "", label), { value });
    const commits = element("optgroup", "");
    commits.label = "本分支的提交";
    commits.append(...(result.commits ?? []).map((commit) => option(commit.sha, `${commit.short} ${commit.subject}`)));
    const base = option("base", `相对 ${result.base_ref ?? "默认分支"}（含提交）`);
    this.scopePicker.replaceChildren(option("wt", "未提交（相对 HEAD）"), base, ...(commits.children.length ? [commits] : []));
    this.scopePicker.value = this.scope.length > 7 && !result.commits?.some((item) => item.sha === this.scope) ? "wt" : this.scope;
  }

  // 3. 左栏：按目录分组，可筛选；当前文件用 aria-selected，刚变的文件带圆点。
  private renderList(): void {
    const files = this.result?.files ?? [];
    const query = this.filter.value.trim().toLowerCase();
    const shown = files.filter((file) => file.path.toLowerCase().includes(query));
    const groups = byDirectory(shown).map(([dir, items]) => {
      const group = element("div", "changes-group");
      if (dir) group.append(element("div", "changes-dir", dir));
      group.append(...items.map((file) => this.fileItem(file)));
      return group;
    });
    const empty = files.length ? "没有匹配的文件" : this.scope === "wt" ? "工作目录没有未提交的改动。" : "这个范围里没有改动。";
    this.list.replaceChildren(...(groups.length ? groups : [element("p", "changes-empty", empty)]));
  }

  private fileItem(file: FileStatus): HTMLButtonElement {
    const item = button("changes-file", "", () => this.select(file.path, false));
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(file.path === this.selected));
    item.title = `${file.path}（${STATUS_LABELS[file.status]}）`;
    const status = element("span", "changes-status", file.status);
    status.dataset.status = file.status;
    item.append(status, element("span", "changes-name", file.path.slice(file.path.lastIndexOf("/") + 1)), counts(file));
    if (this.fresh.has(file.path)) item.append(Object.assign(element("span", "changes-fresh"), { title: "刚刚变化" }));
    return item;
  }

  private select(path: string, focus: boolean): void {
    this.selected = path;
    this.renderList();
    void this.renderPatch();
    if (focus) this.list.querySelector<HTMLElement>("[aria-selected='true']")?.focus();
  }

  // 4. 右栏只看一个文件：按两棵树与路径取 patch，相同时不重取；二进制不取。
  private async renderPatch(): Promise<void> {
    const result = this.result;
    const file = result?.files?.find((item) => item.path === this.selected);
    if (!result || !file) {
      this.patchRequest += 1;
      this.shownPatch = "";
      this.shownText = null;
      delete this.pane.dataset.path;
      return void this.pane.replaceChildren();
    }
    const key = `${result.before}:${result.after}:${file.path}`;
    // 正在写评论时不刷新这份 diff，免得别处的改动把评论框冲掉；关掉评论框后下一次轮询再刷新。
    const writing = this.pane.querySelector(".changes-comment") && this.pane.dataset.path === file.path;
    if (key === this.shownPatch || writing) return this.revealLine();
    this.shownPatch = key;
    const request = ++this.patchRequest;
    const session = this.session ?? "";
    const head = element("header", "changes-pane-head");
    head.append(element("span", "changes-pane-path", file.path), counts(file));
    if (file.added === null) {
      [this.pane.dataset.path, this.shownText] = [file.path, null];
      return void this.pane.replaceChildren(head, element("p", "changes-empty", "二进制文件，不显示内容。"));
    }
    try {
      const response = await this.get("patch", { before: result.before!, after: result.after!, path: file.path });
      const body = (await response.json()) as { status: string; patch?: string; truncated?: boolean; message?: string };
      if (request !== this.patchRequest) return;
      if (body.status !== "ok") {
        // 记下当前显示的是哪个文件的错误：再点回之前的文件时不会因为“内容没变”停在这条错误上。
        [this.pane.dataset.path, this.shownText] = [file.path, null];
        return void this.pane.replaceChildren(head, element("p", "changes-empty", body.message ?? "无法读取 diff"));
      }
      // 内容没变（别的文件变了导致树变了）时不重画，保留滚动位置。
      if (this.pane.dataset.path === file.path && this.shownText === body.patch) return this.revealLine();
      this.pane.dataset.path = file.path;
      this.shownText = body.patch ?? "";
      const rows = element("div", "changes-diff");
      rows.append(...parsePatch(body.patch ?? "").map((row) => this.rowNode(session, file.path, row)));
      const tail = body.truncated ? [element("p", "changes-empty", "diff 过长，只显示前 256 KiB。")] : [];
      this.pane.replaceChildren(head, rows, ...tail);
      this.revealLine();
    } catch (error) {
      if (request !== this.patchRequest) return;
      this.shownPatch = "";
      this.shownText = null;
      this.pane.replaceChildren(head, element("p", "changes-empty is-error", error instanceof Error ? error.message : String(error)));
    }
  }

  // 从回复里带行号打开时：滚到新文件里离这一行最近的 diff 行并短暂高亮；diff 还没换成这个文件时留到下次。
  private revealLine(): void {
    const target = this.target;
    if (target === null || this.pane.dataset.path !== target.path) return;
    this.target = null;
    const line = target.line;
    const rows = [...this.pane.querySelectorAll<HTMLElement>(".changes-row[data-line]:not([data-kind='del'])")];
    const near = rows.reduce<HTMLElement | null>((best, row) =>
      !best || Math.abs(Number(row.dataset.line) - line) < Math.abs(Number(best.dataset.line) - line) ? row : best, null);
    if (!near) return;
    near.scrollIntoView({ block: "center" });
    near.classList.add("is-target");
    window.setTimeout(() => near.classList.remove("is-target"), 1200);
  }

  private rowNode(session: string, path: string, row: Row): HTMLElement {
    const node = element("div", "changes-row");
    node.dataset.kind = row.kind;
    if (row.line !== null) node.dataset.line = String(row.line);
    node.append(element("span", "changes-ln", row.line === null ? "" : String(row.line)), element("span", "changes-code", row.text));
    if (row.line !== null) {
      node.tabIndex = -1;
      node.title = "点击写评论";
      node.addEventListener("click", () => this.comment(session, node, path, row));
    }
    return node;
  }

  // 5. 行评论：写好后放进对话输入框（不自动发送）；Enter 放入，Shift+Enter 换行，Esc 取消。
  private comment(session: string, node: HTMLElement, path: string, row: Row): void {
    // 引用绑定 diff 的来源会话；点击或提交时当前会话可能已改变。
    this.pane.querySelector(".changes-comment")?.remove();
    const box = element("div", "changes-comment");
    const field = element("textarea", "changes-comment-input");
    field.placeholder = `对 ${path} 第 ${row.line} 行的意见`;
    field.setAttribute("aria-label", field.placeholder);
    const status = element("span", "changes-comment-status", "Enter 放进输入框 · Shift+Enter 换行 · Esc 取消");
    const submit = () => {
      const text = field.value.trim();
      if (!text) return;
      // 来源行带上路径、行号与这一行代码，整行不超过宿主的 200 字符上限；正文只放评论。
      const result = this.view.quote({ session, text,
        source: `改动 · ${path}:${row.line} · ${row.text.trim()}`.slice(0, 200) });
      if (result === "inserted") return box.remove();
      status.textContent = result === "stale" ? "会话已经切换，没有放进去。" : "没能放进输入框（内容过长或对话不可用）。";
    };
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); submit(); }
      else if (event.key === "Escape") { event.preventDefault(); box.remove(); node.focus(); }
    });
    const actions = element("div", "changes-comment-actions");
    actions.append(status, button("changes-action", "取消", () => box.remove()), button("changes-action is-primary", "放进输入框", submit));
    box.append(field, actions);
    node.after(box);
    field.focus();
  }

  // j / k 切到下一个、上一个文件（按左栏的顺序）；在输入框里不接。
  private onKey(event: KeyboardEvent): void {
    if (event.ctrlKey || event.metaKey || event.altKey || (event.key !== "j" && event.key !== "k")) return;
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
    const query = this.filter.value.trim().toLowerCase();
    const order = byDirectory((this.result?.files ?? []).filter((file) => file.path.toLowerCase().includes(query))).flatMap(([, items]) => items);
    if (!order.length) return;
    const at = order.findIndex((file) => file.path === this.selected);
    const next = order[Math.min(order.length - 1, Math.max(0, at + (event.key === "j" ? 1 : -1)))];
    event.preventDefault();
    this.select(next.path, true);
  }

  private showNote(text: string): void {
    this.note.textContent = text;
    this.note.hidden = false;
  }

  private message(text: string, failed = false): void {
    this.patchRequest += 1;
    this.result = null;
    this.total.textContent = "";
    this.note.hidden = true;
    this.list.replaceChildren();
    const node = element("p", failed ? "changes-empty is-error" : "changes-empty", text);
    node.setAttribute("role", failed ? "alert" : "status");
    this.pane.replaceChildren(node);
    this.shownPatch = "";
    this.shownText = null;
    delete this.pane.dataset.path;
  }

  private async get(path: string, params: Record<string, string>): Promise<Response> {
    const query = new URLSearchParams({ session: this.session ?? "", ...params });
    const response = await this.ctx.http.request(`${API}/${path}?${query}`);
    if (!response.ok) throw new Error(`读取改动失败（${response.status}）`);
    return response;
  }
}

function renderChanges(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onActiveChange !== "function" || typeof view.onIntent !== "function")
    throw new Error("改动标签缺少 conversation.tools.v2 view");
  const tab = new ChangesTab(ctx, view, view.session);
  host.replaceChildren(tab.root);
  const stops = [
    view.onSessionChange((session) => tab.setSession(session)),
    view.onActiveChange((active) => tab.setActive(active)),
    view.onIntent((intent) => intent.name === OPEN_INTENT || intent.name === FILE_INTENT ? tab.open(intent) : "invalid"),
  ];
  return () => {
    for (const stop of stops) stop();
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "workspace-changes", label: "改动", order: 30, hint: "工作目录相对 git 的改动", shortcut: "Mod+Shift+G", accepts: [OPEN_INTENT, FILE_INTENT],
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3v12M18 9v12M6 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM18 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM6 15c0-4 12-2 12-6"/></svg>',
    render(host, _entryView, props) { return renderChanges(ctx, host, props); },
  }));
}
