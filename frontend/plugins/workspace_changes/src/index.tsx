import type { ConversationIntent, ConversationIntentStatus, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

const API = "/api/dashboard/workspace_changes";
const POLL_MS = 4000;
/** workspace.changes.open.v1 的 payload：{ turn_id: string }，打开“本轮”并选中这一轮。 */
const OPEN_INTENT = "workspace.changes.open.v1";

interface FileChange { path: string; added: number | null; removed: number | null }
interface Failure { reason: string; message: string }
interface TurnRecord {
  turn_id: string;
  phase: "started" | "ended";
  status: "complete" | "quiet" | "abandoned" | null;
  window: { started_at: string };
  files: FileChange[];
  patch?: string | null;
  truncated: boolean;
  baseline_late: boolean;
  end_late: boolean;
  repo: { submodules: boolean } | null;
  error: Failure | null;
}
interface Uncommitted { status: "ok" | "unavailable"; files?: FileChange[]; patch?: string | null; truncated?: boolean; submodules?: boolean; message?: string }
type Mode = "turn" | "uncommitted";

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

function totals(files: FileChange[], uncertain = false): string {
  if (!files.length) return uncertain ? "没有统计到改动" : "没有改动";
  const added = files.reduce((sum, item) => sum + (item.added ?? 0), 0);
  const removed = files.reduce((sum, item) => sum + (item.removed ?? 0), 0);
  return `${files.length} 个文件 +${added} −${removed}`;
}

function clock(iso: string): string {
  const time = new Date(iso);
  const today = time.toDateString() === new Date().toDateString();
  return time.toLocaleString(undefined, today ? { hour: "2-digit", minute: "2-digit" } : { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function turnLabel(turn: TurnRecord): string {
  const state = turn.phase === "started" ? "进行中" : turn.error ? turn.error.message : totals(turn.files, turn.baseline_late || turn.end_late);
  const ending = turn.status === "abandoned" ? "（已中止）" : "";
  return `${clock(turn.window.started_at)} · ${state}${ending}`;
}

// 统计口径的提示：窗口标记、截断与子模块，都写成可读的一句话。
function caveats(source: { baseline_late?: boolean; end_late?: boolean; truncated?: boolean; submodules?: boolean }): string[] {
  const notes: string[] = [];
  if (source.baseline_late) notes.push("起始快照晚于本轮第一次工具调用，可能遗漏改动。");
  if (source.end_late) notes.push("结束快照偏晚（下一轮已开始、后台命令仍在运行或已中止），可能包含窗口外的改动。");
  if (source.truncated) notes.push("改动过多，只列出前 2000 个文件或不显示 diff。");
  if (source.submodules) notes.push("子模块内部的改动不统计。");
  return notes;
}

function lineKind(line: string): string {
  if (line.startsWith("+++ ") || line.startsWith("--- ")) return " is-meta";
  if (line.startsWith("@@")) return " is-hunk";
  if (line.startsWith("+")) return " is-add";
  if (line.startsWith("-")) return " is-del";
  return "";
}

// 把整份 patch 按 "diff --git" 切成每个文件一段，键为 b/ 之后的路径。
function splitPatch(patch: string): Map<string, string[]> {
  const sections = new Map<string, string[]>();
  let current: string[] | null = null;
  for (const line of patch.split("\n")) {
    const head = /^diff --git a\/.* b\/(.*)$/.exec(line);
    if (head) sections.set(head[1], current = []);
    else if (current && !line.startsWith("index ")) current.push(line);
  }
  return sections;
}

// 目录可以省略，文件名始终完整；完整路径放在 title 里。
function pathParts(path: string): HTMLElement[] {
  const slash = path.lastIndexOf("/");
  return [element("span", "changes-dir", path.slice(0, slash + 1)), element("span", "changes-name", path.slice(slash + 1))];
}

function fileDiff(file: FileChange, lines: string[] | undefined): HTMLDetailsElement {
  const details = element("details", "changes-file");
  details.open = true;
  details.dataset.path = file.path;
  const summary = element("summary", "changes-file-head");
  summary.title = file.path;
  summary.append(...pathParts(file.path), counts(file));
  details.append(summary);
  const body = element("pre", "changes-diff");
  if (file.added === null) body.append(element("span", "changes-line is-meta", "二进制文件，不显示内容"));
  else if (!lines) body.append(element("span", "changes-line is-meta", "diff 未保存（超出大小上限）"));
  else for (const line of lines) body.append(element("span", `changes-line${lineKind(line)}`, line + "\n"));
  details.append(body);
  return details;
}

function counts(file: FileChange): HTMLElement {
  const node = element("span", "changes-counts");
  if (file.added === null) node.append(element("span", "changes-binary", "二进制"));
  else node.append(element("span", "changes-add", `+${file.added}`), element("span", "changes-del", `−${file.removed}`));
  return node;
}

// 改动标签：本轮（每轮窗口内的差异）与未提交（工作目录对 HEAD）。可见时才轮询，会话切换后整体换掉。
class ChangesTab {
  readonly root = element("div", "changes-root");
  private readonly turnButton = button("changes-mode", "本轮", () => this.setMode("turn"));
  private readonly uncommittedButton = button("changes-mode", "未提交", () => this.setMode("uncommitted"));
  private readonly picker = element("select", "changes-picker");
  private readonly refresh = button("changes-refresh", "刷新", () => void this.load(true));
  private readonly body = element("div", "changes-body");
  private mode: Mode = "turn";
  private turns: TurnRecord[] = [];
  private selected: string | null = null;
  private following = true;
  private shown = "";
  private generation = 0;
  private timer = 0;

  constructor(private readonly ctx: WebHostContextV1, private session: string | null, private active: boolean) {
    const modes = element("div", "changes-modes");
    modes.setAttribute("role", "group");
    modes.setAttribute("aria-label", "统计范围");
    modes.append(this.turnButton, this.uncommittedButton);
    this.picker.setAttribute("aria-label", "选择轮次");
    this.picker.addEventListener("change", () => {
      this.selected = this.picker.value;
      this.following = this.picker.selectedIndex === 0;
      void this.showTurn();
    });
    const bar = element("div", "changes-bar");
    bar.append(modes, this.picker, this.refresh);
    this.body.setAttribute("aria-live", "polite");
    this.root.append(bar, this.body);
    this.setMode("turn");
  }

  setSession(session: string | null): void {
    this.session = session;
    this.turns = [];
    this.selected = null;
    this.following = true;
    this.shown = "";
    void this.load(true);
  }

  setActive(active: boolean): void {
    this.active = active;
    this.schedule();
    if (active) void this.load(false);
  }

  // 意图：切到本轮并选中指定的一轮；不在列表里（已被覆盖）时如实说明。
  async open(intent: ConversationIntent): Promise<ConversationIntentStatus> {
    const payload = intent.payload as { turn_id?: unknown } | null;
    if (!payload || typeof payload.turn_id !== "string" || !payload.turn_id) return "invalid";
    this.mode = "turn";
    this.selected = payload.turn_id;
    this.following = false;
    this.shown = "";
    this.renderModes();
    await this.load(true);
    return "handled";
  }

  dispose(): void {
    this.generation += 1;
    window.clearTimeout(this.timer);
  }

  private setMode(mode: Mode): void {
    this.mode = mode;
    this.shown = "";
    this.renderModes();
    void this.load(true);
  }

  private renderModes(): void {
    this.turnButton.setAttribute("aria-pressed", String(this.mode === "turn"));
    this.uncommittedButton.setAttribute("aria-pressed", String(this.mode === "uncommitted"));
    this.picker.hidden = this.mode !== "turn" || !this.turns.length;
  }

  // 本轮模式可见时每 4 秒取一次列表，轮次结束后自动出现；未提交只在进入和点刷新时统计。
  private schedule(): void {
    window.clearTimeout(this.timer);
    if (this.active && this.mode === "turn" && this.session) this.timer = window.setTimeout(() => void this.load(false).finally(() => this.schedule()), POLL_MS);
  }

  private async load(explicit: boolean): Promise<void> {
    const generation = ++this.generation;
    if (!this.session) return this.message("发送消息后，这里会显示每轮对工作目录的改动。");
    if (this.mode === "uncommitted") return this.loadUncommitted(generation);
    try {
      const response = await this.get(`${API}/turns`, {});
      if (generation !== this.generation) return;
      this.turns = ((await response.json()) as { turns: TurnRecord[] }).turns;
      if (generation !== this.generation) return;
      this.renderPicker();
      await this.showTurn(explicit);
    } catch (error) {
      if (generation === this.generation) this.message(error instanceof Error ? error.message : String(error), true);
    }
    this.schedule();
  }

  private renderPicker(): void {
    if (this.following || !this.selected) this.selected = this.turns[0]?.turn_id ?? this.selected;
    const options = this.turns.map((turn) => {
      const option = element("option", "", turnLabel(turn));
      option.value = turn.turn_id;
      return option;
    });
    this.picker.replaceChildren(...options);
    if (this.selected) this.picker.value = this.selected;
    this.renderModes();
  }

  // 选中轮次变化或它从进行中变为结束时才重新取详情（含 patch）。
  private async showTurn(force = true): Promise<void> {
    const generation = this.generation;
    const summary = this.turns.find((turn) => turn.turn_id === this.selected);
    const key = `${this.selected}:${summary?.phase}`;
    if (!force && key === this.shown) return;
    if (!this.selected) return this.message("还没有记录到改动。会话工作目录是 git 仓库时，每轮结束后在这里列出这一轮改了什么。");
    if (summary?.phase === "started") return this.show(key, "这一轮还在进行，结束后统计。");
    const response = await this.get(`${API}/turn`, { turn: this.selected });
    if (generation !== this.generation) return;
    if (response.status === 404) return this.show(key, "这一轮的记录已被较新的轮次覆盖，或者在停用、重启期间没有记录。");
    const turn = (await response.json()) as TurnRecord;
    if (generation !== this.generation) return;
    this.shown = key;
    if (turn.error) return this.message(turn.error.message);
    this.renderChanges(turn.files, turn.patch ?? null, caveats({ ...turn, submodules: turn.repo?.submodules }));
  }

  private async loadUncommitted(generation: number): Promise<void> {
    this.message("统计中…");
    try {
      const response = await this.get(`${API}/uncommitted`, {});
      const result = (await response.json()) as Uncommitted;
      if (generation !== this.generation) return;
      if (result.status !== "ok") return this.message(result.message ?? "无法统计");
      this.renderChanges(result.files ?? [], result.patch ?? null, caveats(result));
    } catch (error) {
      if (generation === this.generation) this.message(error instanceof Error ? error.message : String(error), true);
    }
  }

  // 顶部是总计、提示与文件列表（点一下跳到该文件），下面是逐文件的 diff。
  private renderChanges(files: FileChange[], patch: string | null, notes: string[]): void {
    const head = element("div", "changes-summary");
    head.append(element("p", "changes-total", totals(files, notes.length > 0)), ...notes.map((note) => element("p", "changes-note", note)));
    const list = element("ul", "changes-files");
    const sections = patch === null ? new Map<string, string[]>() : splitPatch(patch);
    const diffs = files.map((file) => fileDiff(file, sections.get(file.path)));
    files.forEach((file, index) => {
      const item = element("li", "");
      const jump = button("changes-jump", "", () => {
        diffs[index].open = true;
        diffs[index].scrollIntoView({ block: "start" });
      });
      jump.title = file.path;
      jump.append(...pathParts(file.path), counts(file));
      item.append(jump);
      list.append(item);
    });
    this.body.replaceChildren(head, ...(files.length ? [list, ...diffs] : []));
  }

  private show(key: string, text: string): void {
    this.shown = key;
    this.message(text);
  }

  private message(text: string, failed = false): void {
    const node = element("p", failed ? "changes-message is-error" : "changes-message", text);
    node.setAttribute("role", failed ? "alert" : "status");
    this.body.replaceChildren(node);
  }

  private async get(path: string, params: Record<string, string>): Promise<Response> {
    const query = new URLSearchParams({ session: this.session ?? "", ...params });
    const response = await this.ctx.http.request(`${path}?${query}`);
    if (!response.ok && response.status !== 404) throw new Error(`读取改动失败（${response.status}）`);
    return response;
  }
}

function renderChanges(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onActiveChange !== "function" || typeof view.onIntent !== "function")
    throw new Error("改动标签缺少 conversation.tools.v2 view");
  const tab = new ChangesTab(ctx, view.session, view.active);
  host.replaceChildren(tab.root);
  const stops = [
    view.onSessionChange((session) => tab.setSession(session)),
    view.onActiveChange((active) => tab.setActive(active)),
    view.onIntent((intent) => intent.name === OPEN_INTENT ? tab.open(intent) : "invalid"),
  ];
  return () => {
    for (const stop of stops) stop();
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "workspace-changes", label: "改动", order: 30, accepts: [OPEN_INTENT],
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3v12M18 9v12M6 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM18 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM6 15c0-4 12-2 12-6"/></svg>',
    render(host, _entryView, props) { return renderChanges(ctx, host, props); },
  }));
}
