// 终端标签：只读展示 shell 工具记下的命令，按所属输入分组；数据与停止都走本插件自己的 dashboard 路由。
import type { ConversationIntent, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

interface CommandRow {
  readonly slot: number;
  readonly call_id: string;
  readonly input_id: string;
  readonly input_preview: string;
  readonly command: string;
  readonly cwd: string;
  readonly background: boolean;
  readonly state: "running" | "exited" | "timeout" | "terminated" | "unknown";
  readonly exit_code: number | null;
  readonly started_at: number;
  readonly ended_at: number | null;
  readonly output_truncated: boolean;
  readonly stoppable: boolean;
  readonly output?: string;
}

const POLL_MS = 2000;
const QUOTE_LIMIT = 8000;
const STATE_LABEL: Record<CommandRow["state"], string> = {
  running: "运行中", exited: "已结束", timeout: "超时", terminated: "已停止", unknown: "结局未知",
};

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text) node.textContent = text;
  return node;
}

function duration(row: CommandRow): string {
  const seconds = Math.max(0, (row.ended_at ?? Date.now() / 1000) - row.started_at);
  return seconds < 1 ? "<1s" : seconds < 60 ? `${Math.round(seconds)}s` : `${Math.floor(seconds / 60)}m${Math.round(seconds % 60)}s`;
}

function verdict(row: CommandRow): { text: string; tone: string } {
  if (row.state === "exited") return row.exit_code === 0 ? { text: "退出 0", tone: "ok" } : { text: `退出 ${row.exit_code}`, tone: "error" };
  return { text: STATE_LABEL[row.state], tone: row.state === "running" ? "busy" : "warn" };
}

class TerminalTab {
  readonly root = element("div", "terminal-view");
  private readonly list = element("div", "terminal-list");
  private readonly empty = element("p", "terminal-empty");
  private readonly expanded = new Map<number, string | null>();
  private rows: CommandRow[] = [];
  private signature = "";
  private timer = 0;
  private generation = 0;

  constructor(private readonly ctx: WebHostContextV1, private readonly view: ConversationTabView) {
    const head = element("div", "terminal-head");
    head.append(element("strong", "", "Agent 执行的命令"), element("span", "terminal-note", "显示 Agent 已读取的输出"));
    this.empty.setAttribute("role", "status");
    this.root.append(head, this.empty, this.list);
  }

  // 1. 会话或可见性变化时重置轮询；不可见时不请求。
  sync() {
    if (this.view.session && this.view.active) {
      void this.refresh();
      return;
    }
    window.clearTimeout(this.timer);
    this.generation++;
    if (!this.view.session) this.show([], "选择或开始一个会话后，这里会列出 Agent 在这个会话里执行过的命令。");
  }

  private refresh(): Promise<void> {
    window.clearTimeout(this.timer);
    return this.poll(++this.generation);
  }

  private async poll(generation: number) {
    const session = this.view.session;
    try {
      const response = await this.ctx.http.request(`/api/dashboard/standard-tools/commands?session=${encodeURIComponent(session ?? "")}`);
      if (!response.ok) throw new Error(`读取命令记录失败（${response.status}）`);
      const body = await response.json() as { session: string; commands: CommandRow[] };
      if (generation !== this.generation || body.session !== this.view.session) return;
      this.show(body.commands, "这个会话还没有执行过命令。");
      for (const row of body.commands)
        if (this.expanded.has(row.slot) && (row.state === "running" || this.expanded.get(row.slot) === null)) void this.loadOutput(row.slot);
    } catch (error) {
      if (generation === this.generation) this.show(this.rows, error instanceof Error ? error.message : String(error));
    }
    if (generation === this.generation && this.view.active) this.timer = window.setTimeout(() => void this.poll(generation), POLL_MS);
  }

  private show(rows: CommandRow[], emptyText: string) {
    const signature = JSON.stringify(rows.map(row => [row.slot, row.call_id, row.state, row.exit_code, row.ended_at]));
    this.empty.textContent = rows.length ? "" : emptyText;
    this.empty.hidden = rows.length > 0;
    if (signature === this.signature && rows.length) return this.refreshDurations();
    this.signature = signature;
    this.rows = rows;
    this.render();
  }

  private refreshDurations() {
    for (const node of this.list.querySelectorAll<HTMLElement>("[data-running] .terminal-time")) {
      const row = this.rows.find(item => String(item.slot) === node.dataset.slot);
      if (row) node.textContent = duration(row);
    }
  }

  // 2. 按所属输入分组；组头是那条输入的开头，帮助对上对话里的哪一问。
  private render() {
    const groups = new Map<string, CommandRow[]>();
    for (const row of this.rows) groups.set(row.input_id, [...(groups.get(row.input_id) ?? []), row]);
    this.list.replaceChildren(...[...groups.values()].map(rows => {
      const group = element("section", "terminal-group");
      group.append(element("h3", "terminal-group-title", rows[0].input_preview || "不是由对话输入发起（例如后台任务唤醒）"), ...rows.map(row => this.renderRow(row)));
      return group;
    }));
  }

  private renderRow(row: CommandRow): HTMLElement {
    const item = element("article", "terminal-row");
    item.dataset.slot = String(row.slot);
    if (row.state === "running") item.dataset.running = "";
    const summary = element("button", "terminal-summary");
    summary.type = "button";
    summary.setAttribute("aria-expanded", String(this.expanded.has(row.slot)));
    const state = verdict(row);
    const badge = element("span", "terminal-state", state.text);
    badge.dataset.tone = state.tone;
    const time = element("span", "terminal-time", duration(row));
    time.dataset.slot = String(row.slot);
    summary.append(element("code", "terminal-command", row.command), badge, time);
    summary.title = `${row.command}\n${row.cwd}`;
    summary.addEventListener("click", () => this.toggle(row.slot));
    item.append(summary);
    if (this.expanded.has(row.slot)) item.append(this.renderDetail(row));
    return item;
  }

  private renderDetail(row: CommandRow): HTMLElement {
    const detail = element("div", "terminal-detail");
    const meta = element("p", "terminal-meta", `${row.cwd || "（默认目录）"}${row.background ? " · 后台" : ""}`);
    const output = element("pre", "terminal-output", this.expanded.get(row.slot) ?? "正在读取输出…");
    const actions = element("div", "terminal-actions");
    const quote = element("button", "terminal-button", "引用到对话");
    quote.type = "button";
    quote.addEventListener("click", () => this.quote(row, quote));
    actions.append(quote);
    if (row.stoppable) {
      const stop = element("button", "terminal-button", "停止");
      stop.type = "button";
      stop.addEventListener("click", () => void this.stop(row.slot, stop));
      actions.append(stop);
    }
    if (row.output_truncated) actions.append(element("span", "terminal-note", "较早的输出已截断"));
    detail.append(meta, output, actions);
    return detail;
  }

  private toggle(slot: number) {
    if (this.expanded.has(slot)) this.expanded.delete(slot);
    else {
      this.expanded.set(slot, null);
      void this.loadOutput(slot);
    }
    this.render();
  }

  private async loadOutput(slot: number) {
    const session = this.view.session ?? "";
    const response = await this.ctx.http.request(`/api/dashboard/standard-tools/commands/${slot}?session=${encodeURIComponent(session)}`);
    if (session !== this.view.session || !this.expanded.has(slot)) return;
    const text = response.ok ? ((await response.json()) as CommandRow).output || "（没有输出）" : "这条记录已被新的命令覆盖。";
    if (this.expanded.get(slot) === text) return;
    this.expanded.set(slot, text);
    const node = this.list.querySelector(`[data-slot="${slot}"] .terminal-output`);
    if (node) node.textContent = text;
  }

  // 3. 引用带上发起时的会话；宿主和聊天端核对后才写进草稿。
  private quote(row: CommandRow, button: HTMLButtonElement) {
    const output = this.expanded.get(row.slot) ?? "";
    const body = `$ ${row.command}\n${output}`;
    const text = "```\n" + (body.length > QUOTE_LIMIT ? `…${body.slice(-QUOTE_LIMIT)}` : body) + "\n```";
    const status = this.view.quote({ session: this.view.session ?? "", text, source: `终端 · ${row.command.slice(0, 60)}` });
    button.textContent = status === "inserted" ? "已引用" : "会话已切换，未引用";
    window.setTimeout(() => { button.textContent = "引用到对话"; }, 1600);
  }

  private async stop(slot: number, button: HTMLButtonElement) {
    button.disabled = true;
    const session = this.view.session ?? "";
    const response = await this.ctx.http.request(`/api/dashboard/standard-tools/commands/${slot}/stop?session=${encodeURIComponent(session)}`, { method: "POST" });
    const stopped = response.ok && ((await response.json()) as { stopped: boolean }).stopped;
    button.textContent = stopped ? "已停止" : "已经结束";
  }

  // 对话里点某条 shell 调用的“在侧栏打开”：展开并滚到它；刚挂载还没取到列表时先取一次，记录已被覆盖时如实回报。
  async open(intent: ConversationIntent): Promise<"handled" | "invalid"> {
    const payload = intent.payload as { call_id?: unknown } | null;
    const find = () => this.rows.find(item => item.call_id === payload?.call_id);
    if (!find() && this.view.session === intent.session) await this.refresh();
    const row = find();
    if (!row) return "invalid";
    if (!this.expanded.has(row.slot)) this.toggle(row.slot);
    this.list.querySelector<HTMLElement>(`[data-slot="${row.slot}"] .terminal-summary`)?.focus();
    this.list.querySelector(`[data-slot="${row.slot}"]`)?.scrollIntoView({ block: "nearest" });
    return "handled";
  }

  dispose() {
    this.generation++;
    window.clearTimeout(this.timer);
  }
}

function renderTerminal(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onSessionChange !== "function" || typeof view.quote !== "function")
    throw new Error("终端缺少 conversation.tools.v2 view");
  const tab = new TerminalTab(ctx, view);
  host.replaceChildren(tab.root);
  const stops = [view.onSessionChange(() => tab.sync()), view.onActiveChange(() => tab.sync()),
    view.onIntent(intent => tab.open(intent))];
  return () => {
    for (const stop of stops) stop();
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", mount => mount.register({
    id: "terminal", label: "终端", order: 20, accepts: ["tool.shell.open.v1"],
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6l5 5-5 5M12 18h8"/></svg>',
    render(host, _entryView, props) { return renderTerminal(ctx, host, props); },
  }));
}
