import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import { currentTheme, subscribeTheme, type WebHostContextV1, type WebUiDisposer } from "@akashic/web-ui-v1";
import { Pane, terminalTheme, type PaneHost, type TerminalFacts } from "./pane";
import "@xterm/xterm/css/xterm.css";
import "./style.css";

const API = "/api/dashboard/terminal/terminals";

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function button(className: string, label: string, text: string, onClick: () => void): HTMLButtonElement {
  const node = element("button", className, text);
  node.type = "button";
  node.title = label;
  node.setAttribute("aria-label", label);
  node.addEventListener("click", onClick);
  return node;
}

// 终端标签：本会话的终端。终端只在标签可见时连接，会话切换后整组换掉。
class TerminalTab implements PaneHost {
  readonly root = element("div", "terminal-root");
  private readonly stage = element("div", "terminal-stage");
  private readonly notice = element("div", "terminal-notice");
  private panes: Pane[] = [];
  private current: Pane | null = null;
  private loaded: string | null = null;
  private generation = 0;
  private active: boolean;
  session: string;

  constructor(private readonly ctx: WebHostContextV1, view: ConversationTabView) {
    this.active = view.active;
    this.session = view.session ?? "";
    this.notice.setAttribute("role", "status");
    this.stage.append(this.notice);
    this.root.append(this.stage);
  }

  // 1. 可见且有会话时载入本会话的终端；不可见时断开连接，xterm 保留，回来从偏移续接。
  sync(): void {
    if (!this.active) return this.panes.forEach((pane) => pane.disconnect());
    if (!this.session) return this.show("新对话发出第一条消息后就能在这里使用终端。");
    if (this.loaded !== this.session) return void this.load();
    this.panes.forEach((pane) => pane.connect());
    this.fitCurrent();
    this.current?.term.focus();
  }

  setActive(active: boolean): void {
    this.active = active;
    this.sync();
  }

  setSession(session: string | null): void {
    if ((session ?? "") === this.session) return;
    this.session = session ?? "";
    this.clear();
    this.sync();
  }

  private clear(): void {
    this.generation += 1;
    this.loaded = null;
    this.panes.forEach((pane) => pane.dispose());
    this.panes = [];
    this.current = null;
  }

  private async load(): Promise<void> {
    const generation = ++this.generation;
    const session = this.session;
    this.loaded = session;
    try {
      const response = await this.ctx.http.request(`${API}?${new URLSearchParams({ session })}`);
      if (!response.ok) throw new Error(`读取终端失败（${response.status}）`);
      const body = await response.json() as { terminals: TerminalFacts[] };
      if (generation !== this.generation) return;
      this.panes = body.terminals.map((facts) => this.mount(facts));
      if (this.panes.length) this.select(this.panes[this.panes.length - 1]);
      else await this.create();
    } catch (error) {
      if (generation !== this.generation) return;
      this.loaded = null;
      this.show(error instanceof Error ? error.message : String(error), () => this.sync());
    }
  }

  private mount(facts: TerminalFacts): Pane {
    const pane = new Pane(this.ctx, this, this.session, facts, terminalTheme(this.root, currentTheme().colorScheme === "dark"));
    pane.element.hidden = true;
    this.stage.append(pane.element);
    if (this.active) pane.connect();
    return pane;
  }

  // 2. 新开终端：按舞台当前尺寸开 PTY，shell 首屏就按真实列宽排版。
  async create(): Promise<void> {
    if (!this.session) return;
    const generation = this.generation;
    try {
      const facts = await this.open(this.session, this.proposeSize());
      if (generation !== this.generation) return;
      const pane = this.mount(facts);
      this.panes.push(pane);
      this.select(pane);
    } catch (error) {
      if (generation !== this.generation) return;
      this.show(error instanceof Error ? error.message : String(error), () => void this.create());
    }
  }

  // 按尺寸新开一个 PTY；失败时带上后端给的原因。
  private async open(session: string, size: { cols: number; rows: number }): Promise<TerminalFacts> {
    const response = await this.ctx.http.request(`${API}?${new URLSearchParams({ session })}`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(size),
    });
    const body = await response.json() as TerminalFacts & { detail?: string };
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `新开终端失败（${response.status}）`);
    return body;
  }

  private proposeSize(): { cols: number; rows: number } {
    const probe = this.current ?? this.panes[0];
    if (probe) return { cols: probe.term.cols, rows: probe.term.rows };
    const width = this.stage.clientWidth - 20;
    const height = this.stage.clientHeight - 12;
    return { cols: Math.max(20, Math.floor(width / 7.8)), rows: Math.max(5, Math.floor(height / 15.6)) };
  }

  private select(pane: Pane): void {
    this.current = pane;
    for (const item of this.panes) item.element.hidden = item !== pane;
    this.notice.hidden = true;
    this.fitCurrent();
    if (this.active) pane.term.focus();
  }

  fitCurrent(): void {
    if (!this.current || this.current.element.hidden || !this.stage.clientWidth) return;
    try { this.current.fit.fit(); } catch { /* 尺寸为 0 时跳过 */ }
  }

  // 关闭远端 PTY 不等结果：失败时它随会话 owner 清理或换代关闭一并结束。
  private release(session: string, id: number): void {
    const query = new URLSearchParams({ session });
    void this.ctx.http.request(`${API}/${id}?${query}`, { method: "DELETE" }).catch(() => undefined);
  }

  private show(text: string, retry?: () => void): void {
    this.notice.replaceChildren(element("p", "terminal-notice-text", text));
    if (retry) this.notice.append(button("terminal-tool", "重试", "重试", retry));
    else if (this.session) this.notice.append(button("terminal-tool", "新建终端", "新建终端", () => void this.create()));
    this.notice.hidden = false;
  }

  gone(pane: Pane): void {
    pane.dispose();
    this.panes = this.panes.filter((item) => item !== pane);
    if (this.current === pane) this.current = null;
    if (this.panes.length) this.select(this.panes[this.panes.length - 1]);
    else this.show("这个会话没有打开的终端。");
  }

  // 退出后按 Enter：关掉旧 PTY，在原处开一个新的。会话已切换或 Pane 已关时不再接上，新开的随即关掉。
  async restart(pane: Pane): Promise<void> {
    const generation = this.generation;
    this.release(pane.session, pane.facts.id);
    try {
      const facts = await this.open(pane.session, { cols: pane.term.cols, rows: pane.term.rows });
      if (generation !== this.generation || !pane.replace(facts)) this.release(pane.session, facts.id);
    } catch (error) {
      if (!pane.disposed) pane.term.write(`\r\n\x1b[31m${error instanceof Error ? error.message : String(error)}\x1b[0m\r\n`);
    }
  }

  // 3. 快捷键与 kitty 一致：Ctrl+Shift+C/V 复制粘贴，其余组合照常交给终端。
  shortcut(pane: Pane, event: KeyboardEvent): boolean {
    switch (event.code) {
      case "KeyC":
        if (pane.term.hasSelection()) void navigator.clipboard.writeText(pane.term.getSelection()).catch(() => undefined);
        event.preventDefault();
        return true;
      case "KeyV":
        return true;
      default:
        return false;
    }
  }

  retheme(): void {
    const theme = terminalTheme(this.root, currentTheme().colorScheme === "dark");
    for (const pane of this.panes) pane.term.options.theme = theme;
  }

  dispose(): void {
    this.clear();
  }
}

function renderTerminal(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onActiveChange !== "function" || typeof view.onSessionChange !== "function")
    throw new Error("终端缺少 conversation.tools.v2 view");
  const tab = new TerminalTab(ctx, view);
  host.replaceChildren(tab.root);
  let frame = 0;
  const resize = new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => tab.fitCurrent());
  });
  resize.observe(host);
  const stops = [
    view.onSessionChange((session) => tab.setSession(session)),
    view.onActiveChange((active) => tab.setActive(active)),
    subscribeTheme(() => tab.retheme()),
  ];
  // 字体就绪后重新量一次字宽，避免按后备字体排版。
  void document.fonts.load('13px "JetBrains Mono"').then(() => tab.fitCurrent()).catch(() => undefined);
  return () => {
    for (const stop of stops) stop();
    resize.disconnect();
    cancelAnimationFrame(frame);
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "terminal", label: "终端", order: 20,
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M12 15h5"/></svg>',
    render(host, _entryView, props) { return renderTerminal(ctx, host, props); },
  }));
}
