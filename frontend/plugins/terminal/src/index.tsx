import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import { currentTheme, subscribeTheme, type WebHostContextV1, type WebUiDisposer } from "@akashic/web-ui-v1";
import { Pane, terminalTheme, type PaneHost, type TerminalFacts } from "./pane";
import "@xterm/xterm/css/xterm.css";
import "./style.css";

const API = "/api/dashboard/terminal/terminals";
const CLAIMS_KEY = "akashic.terminal.claims.";
const STATE_DETAILS = { exited: "已退出", reconnecting: "重新连接中" } as const;

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

const reason = (error: unknown): string => (error instanceof Error ? error.message : String(error));

// 实例 → 终端的对应按会话记在模块内（主数据）；sessionStorage 只用于页面刷新后恢复，读写失败不影响本页内的判断。
const claimStore = new Map<string, Record<string, number>>();

function restoreClaims(session: string): Record<string, number> {
  try {
    const value: unknown = JSON.parse(window.sessionStorage.getItem(CLAIMS_KEY + session) ?? "{}");
    if (!value || typeof value !== "object") return {};
    return Object.fromEntries(Object.entries(value).filter((pair): pair is [string, number] => Number.isInteger(pair[1])));
  } catch {
    return {};
  }
}

function readClaims(session: string): Record<string, number> {
  const claims = claimStore.get(session) ?? restoreClaims(session);
  claimStore.set(session, claims);
  return { ...claims };
}

function writeClaims(session: string, claims: Record<string, number>): void {
  claimStore.set(session, { ...claims });
  try {
    window.sessionStorage.setItem(CLAIMS_KEY + session, JSON.stringify(claims));
  } catch {
    /* 只影响刷新后的恢复：没记下的终端刷新后由新实例接管 */
  }
}

// 剪掉已关标签（不在实例快照里）和已不存在的终端的认领；已关标签留下的终端、没人认领又已退出的终端交给调用方关闭。
function settleClaims(stored: Record<string, number>, siblings: ReadonlySet<string>, terminals: readonly TerminalFacts[]):
  { claims: Record<string, number>; orphans: number[] } {
  const live = new Set(terminals.map((item) => item.id));
  const claims = Object.fromEntries(Object.entries(stored).filter(([instance, id]) => siblings.has(instance) && live.has(id)));
  const kept = new Set(Object.values(claims));
  const stale = Object.entries(stored).filter(([instance, id]) => !siblings.has(instance) && live.has(id)).map(([, id]) => id);
  const exited = terminals.filter((item) => item.exited).map((item) => item.id);
  return { claims, orphans: [...new Set([...stale, ...exited])].filter((id) => !kept.has(id)) };
}

// 同一会话的认领、重开与删除认领串行：布局恢复时几个实例同时渲染，也不会认领到同一个终端或互相覆盖记录。
const claimQueues = new Map<string, Promise<unknown>>();
function serial<T>(session: string, task: () => Promise<T>): Promise<T> {
  const next = (claimQueues.get(session) ?? Promise.resolve()).catch(() => undefined).then(task);
  claimQueues.set(session, next);
  return next;
}

// 正在关闭的终端（会话 → 终端 → 请求）：关完之前不再被认领；需要新开时先等它们关完，腾出每会话的名额。
const releasing = new Map<string, Map<number, Promise<void>>>();
const pause = (ms: number): Promise<void> => new Promise((resolve) => window.setTimeout(resolve, ms));

// 终端标签的一个实例：对应一个 PTY，属于创建它的会话。只在可见时连接；用户关掉标签时关闭 PTY，换会话卸下时 PTY 留着。
class TerminalInstance implements PaneHost {
  readonly root = element("div", "terminal-root");
  private readonly stage = element("div", "terminal-stage");
  private readonly notice = element("div", "terminal-notice");
  private readonly finder = element("div", "terminal-search");
  private readonly query = element("input", "terminal-search-input");
  private readonly session: string;
  private readonly instance: string;
  private pane: Pane | null = null;
  private active: boolean;
  private loading = false;
  private disposed = false;
  private closed = false;

  constructor(private readonly ctx: WebHostContextV1, private readonly view: ConversationTabView) {
    this.active = view.active;
    this.session = view.session ?? "";
    this.instance = view.instance ?? "";
    this.notice.setAttribute("role", "status");
    this.query.type = "search";
    this.query.placeholder = "搜索终端输出";
    this.query.setAttribute("aria-label", "搜索终端输出");
    this.query.addEventListener("keydown", (event) => this.searchKey(event));
    this.finder.append(this.query, button("terminal-tool", "上一个", "↑", () => this.find(true)),
      button("terminal-tool", "下一个", "↓", () => this.find(false)), button("terminal-tool", "关闭搜索", "✕", () => this.closeSearch()));
    this.finder.hidden = true;
    this.stage.append(this.notice);
    this.root.append(this.finder, this.stage);
  }

  // 1. 可见时连上（第一次可见时先认领终端）；不可见时断开连接，xterm 保留，回来从偏移续接。
  sync(): void {
    if (!this.active) return void this.pane?.disconnect();
    if (!this.session || !this.instance) return this.show("新对话发出第一条消息后就能在这里使用终端。");
    if (!this.pane) return void this.attach();
    this.pane.connect();
    this.fit();
    this.pane.term.focus();
  }

  setActive(active: boolean): void {
    this.active = active;
    this.sync();
  }

  // 2. 认领：先找自己记下的终端，其次是本会话里没有被其他实例认领的、还在运行的终端，都没有就新开一个。
  private async attach(): Promise<void> {
    if (this.loading) return;
    this.loading = true;
    // 实例列表在任何 await 之前取快照：getter 跟着宿主当前的会话走，换会话后读到的是别的会话的标签。
    const siblings = new Set(this.view.instances);
    try {
      const facts = await serial(this.session, () => this.claim(siblings));
      if (this.disposed) {
        if (this.closed && facts) this.forget(facts.id);
        return;
      }
      if (facts) this.mount(facts);
    } catch (error) {
      if (!this.disposed) this.show(reason(error), () => this.sync());
    } finally {
      this.loading = false;
    }
  }

  // 中途被卸下（换会话或关标签）就不写记录、不新开，返回 null。
  private async claim(siblings: ReadonlySet<string>): Promise<TerminalFacts | null> {
    if (this.disposed) return null;
    if (!siblings.has(this.instance)) throw new Error("工作台标签还没就绪，请重试。");
    const terminals = await this.list();
    if (this.disposed) return null;
    // 1. 已关标签留下的终端、没人认领又已退出的终端直接关闭，不再留给别的实例接管。
    const { claims, orphans } = settleClaims(readClaims(this.session), siblings, terminals);
    for (const id of orphans) void this.release(id);
    const closing = this.pendingReleases();
    // 2. 自己记下的 → 没人认领、还在运行的 → 新开。
    const taken = new Set(Object.values(claims));
    const facts = terminals.find((item) => item.id === claims[this.instance])
      ?? terminals.find((item) => !taken.has(item.id) && !item.exited && !closing.has(item.id))
      ?? await this.openAfter([...closing.values()]);
    if (!facts) return null;
    // 3. 新开期间标签被关掉：新终端随即关闭；只是换会话则照常记下，回来接着用。
    if (this.closed) {
      void this.release(facts.id);
      return null;
    }
    writeClaims(this.session, { ...claims, [this.instance]: facts.id });
    return facts;
  }

  private async list(): Promise<TerminalFacts[]> {
    const response = await this.ctx.http.request(`${API}?${new URLSearchParams({ session: this.session })}`);
    if (!response.ok) throw new Error(`读取终端失败（${response.status}）`);
    return (await response.json() as { terminals: TerminalFacts[] }).terminals;
  }

  // 等正在关闭的终端关完（腾出名额）再新开；等待期间被卸下就不开。
  private async openAfter(closing: readonly Promise<void>[]): Promise<TerminalFacts | null> {
    await Promise.all(closing);
    return this.disposed ? null : this.open(this.proposeSize());
  }

  private mount(facts: TerminalFacts): void {
    this.pane = new Pane(this.ctx, this, this.session, facts, terminalTheme(this.root, currentTheme().colorScheme === "dark"));
    this.notice.hidden = true;
    this.stage.append(this.pane.element);
    this.changed(this.pane);
    this.sync();
  }

  // 按尺寸新开一个 PTY；名额已满（409）时稍等重试两次，刚关闭的终端可能还没从后端登记表移除；失败时带上后端给的原因。
  private async open(size: { cols: number; rows: number }): Promise<TerminalFacts> {
    for (let attempt = 0; ; attempt += 1) {
      const response = await this.ctx.http.request(`${API}?${new URLSearchParams({ session: this.session })}`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(size),
      });
      const body = await response.json() as TerminalFacts & { detail?: string };
      if (response.ok) return body;
      if (response.status !== 409 || attempt >= 2)
        throw new Error(typeof body.detail === "string" ? body.detail : `新开终端失败（${response.status}）`);
      await pause(1000);
    }
  }

  private proposeSize(): { cols: number; rows: number } {
    const width = this.stage.clientWidth - 20;
    const height = this.stage.clientHeight - 12;
    return { cols: Math.max(20, Math.floor(width / 7.8)), rows: Math.max(5, Math.floor(height / 15.6)) };
  }

  fit(): void {
    if (!this.pane || !this.stage.clientWidth) return;
    try { this.pane.fit.fit(); } catch { /* 尺寸为 0 时跳过 */ }
  }

  // 3. 标签标题跟 shell 标题（或 shell · 目录）；退出、重连写进状态，只出现在提示与新标签页里。
  changed(pane: Pane): void {
    const state = pane.facts.exited ? "exited" : pane.reconnecting ? "reconnecting" : null;
    this.view.setTitle(pane.label);
    this.view.setIndicator(state ? { detail: STATE_DETAILS[state] } : null);
  }

  exited(pane: Pane): void {
    this.changed(pane);
  }

  // 终端在别处被关掉（4404）：撤下 xterm，留一个重新开始的入口。
  gone(pane: Pane): void {
    pane.dispose();
    if (this.pane === pane) this.pane = null;
    this.forget(null);
    this.view.setIndicator({ detail: "已关闭" });
    this.show("这个终端已经关闭。", () => this.sync(), "新开终端");
  }

  // 退出后按 Enter：等旧 PTY 关完（腾出名额）再在原处开一个新的，与认领串行；标签已关或已换掉时，新开的随即关掉。
  async restart(pane: Pane): Promise<void> {
    try {
      await serial(this.session, async () => {
        await this.release(pane.facts.id);
        const facts = await this.open({ cols: pane.term.cols, rows: pane.term.rows });
        if (this.disposed || pane !== this.pane || !pane.replace(facts)) return void this.release(facts.id);
        writeClaims(this.session, { ...readClaims(this.session), [this.instance]: facts.id });
        this.changed(pane);
      });
    } catch (error) {
      if (!pane.disposed) pane.term.write(`\r\n\x1b[31m${reason(error)}\x1b[0m\r\n`);
    }
  }

  // 用户关掉了标签：关闭 PTY 并删掉认领；还在认领中的，认领完成后再关。
  close(): void {
    this.closed = true;
    if (this.pane) this.forget(this.pane.facts.id);
  }

  // 删掉本实例的认领并关闭它的 PTY；与认领串行，不会被正在进行的认领写回。
  private forget(id: number | null): void {
    void serial(this.session, async () => {
      const claims = readClaims(this.session);
      delete claims[this.instance];
      writeClaims(this.session, claims);
      if (id !== null) void this.release(id);
    });
  }

  // 关闭远端 PTY，同一终端只发一次；失败时它已不在认领里，由下一个新开的实例接管或关闭，或随会话 owner 清理。
  private release(id: number): Promise<void> {
    const pending = this.pendingReleases();
    const known = pending.get(id);
    if (known) return known;
    const query = new URLSearchParams({ session: this.session });
    const request = this.ctx.http.request(`${API}/${id}?${query}`, { method: "DELETE" })
      .then(() => undefined, () => undefined)
      .finally(() => pending.delete(id));
    pending.set(id, request);
    return request;
  }

  private pendingReleases(): Map<number, Promise<void>> {
    const pending = releasing.get(this.session) ?? new Map<number, Promise<void>>();
    releasing.set(this.session, pending);
    return pending;
  }

  private show(text: string, retry?: () => void, label = "重试"): void {
    this.notice.replaceChildren(element("p", "terminal-notice-text", text));
    if (retry) this.notice.append(button("terminal-tool", label, label, retry));
    this.notice.hidden = false;
  }

  // 4. 快捷键与 kitty 一致：Ctrl+Shift+C/V 复制粘贴，T 新开终端，F 搜索，←/→ 切到相邻标签。
  //    普通浏览器标签页里 Ctrl+Shift+T 被浏览器保留（恢复关闭的标签页），只在 PWA / 桌面壳里到得了页面。
  shortcut(pane: Pane, event: KeyboardEvent): boolean {
    switch (event.code) {
      case "KeyC":
        if (pane.term.hasSelection()) void navigator.clipboard.writeText(pane.term.getSelection()).catch(() => undefined);
        event.preventDefault();
        return true;
      case "KeyV":
        return true;
      case "KeyT":
        event.preventDefault();
        this.view.openNew();
        return true;
      case "KeyF":
        event.preventDefault();
        this.openSearch();
        return true;
      case "ArrowLeft":
      case "ArrowRight":
        return this.view.cycle(event.code === "ArrowLeft" ? -1 : 1);
      default:
        return false;
    }
  }

  private openSearch(): void {
    this.finder.hidden = false;
    this.query.value = this.pane?.term.getSelection() || this.query.value;
    this.query.focus();
    this.query.select();
  }

  private closeSearch(): void {
    this.finder.hidden = true;
    this.pane?.search.clearDecorations();
    this.pane?.term.focus();
  }

  private searchKey(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      event.preventDefault();
      this.closeSearch();
    } else if (event.key === "Enter") {
      event.preventDefault();
      this.find(event.shiftKey);
    }
  }

  private find(backward: boolean): void {
    const term = this.query.value;
    if (!term || !this.pane) return;
    // xterm 的装饰色只认不透明 #RRGGBB；按明暗各取一组，命中处保持前景可读。
    const dark = currentTheme().colorScheme === "dark";
    const options = { decorations: { matchOverviewRuler: "#a37f00", activeMatchColorOverviewRuler: "#d93025",
      matchBackground: dark ? "#5c5220" : "#f3e3a0", activeMatchBackground: dark ? "#8a4a1e" : "#f0b48a" } };
    if (backward) this.pane.search.findPrevious(term, options);
    else this.pane.search.findNext(term, options);
  }

  retheme(): void {
    if (this.pane) this.pane.term.options.theme = terminalTheme(this.root, currentTheme().colorScheme === "dark");
  }

  dispose(): void {
    this.disposed = true;
    this.pane?.dispose();
    this.pane = null;
  }
}

function renderTerminal(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onActiveChange !== "function" || typeof view.onClose !== "function")
    throw new Error("终端缺少 conversation.tools.v2 view");
  const tab = new TerminalInstance(ctx, view);
  host.replaceChildren(tab.root);
  let frame = 0;
  const resize = new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => tab.fit());
  });
  resize.observe(host);
  const stops = [
    view.onActiveChange((active) => tab.setActive(active)),
    view.onClose(() => tab.close()),
    subscribeTheme(() => tab.retheme()),
  ];
  // 字体就绪后重新量一次字宽，避免按后备字体排版。
  void document.fonts.load('13px "JetBrains Mono"').then(() => tab.fit()).catch(() => undefined);
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
    id: "terminal", label: "终端", order: 20, hint: "在会话的工作目录里", shortcut: "Ctrl+`", multiple: true,
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M12 15h5"/></svg>',
    render(host, _entryView, props) { return renderTerminal(ctx, host, props); },
  }));
}
