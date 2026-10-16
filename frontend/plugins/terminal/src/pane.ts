import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { Terminal, type ITheme } from "@xterm/xterm";
import { ClipboardAddon } from "@xterm/addon-clipboard";
import { FitAddon } from "@xterm/addon-fit";
import { ImageAddon } from "@xterm/addon-image";
import { SearchAddon } from "@xterm/addon-search";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { WebglAddon } from "@xterm/addon-webgl";

export interface TerminalFacts {
  id: number;
  shell: string;
  cwd: string;
  exited: boolean;
  exit_code: number | null;
}

export interface PaneHost {
  changed?(pane: Pane): void;
  exited?(pane: Pane): void;
  gone(pane: Pane): void;
  restart(pane: Pane): Promise<void>;
  shortcut(pane: Pane, event: KeyboardEvent): boolean;
}

const FONT = '"JetBrains Mono", ui-monospace, "Noto Sans Mono CJK SC", "Sarasa Mono SC", monospace';
// 单帧输入按字符切块，UTF-8 后仍低于执行主机 64 KiB 的单次写入上限。
const INPUT_CHUNK = 16 * 1024;
const RETRY_MS = [300, 1000, 2000, 5000];
// 浅色沿用纸面墨色；深色取 kitty 默认配色，保证常见工具的配色可读。
const LIGHT_ANSI = ["#292923", "#b3261e", "#2e7d32", "#8d6e00", "#1f5fa8", "#8e3b8e", "#00796b", "#9e9b91",
  "#68665e", "#d93025", "#388e3c", "#a37f00", "#2a74c9", "#a64ca6", "#00897b", "#c9c6bb"];
const DARK_ANSI = ["#3a3a36", "#e5534b", "#57c24a", "#d6c25a", "#4c9be8", "#c879d6", "#3fc1c9", "#dddddd",
  "#767676", "#f2706a", "#7fdb6f", "#f0dc7a", "#76b6f5", "#dd9be6", "#6fdde3", "#ffffff"];
const ANSI_NAMES = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"] as const;

// 终端配色跟随宿主主题：底色与前景取纸面 token，16 色按明暗选一套。
export function terminalTheme(host: HTMLElement, dark: boolean): ITheme {
  const style = getComputedStyle(host);
  const token = (name: string, fallback: string): string => style.getPropertyValue(name).trim() || fallback;
  const background = token("--ak-paper-sheet", dark ? "#1c1c1a" : "#fffdf7");
  const foreground = token("--ak-ink-primary", dark ? "#e6e4dc" : "#292923");
  const palette = dark ? DARK_ANSI : LIGHT_ANSI;
  const theme: Record<string, string> = {
    background, foreground, cursor: foreground, cursorAccent: background,
    selectionBackground: dark ? "rgba(255, 255, 255, 0.24)" : "rgba(41, 41, 35, 0.16)",
  };
  ANSI_NAMES.forEach((name, index) => {
    theme[name] = palette[index];
    theme[`bright${name[0].toUpperCase()}${name.slice(1)}`] = palette[index + 8];
  });
  return theme as ITheme;
}

// 一个终端：xterm 实例加一条到执行主机 PTY 的 WebSocket；输出按偏移续接，断线后从记下的偏移重连。
export class Pane {
  readonly element = document.createElement("div");
  readonly term: Terminal;
  readonly fit = new FitAddon();
  readonly search = new SearchAddon();
  title = "";
  private socket: WebSocket | null = null;
  private offset = 0;
  private attempt = 0;
  private timer = 0;
  private wanted = false;
  // 已显示过退出提示；之后不再连接，重新开始时清除。
  private noticed = false;
  // 每次连上后的第一帧是积压输出的回放；其中旧的查询（CSI 14t、DA1 等）xterm 会再应答一遍，
  // 解析这一帧期间的应答不能当输入送进 shell。
  private caughtUp = false;
  private replaying = false;

  disposed = false;
  closing = false;
  private restarting = false;

  // session 固定为创建时的会话：会话切换后迟到的回调也不会连到别的会话。
  constructor(
    private readonly ctx: WebHostContextV1, private readonly host: PaneHost,
    readonly session: string, public facts: TerminalFacts, theme: ITheme,
  ) {
    this.element.className = "terminal-pane";
    this.term = new Terminal({
      fontFamily: FONT, fontSize: 13, lineHeight: 1.2, cursorBlink: true, cursorStyle: "block",
      cursorInactiveStyle: "outline", scrollback: 10_000, allowProposedApi: true, macOptionIsMeta: true,
      drawBoldTextInBrightColors: false, theme,
    });
    // 图片：sixel 与 iTerm2 内联图（img2sixel、chafa、neofetch --sixel），并回报像素尺寸供程序缩放；
    // kitty 图形协议要等 xterm.js 6.1 正式版。
    for (const addon of [this.fit, this.search, new Unicode11Addon(), new ClipboardAddon(),
      new WebLinksAddon((_event, uri) => window.open(uri, "_blank", "noopener")),
      new ImageAddon({ storageLimit: 32 })]) this.term.loadAddon(addon);
    this.term.unicode.activeVersion = "11";
    this.term.open(this.element);
    this.useWebgl();
    this.term.attachCustomKeyEventHandler((event) => this.key(event));
    this.term.onData((data) => this.input(data));
    this.term.onResize(({ cols, rows }) => this.send({ type: "resize", cols, rows }));
    this.term.onTitleChange((title) => { this.title = title; this.host.changed?.(this); });
  }

  // WebGL 渲染更接近原生终端的刷新速度；上下文丢失或不支持时退回 DOM 渲染。
  private useWebgl(): void {
    try {
      const webgl = new WebglAddon();
      webgl.onContextLoss(() => webgl.dispose());
      this.term.loadAddon(webgl);
    } catch {
      /* DOM 渲染照常可用 */
    }
  }

  get label(): string {
    const folder = this.facts.cwd.split("/").filter(Boolean).pop() ?? "~";
    return this.title || `${this.facts.shell.split("/").pop() ?? "shell"} · ${folder}`;
  }

  // Esc 留给终端里的程序（vim 等），不让工作台拿去还原或收起；Ctrl+Shift 组合交给标签处理。
  private key(event: KeyboardEvent): boolean {
    if (event.type !== "keydown") return true;
    if (event.key === "Escape") event.preventDefault();
    if (event.ctrlKey && event.shiftKey && !event.altKey) return !this.host.shortcut(this, event);
    return true;
  }

  private input(data: string): void {
    if (this.replaying) return;
    if (this.facts.exited) {
      if (!data.includes("\r") || this.restarting) return;
      this.restarting = true;
      void this.host.restart(this).finally(() => { this.restarting = false; });
      return;
    }
    // 切点落在代理对中间时后移一位，不发出孤立代理项。
    for (let index = 0; index < data.length;) {
      let end = Math.min(index + INPUT_CHUNK, data.length);
      if (end < data.length && /[\uD800-\uDBFF]/.test(data[end - 1])) end += 1;
      this.send({ type: "input", data: data.slice(index, end) });
      index = end;
    }
  }

  private send(message: object): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message));
  }

  // 换成新开的 PTY（重新开始）：清屏，从偏移 0 接上；已销毁时拒绝，收起期间只换不连。
  replace(facts: TerminalFacts): boolean {
    if (this.disposed) return false;
    const visible = this.wanted;
    this.disconnect();
    this.facts = facts;
    this.title = "";
    this.offset = 0;
    this.noticed = false;
    this.term.reset();
    if (visible) this.connect();
    return true;
  }

  connect(): void {
    this.wanted = true;
    if (this.socket || this.noticed) return;
    const query = new URLSearchParams({ session: this.session, terminal: String(this.facts.id), offset: String(this.offset) });
    const socket = new WebSocket(this.ctx.http.webSocketUrl(`/api/dashboard/terminal/attach?${query}`));
    this.caughtUp = false;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      const recovered = this.attempt > 0;
      this.attempt = 0;
      this.send({ type: "resize", cols: this.term.cols, rows: this.term.rows });
      // 重连成功立刻撤下“重新连接中”，不等下一次输出或标题变化。
      if (recovered) this.host.changed?.(this);
    };
    socket.onmessage = (event) => this.receive(event.data);
    socket.onclose = (event) => this.closed(socket, event.code);
    this.socket = socket;
  }

  private receive(data: unknown): void {
    if (data instanceof ArrayBuffer) {
      this.offset += data.byteLength;
      if (this.caughtUp) return this.term.write(new Uint8Array(data));
      this.caughtUp = true;
      this.replaying = true;
      this.term.write(new Uint8Array(data), () => { this.replaying = false; });
      return;
    }
    const message = JSON.parse(String(data)) as { type: string; offset?: number; code?: number | null };
    if (message.type === "reset" && typeof message.offset === "number") {
      this.term.reset();
      this.offset = message.offset;
    } else if (message.type === "exit") {
      this.facts = { ...this.facts, exited: true, exit_code: message.code ?? null };
      if (this.noticed) return;
      this.noticed = true;
      const code = message.code === null || message.code === undefined ? "未知" : String(message.code);
      this.term.write(`\r\n\x1b[2m[进程已退出，代码 ${code}。按 Enter 重新开始]\x1b[0m\r\n`);
      this.host.exited?.(this);
    }
  }

  // 意外断开按退避重连；4404 表示终端已不存在，交给标签移除。
  private closed(socket: WebSocket, code: number): void {
    if (this.socket !== socket) return;
    this.socket = null;
    if (code === 4404) return this.host.gone(this);
    if (!this.wanted || this.facts.exited) return;
    const delay = RETRY_MS[Math.min(this.attempt, RETRY_MS.length - 1)];
    this.attempt += 1;
    this.host.changed?.(this);
    this.timer = window.setTimeout(() => this.connect(), delay);
  }

  get reconnecting(): boolean {
    return this.wanted && !this.facts.exited && this.attempt > 0 && this.socket?.readyState !== WebSocket.OPEN;
  }

  disconnect(): void {
    this.wanted = false;
    window.clearTimeout(this.timer);
    const socket = this.socket;
    this.socket = null;
    socket?.close(1000);
  }

  dispose(): void {
    this.disposed = true;
    this.disconnect();
    this.term.dispose();
    this.element.remove();
  }
}
