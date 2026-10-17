import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { AgentCursor } from "./agent-cursor";
import type { ComputerEvents } from "./computer-events";

interface StreamWindow extends Window {
  computerInputHandlers?: InputHandlers & { touch(event: Event): void };
  fps?: number;
  selkiesTransport?: StreamTransport;
  webrtcInput?: {
    readonly element: HTMLElement;
    attach_context(): void;
    detach_context(): void;
    _sendKeyEvent(keysym: number, code: string, down: boolean): void;
  };
}

interface StreamTransport {
  readonly readyState: number;
  close(): void;
  send(data: string): void;
  addEventListener(type: "close" | "message", listener: (event: { reason?: string; data?: unknown }) => void): void;
  removeEventListener(type: "message", listener: (event: { data?: unknown }) => void): void;
}

interface InputHandlers {
  keydown(event: KeyboardEvent): void;
  keyup(event: KeyboardEvent): void;
  paste(event: ClipboardEvent): void;
  blur(): void;
}

/** 在独立 iframe 中运行固定版本的流客户端，避免全局变量和输入互相干扰。 */
export class ComputerDisplay extends EventTarget {
  private readonly frame = document.createElement("iframe");
  private readonly abort = new AbortController();
  private readonly urls: string[] = [];
  private timer = 0;
  private startupTimer = 0;
  private connected = false;
  private closed = false;
  private inputReady = false;
  // 接管时始终可输入；没接管时由面板按 Agent 是否空闲开关，不需要重连。
  private inputEnabled: boolean;
  private transportWatched = false;
  private clipboardId = 0;
  private cursor: AgentCursor | null = null;
  private governor: LinkGovernor | null = null;
  private readonly clipboardRequests = new Map<number, {
    resolve(): void;
    reject(error: Error): void;
    timer: number;
  }>();

  constructor(
    private readonly host: HTMLElement,
    private readonly ctx: WebHostContextV1,
    private readonly handlers: InputHandlers,
    private readonly events: ComputerEvents,
    private readonly control = "",
    private readonly target = "",
  ) {
    super();
    this.frame.className = "computer-stream";
    this.frame.title = "浏览器画面";
    this.inputEnabled = !this.control;
    this.setInput(Boolean(this.control));
    this.frame.allow = "clipboard-read; clipboard-write; fullscreen";
    this.frame.addEventListener("load", () => this.watch());
    // 获取脚本、iframe 启动和首帧共用一个期限，避免卡在 load 事件之前。
    this.startupTimer = window.setTimeout(() => {
      const transport = this.stream?.selkiesTransport;
      const reason = !transport
        ? "显示客户端启动超时，请刷新页面或检查浏览器是否阻止了脚本"
        : transport.readyState !== WebSocket.OPEN
          ? "远程桌面连接超时，请检查网络后重试"
          : "远程桌面首帧超时，请重新连接";
      this.fail(reason);
    }, 15_000);
    void this.load().catch((error: unknown) => {
      if (this.closed) return;
      this.fail(String(error));
    });
  }

  private fail(reason: string) {
    if (this.closed) return;
    this.dispatchEvent(new CustomEvent("error", { detail: { reason } }));
    this.disconnect();
  }

  private get stream(): StreamWindow | null {
    return this.frame.contentWindow as StreamWindow | null;
  }

  /** 只从当前 generation 读取客户端，再交入同一身份的完整 WebSocket 地址。 */
  private async load() {
    if (!window.isSecureContext) {
      throw new Error("Computer 需要安全连接。请通过 HTTPS 地址打开 Akashic；局域网 HTTP 地址无法显示桌面。");
    }
    // 1. 资源与 WebSocket 必须来自同一 activation。
    const response = await this.ctx.http.request("/api/dashboard/computer/stream-client", {
      signal: this.abort.signal,
      cache: "no-store",
    });
    if (response.headers.get("X-Akashic-Web-Stale") === "1"
      || response.headers.get("X-Akashic-Web-Rebound") === "1") {
      this.dispatchEvent(new Event("stale"));
      return;
    }
    if (!response.ok) throw new Error(`Computer 显示客户端不可用（${response.status}）`);
    const source = await response.text();
    if (this.closed) return;
    const socketUrl = new URL(this.ctx.http.webSocketUrl("/api/dashboard/computer/stream"));
    if (this.control) socketUrl.searchParams.set("control", this.control);
    if (this.control && this.target) socketUrl.searchParams.set("target", this.target);
    const socket = socketUrl.href;
    // srcdoc 的 location.origin 是 "null"；客户端消息使用真实的窗口 origin。
    const moduleUrl = this.blob(source.replaceAll("window.location.origin", "window.origin"), "text/javascript");
    const settingsUrl = `${location.origin}/api/dashboard/computer/stream`;
    // 2. 键盘事件先交给面板，避免上游剪贴板 hold 重复消费同一次粘贴。
    const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Computer</title></head><body><div id="app"></div><script>
      window.__SELKIES_STREAMING_MODE__ = "websockets";
      window.__SELKIES_WEBSOCKET_URL__ = ${JSON.stringify(socket)};
      window.__SELKIES_STORAGE_URL__ = ${JSON.stringify(settingsUrl)};
      for (const name of ["keydown", "keyup", "paste"]) {
        window.addEventListener(name, event => {
          const handlers = window.computerInputHandlers;
          if (!handlers) return;
          handlers.touch(event);
          handlers[name](event);
          event.stopImmediatePropagation();
        }, true);
      }
      </script><script type="module" src="${moduleUrl}"></script></body></html>`;
    // 3. srcdoc 保持同源，也支持阻止 Blob 页面导航的内置浏览器。
    this.frame.srcdoc = html;
    window.addEventListener("message", this.onMessage);
    this.host.replaceChildren(this.frame);
  }

  private blob(source: string, type: string): string {
    const url = URL.createObjectURL(new Blob([source], { type }));
    this.urls.push(url);
    return url;
  }

  /** 同源且来自当前 iframe 的剪贴板消息才可更新本机界面。 */
  private readonly onMessage = (event: MessageEvent) => {
    if (event.source !== this.stream || event.origin !== location.origin) return;
    const value: unknown = event.data;
    if (!value || typeof value !== "object") return;
    const message = value as { type?: unknown; text?: unknown; id?: unknown; error?: unknown };
    if (message.type === "computerClipboardSent" && typeof message.id === "number"
      && typeof message.error === "string") {
      const request = this.clipboardRequests.get(message.id);
      if (!request) return;
      window.clearTimeout(request.timer);
      this.clipboardRequests.delete(message.id);
      if (message.error) request.reject(new Error(message.error));
      else request.resolve();
    }
    if (message.type === "clipboardContentUpdate" && typeof message.text === "string") {
      this.dispatchEvent(new CustomEvent("clipboard", { detail: { text: message.text } }));
    }
  };

  // 人工输入时立即收起 Agent 光标。
  private readonly touch = (event: Event) => {
    if (event.isTrusted) this.cursor?.hide();
  };

  private viewport(): DOMRect | null {
    for (const element of this.frame.contentDocument?.querySelectorAll("video, #videoCanvas, #videoWorkerCanvas") ?? []) {
      const rect = element.getBoundingClientRect();
      if (getComputedStyle(element).display !== "none" && rect.width > 0 && rect.height > 0) {
        const frame = this.frame.getBoundingClientRect();
        return new DOMRect(frame.x + rect.x, frame.y + rect.y, rect.width, rect.height);
      }
    }
    return null;
  }

  /** 首帧和输入一起就绪才报告连接；断线交给面板的唯一重连 owner。 */
  private watch() {
    if (this.closed || this.timer) return;
    this.timer = window.setInterval(() => {
      const stream = this.stream;
      const transport = stream?.selkiesTransport;
      // 1. 接管转移与普通断线采用不同重连语义。
      if (transport && !this.transportWatched) {
        this.transportWatched = true;
        transport.addEventListener("message", (event) => {
          // 固定版本用 KILL 文本通知接管，Core relay 不转发关闭原因。
          if (!this.closed && event.data === "KILL a new primary client connected connection killed") {
            this.dispatchEvent(new Event("superseded"));
            this.disconnect();
          }
        });
        transport.addEventListener("close", (event) => {
          if (this.closed) return;
          if (/superseded/i.test(event.reason ?? "")) {
            this.dispatchEvent(new Event("superseded"));
            this.disconnect();
          } else if (!this.connected) {
            this.fail("远程桌面在显示首帧前断开，请重新连接");
          } else {
            this.disconnect();
          }
        });
      }
      const input = stream?.webrtcInput;
      const document = this.frame.contentDocument;
      if (input && document && !this.inputReady) {
        this.inputReady = true;
        stream!.computerInputHandlers = { ...this.handlers, touch: this.touch };
        for (const name of ["pointerdown", "pointermove", "pointerup", "wheel"]) {
          document.addEventListener(name, this.touch, true);
        }
        stream?.addEventListener("blur", this.handlers.blur);
      }
      // 2. 就绪依赖已显示画面，而非空白 canvas 的默认尺寸。
      const video = document?.querySelector("video");
      const canvas = document?.querySelector<HTMLCanvasElement>("#videoCanvas");
      const hasFrame = (video && video.videoWidth > 0 && video.readyState >= 2
        && getComputedStyle(video).display !== "none")
        || (canvas && getComputedStyle(canvas).display !== "none" && (stream?.fps ?? 0) > 0);
      if (!this.connected && transport?.readyState === WebSocket.OPEN && input && hasFrame) {
        this.connected = true;
        window.clearTimeout(this.startupTimer);
        this.cursor = new AgentCursor(this.host, this.events, () => this.viewport());
        this.governor = new LinkGovernor(transport, this.frame);
        this.dispatchEvent(new Event("connect"));
      }
      if (transport?.readyState === WebSocket.CLOSED) {
        if (this.connected) this.disconnect();
        else this.fail("远程桌面在显示首帧前断开，请重新连接");
      }
    }, 100);
  }

  focus(options?: FocusOptions) {
    const input = this.stream?.webrtcInput;
    input?.attach_context();
    input?.element.focus(options);
  }

  blur() {
    this.cursor?.hide();
    this.stream?.webrtcInput?.detach_context();
  }

  setInput(enabled: boolean) {
    if (enabled === this.inputEnabled) return;
    this.inputEnabled = enabled;
    this.frame.tabIndex = enabled ? 0 : -1;
    this.frame.style.pointerEvents = enabled ? "auto" : "none";
  }

  sendKey(keysym: number, code: string, down: boolean) {
    if (this.inputEnabled) this.stream?.webrtcInput?._sendKeyEvent(keysym, code, down);
  }

  // 从流客户端正在显示的 video 或 canvas 取当前帧；两者都在同源 srcdoc 里，可以直接绘制。
  // 元素属于 srcdoc 的 realm，父窗口的 instanceof 不成立，按标签名区分；video 用自然尺寸。
  snapshot(): Promise<Blob> {
    const document = this.frame.contentDocument;
    const size = (node: HTMLVideoElement | HTMLCanvasElement) => node.tagName === "VIDEO"
      ? { width: (node as HTMLVideoElement).videoWidth, height: (node as HTMLVideoElement).videoHeight }
      : { width: node.width, height: node.height };
    const source = [...document?.querySelectorAll<HTMLVideoElement | HTMLCanvasElement>("video, #videoCanvas, #videoWorkerCanvas") ?? []]
      .find(node => getComputedStyle(node).display !== "none" && size(node).width > 0
        && (node.tagName !== "VIDEO" || (node as HTMLVideoElement).readyState >= HTMLMediaElement.HAVE_CURRENT_DATA));
    if (!source) return Promise.reject(new Error("还没有画面"));
    const canvas = document!.createElement("canvas");
    ({ width: canvas.width, height: canvas.height } = size(source));
    canvas.getContext("2d")!.drawImage(source, 0, 0);
    return new Promise((resolve, reject) =>
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("浏览器没有生成图片")), "image/png"));
  }

  /** 回执到达后才允许发送粘贴按键，断线或超时明确失败。 */
  clipboardPasteFrom(text: string): Promise<void> {
    if (!this.inputEnabled) return Promise.reject(new Error("Agent 正在使用浏览器，先接管再粘贴"));
    if (!this.connected || this.closed) return Promise.reject(new Error("Computer 尚未连接"));
    const requestId = ++this.clipboardId;
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.clipboardRequests.delete(requestId);
        reject(new Error("Computer 剪贴板发送超时"));
      }, 5_000);
      this.clipboardRequests.set(requestId, { resolve, reject, timer });
      this.stream!.postMessage({ type: "clipboardUpdateFromUI", text, requestId }, location.origin);
    });
  }

  sendCtrlAltDel() {
    for (const [keysym, code] of [[0xffe3, "ControlLeft"], [0xffe9, "AltLeft"], [0xffff, "Delete"]] as const) {
      this.sendKey(keysym, code, true);
    }
    for (const [keysym, code] of [[0xffff, "Delete"], [0xffe9, "AltLeft"], [0xffe3, "ControlLeft"]] as const) {
      this.sendKey(keysym, code, false);
    }
  }

  /** 先释放远端输入，再关闭 transport、iframe、计时器和临时 Blob。 */
  disconnect() {
    if (this.closed) return;
    this.closed = true;
    this.abort.abort();
    for (const request of this.clipboardRequests.values()) {
      window.clearTimeout(request.timer);
      request.reject(new Error("Computer 连接已关闭"));
    }
    this.clipboardRequests.clear();
    window.clearInterval(this.timer);
    window.clearTimeout(this.startupTimer);
    window.removeEventListener("message", this.onMessage);
    this.cursor?.destroy();
    this.governor?.stop();
    this.blur();
    this.stream?.selkiesTransport?.close();
    this.frame.remove();
    for (const url of this.urls) URL.revokeObjectURL(url);
    this.dispatchEvent(new Event("disconnect"));
  }
}

// 远程链路上，视频帧一旦塞满带宽，输入和回声都排在它们后面，延迟会累积到几秒。
// 按回声往返相对近期最低值的膨胀判断拥塞：膨胀就降一档帧率和画质，持续畅通再升一档（同视频通话的拥塞控制）。
// 局域网里往返不会膨胀，始终停在最高档。回声借用 `_gz,1`：服务端只是重复确认已开启的压缩，可以安全重发。
// 最高档与 start.sh 的默认值一致；服务端会把帧率记成会话默认，所以每次连上都先回到最高档。
const LEVELS = [{ fps: 60, crf: 25 }, { fps: 30, crf: 30 }, { fps: 20, crf: 35 }, { fps: 15, crf: 35 }] as const;
const PROBE_MS = 500;
const CONGESTED_MS = 150;
const SEVERE_MS = 500;
const CALM_MS = 40;
const STEP_UP_AFTER_MS = 8000;
const STEP_DOWN_GAP_MS = 1000;

class LinkGovernor {
  private readonly timer: number;
  private readonly samples: number[] = [];
  private level = 0;
  private sentAt = 0;
  private calmSince = performance.now();
  private changedAt = 0;

  // 当前档位和最近一次往返写在画面 iframe 的 data-link-* 上，排查远程卡顿时可以直接看。
  constructor(private readonly transport: StreamTransport, private readonly report: HTMLElement) {
    transport.addEventListener("message", this.onMessage);
    this.timer = window.setInterval(this.probe, PROBE_MS);
    this.apply(0, performance.now());
  }

  stop() {
    window.clearInterval(this.timer);
    this.transport.removeEventListener("message", this.onMessage);
  }

  // 上一次回声还没回来就不再发；等得太久本身就是拥塞的证据。
  private readonly probe = () => {
    if (this.transport.readyState !== WebSocket.OPEN) return;
    const now = performance.now();
    if (this.sentAt) {
      const waited = now - this.sentAt - this.floor();
      if (waited > CONGESTED_MS) this.congested(now, waited);
      return;
    }
    this.sentAt = now;
    this.transport.send("_gz,1");
  };

  private readonly onMessage = (event: { data?: unknown }) => {
    if (event.data !== "_gz,1" || !this.sentAt) return;
    const now = performance.now();
    const rtt = now - this.sentAt;
    this.sentAt = 0;
    this.report.dataset.linkRtt = String(Math.round(rtt));
    // 1. 最近 30 秒的最低往返当作链路本身的延迟。
    this.samples.push(rtt);
    if (this.samples.length > 30_000 / PROBE_MS) this.samples.shift();
    // 2. 明显高于它就是在排队；一直贴着它就逐级恢复。
    const inflation = rtt - this.floor();
    if (inflation > CONGESTED_MS) this.congested(now, inflation);
    else if (inflation > CALM_MS) this.calmSince = now;
    else if (now - this.calmSince > STEP_UP_AFTER_MS && this.level > 0) this.apply(this.level - 1, now);
  };

  private floor(): number {
    return this.samples.length ? Math.min(...this.samples) : 0;
  }

  // 严重排队一次降两档，尽快把积压的帧排空。
  private congested(now: number, inflation: number) {
    this.calmSince = now;
    if (now - this.changedAt <= STEP_DOWN_GAP_MS || this.level === LEVELS.length - 1) return;
    this.apply(Math.min(LEVELS.length - 1, this.level + (inflation > SEVERE_MS ? 2 : 1)), now);
  }

  private apply(level: number, now: number) {
    this.level = level;
    this.report.dataset.linkLevel = String(level);
    this.changedAt = now;
    this.calmSince = now;
    this.transport.send(`_arg_fps,${LEVELS[level].fps}`);
    this.transport.send(`_crf,${LEVELS[level].crf}`);
  }
}
