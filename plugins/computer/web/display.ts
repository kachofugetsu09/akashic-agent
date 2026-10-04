import type { WebHostContextV1 } from "@akashic/web-ui-v1";

interface StreamWindow extends Window {
  computerInputHandlers?: InputHandlers;
  fps?: number;
  selkiesTransport?: { readonly readyState: number; close(): void;
    addEventListener(type: "close" | "message", listener: (event: { reason?: string; data?: unknown }) => void): void };
  webrtcInput?: {
    readonly element: HTMLElement;
    attach_context(): void;
    detach_context(): void;
    _sendKeyEvent(keysym: number, code: string, down: boolean): void;
  };
}

interface InputHandlers {
  keydown(event: KeyboardEvent): void;
  keyup(event: KeyboardEvent): void;
  paste(event: ClipboardEvent): void;
  touch(event: Event): void;
  blur(): void;
}

/** 在独立 iframe 中运行固定版本的流客户端，避免全局变量和输入互相干扰。 */
export class ComputerDisplay extends EventTarget {
  private readonly frame = document.createElement("iframe");
  private readonly abort = new AbortController();
  private readonly urls: string[] = [];
  private timer = 0;
  private connected = false;
  private closed = false;
  private inputReady = false;
  private transportWatched = false;
  private clipboardId = 0;
  private readonly clipboardRequests = new Map<number, {
    resolve(): void;
    reject(error: Error): void;
    timer: number;
  }>();

  constructor(
    private readonly host: HTMLElement,
    private readonly ctx: WebHostContextV1,
    private readonly handlers: InputHandlers,
  ) {
    super();
    this.frame.className = "computer-stream";
    this.frame.title = "Computer 远程桌面";
    this.frame.allow = "clipboard-read; clipboard-write; fullscreen";
    this.frame.addEventListener("load", () => this.watch());
    void this.load().catch((error: unknown) => {
      if (this.closed) return;
      this.dispatchEvent(new CustomEvent("error", { detail: { reason: String(error) } }));
      this.disconnect();
    });
  }

  private get stream(): StreamWindow | null {
    return this.frame.contentWindow as StreamWindow | null;
  }

  /** 只从当前 generation 读取客户端，再交入同一身份的完整 WebSocket 地址。 */
  private async load() {
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
    const socket = this.ctx.http.webSocketUrl("/api/dashboard/computer/stream");
    const moduleUrl = this.blob(source, "text/javascript");
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
    // 3. 每次连接独占 iframe 与 Blob；dispose 统一释放。
    this.frame.src = this.blob(html, "text/html");
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

  /** 首帧和输入一起就绪才报告连接；断线交给面板的唯一重连 owner。 */
  private watch() {
    if (this.closed || this.timer) return;
    const deadline = Date.now() + 15_000;
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
          if (/superseded/i.test(event.reason ?? "")) this.dispatchEvent(new Event("superseded"));
          this.disconnect();
        });
      }
      const input = stream?.webrtcInput;
      const document = this.frame.contentDocument;
      if (input && document && !this.inputReady) {
        this.inputReady = true;
        stream!.computerInputHandlers = this.handlers;
        for (const name of ["pointerdown", "pointermove", "pointerup", "wheel"]) {
          document.addEventListener(name, this.handlers.touch, true);
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
        this.dispatchEvent(new Event("connect"));
      }
      if (transport?.readyState === WebSocket.CLOSED || (!this.connected && Date.now() > deadline)) {
        this.disconnect();
      }
    }, 100);
  }

  focus(options?: FocusOptions) {
    const input = this.stream?.webrtcInput;
    input?.attach_context();
    input?.element.focus(options);
  }

  blur() {
    this.stream?.webrtcInput?.detach_context();
  }

  sendKey(keysym: number, code: string, down: boolean) {
    this.stream?.webrtcInput?._sendKeyEvent(keysym, code, down);
  }

  /** 回执到达后才允许发送粘贴按键，断线或超时明确失败。 */
  clipboardPasteFrom(text: string): Promise<void> {
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
    window.removeEventListener("message", this.onMessage);
    this.blur();
    this.stream?.selkiesTransport?.close();
    this.frame.remove();
    for (const url of this.urls) URL.revokeObjectURL(url);
    this.dispatchEvent(new Event("disconnect"));
  }
}
