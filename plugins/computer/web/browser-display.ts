import type { WebHostContextV1 } from "@akashic/web-ui-v1";

/** 一份真实匿名页面画面；输入只沿持有权限的操作连接发送。 */
export class BrowserDisplay extends EventTarget {
  private readonly image = document.createElement("img");
  private readonly socket: WebSocket;
  private readonly pending = new Map<number, { resolve(): void; reject(error: Error): void }>();
  private serial = 0;
  private url = "";
  private connected = false;
  private closed = false;
  private drawing = Promise.resolve();
  private timer: number;

  constructor(host: HTMLElement, ctx: WebHostContextV1, target: string, private readonly control: string) {
    super();
    this.image.className = "computer-browser-screen";
    this.image.alt = "匿名浏览器页面";
    this.image.tabIndex = 0;
    this.image.draggable = false;
    this.image.style.touchAction = control ? "none" : "auto";
    const socket = new URL(ctx.http.webSocketUrl("/api/dashboard/computer/browser-stream"));
    socket.searchParams.set("target", target);
    if (control) socket.searchParams.set("control", control);
    this.socket = new WebSocket(socket);
    this.socket.binaryType = "blob";
    this.socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const ack = JSON.parse(event.data) as { id: number };
        this.pending.get(ack.id)?.resolve();
        this.pending.delete(ack.id);
        return;
      }
      const frame = event.data as Blob;
      this.drawing = this.drawing.then(async () => {
        if (this.closed) return;
        const url = URL.createObjectURL(frame);
        this.image.src = url;
        try { await this.image.decode(); }
        finally { if (this.url) URL.revokeObjectURL(this.url); this.url = url; }
        if (this.closed) { URL.revokeObjectURL(url); return; }
        if (!this.connected) {
          this.connected = true;
          window.clearTimeout(this.timer);
          this.dispatchEvent(new Event("connect"));
        }
      }).catch((error: unknown) => this.fail(String(error)));
    };
    this.socket.onclose = () => {
      if (!this.closed) this.fail("匿名页面已关闭或连接中断");
    };
    this.socket.onerror = () => this.fail("匿名浏览器连接失败，请重新连接");
    this.timer = window.setTimeout(() => this.fail("匿名页面首帧超时"), 15000);
    for (const name of ["pointerdown", "pointermove", "pointerup", "wheel"] as const)
      this.image.addEventListener(name, (event) => this.pointer(event), { passive: false });
    this.image.addEventListener("contextmenu", (event) => event.preventDefault());
    host.replaceChildren(this.image);
  }

  private fail(reason: string) {
    if (this.closed) return;
    this.dispatchEvent(new CustomEvent("error", { detail: { reason } }));
    this.disconnect();
  }

  private send(input: object): Promise<void> {
    if (!this.control || this.closed || this.socket.readyState !== WebSocket.OPEN)
      return Promise.reject(new Error("请先接管操作"));
    if (this.pending.size >= 128) return Promise.reject(new Error("输入连接繁忙，请等待"));
    const id = ++this.serial;
    const receipt = new Promise<void>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.socket.send(JSON.stringify({ id, input }));
    return receipt;
  }

  private pointer(event: PointerEvent | WheelEvent) {
    if (!this.control || !this.connected) return;
    event.preventDefault();
    const rect = this.image.getBoundingClientRect();
    const scale = Math.min(rect.width / this.image.naturalWidth, rect.height / this.image.naturalHeight);
    const width = this.image.naturalWidth * scale, height = this.image.naturalHeight * scale;
    const x = (event.clientX - rect.x - (rect.width - width) / 2) / scale;
    const y = (event.clientY - rect.y - (rect.height - height) / 2) / scale;
    if (x < 0 || y < 0 || x >= this.image.naturalWidth || y >= this.image.naturalHeight) return;
    if (event.type === "pointerdown") {
      this.image.focus();
      this.image.setPointerCapture((event as PointerEvent).pointerId);
    }
    const type = { pointerdown: "mousePressed", pointerup: "mouseReleased",
      pointermove: "mouseMoved", wheel: "mouseWheel" }[event.type as "pointerdown"];
    const button = event.type === "wheel" || event.type === "pointermove" ? "none"
      : ["left", "middle", "right"][event.button];
    void this.send({ kind: "mouse", type, x, y, button,
      deltaX: event instanceof WheelEvent ? event.deltaX : 0,
      deltaY: event instanceof WheelEvent ? event.deltaY : 0 }).catch(error => this.fail(String(error)));
  }

  key(event: KeyboardEvent) {
    event.preventDefault();
    if (!this.control) return;
    const modifiers = (event.altKey ? 1 : 0) | (event.ctrlKey ? 2 : 0)
      | (event.metaKey ? 4 : 0) | (event.shiftKey ? 8 : 0);
    void this.send({ kind: "key", type: event.type === "keydown" ? "keyDown" : "keyUp",
      key: event.key, code: event.code, modifiers }).catch(error => this.fail(String(error)));
  }

  clipboardPasteFrom(text: string) { return this.send({ kind: "text", text }); }
  focus(options?: FocusOptions) { this.image.focus(options); }
  blur() { this.image.blur(); }
  disconnect() {
    if (this.closed) return;
    this.closed = true;
    window.clearTimeout(this.timer);
    for (const receipt of this.pending.values()) receipt.reject(new Error("操作连接已关闭"));
    this.pending.clear();
    this.socket.close();
    if (this.url) URL.revokeObjectURL(this.url);
    this.image.remove();
    this.dispatchEvent(new Event("disconnect"));
  }
}
