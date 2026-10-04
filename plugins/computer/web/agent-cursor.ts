import type { WebHostContextV1 } from "@akashic/web-ui-v1";

interface Point {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly kind: "click" | "move";
}
interface CursorState {
  readonly revision: number;
  readonly point: Point | null;
  readonly ttlMs: number;
  readonly error: string;
}

/** 在只读传输边界检查坐标；展示不能反向发送输入。 */
function checkState(raw: string): CursorState {
  const value = JSON.parse(raw) as Partial<CursorState> | null;
  if (!value || !Number.isInteger(value.revision) || value.revision! < 0
    || typeof value.ttlMs !== "number" || !Number.isFinite(value.ttlMs)
    || value.ttlMs < 0 || value.ttlMs > 5000 || typeof value.error !== "string") {
    throw new Error("Computer 操作位置回执无效");
  }
  if (value.point !== null) {
    const point = value.point;
    if (!point || ![point.x, point.y, point.width, point.height].every(Number.isFinite)
      || point.width <= 0 || point.height <= 0 || point.x < 0 || point.y < 0
      || point.x >= point.width || point.y >= point.height
      || !["click", "move"].includes(point.kind)) {
      throw new Error("Computer 操作坐标无效");
    }
  }
  return value as CursorState;
}

/** 光标只拥有临时画面、订阅和动画；人工输入或断线立即清除。 */
export class AgentCursor {
  private readonly element = document.createElement("div");
  private readonly notice = document.createElement("div");
  private readonly socket: WebSocket;
  private readonly resize: ResizeObserver;
  private readonly reduced = matchMedia("(prefers-reduced-motion: reduce)");
  private point: Point | null = null;
  private shown: { x: number; y: number } | null = null;
  private revision = -1;
  private expiry = 0;
  private frame = 0;
  private pulse: Animation | null = null;
  private closed = false;

  constructor(
    private readonly host: HTMLElement,
    ctx: WebHostContextV1,
    private readonly viewport: () => DOMRect | null,
  ) {
    this.element.className = "computer-agent-cursor";
    this.element.setAttribute("aria-hidden", "true");
    this.element.hidden = true;
    this.element.innerHTML = `<svg viewBox="0 0 24 24"><path
      d="M2 2 L8.5 21 L12 13 L20 9.5 Z"/></svg>`;
    this.notice.className = "computer-cursor-notice";
    this.notice.setAttribute("role", "status");
    this.notice.hidden = true;
    host.append(this.element, this.notice);
    this.resize = new ResizeObserver(() => this.place(false));
    this.resize.observe(host);
    this.socket = new WebSocket(ctx.http.webSocketUrl("/api/dashboard/computer/cursor"));
    this.socket.addEventListener("message", (event) => {
      if (this.closed) return;
      try {
        const state = checkState(event.data);
        if (state.revision <= this.revision) return;
        this.revision = state.revision;
        const previous = this.shown;
        this.hide();
        this.notice.textContent = state.error;
        this.notice.hidden = !state.error;
        if (!state.point || state.ttlMs === 0) return;
        this.point = state.point;
        this.shown = previous;
        this.place(true);
        this.expiry = window.setTimeout(() => this.hide(), state.ttlMs);
      } catch (error) {
        this.unavailable(String(error));
        this.socket.close(1002, "Invalid cursor receipt");
      }
    });
    this.socket.addEventListener("close", () => {
      if (!this.closed) this.unavailable("操作位置标记已断开；重新连接 Computer 可恢复。");
    });
  }

  /** 坐标对齐实际视频区域，留白、全屏和窄屏缩放不改变热点。 */
  private place(animate: boolean) {
    if (!this.point || this.closed) return;
    const view = this.viewport();
    if (!view) { this.hide(); return; }
    const bounds = this.host.getBoundingClientRect();
    const end = { x: view.x - bounds.x + this.point.x * view.width / this.point.width,
      y: view.y - bounds.y + this.point.y * view.height / this.point.height };
    const start = this.shown ?? end;
    window.cancelAnimationFrame(this.frame);
    this.element.hidden = false;
    // 1. 只有操作后的短动效；减少动态效果时直接对齐目标。
    const duration = Number.parseFloat(getComputedStyle(this.element).getPropertyValue("--ak-sys-duration-medium"));
    if (!animate || this.reduced.matches || !Number.isFinite(duration) || duration <= 0) {
      this.draw(end, 0, 1);
      return;
    }
    const dx = end.x - start.x, dy = end.y - start.y;
    const distance = Math.hypot(dx, dy);
    const bend = distance > 196 ? Math.min(32, distance * 0.12) : 0;
    const began = performance.now();
    // 2. 长移动略带弧度和转向，结束后恢复普通箭头；不延迟真实输入。
    const step = (now: number) => {
      const progress = this.reduced.matches ? 1 : Math.min(1, (now - began) / duration);
      const ratio = 1 - (1 - progress) ** 3;
      const wave = Math.sin(Math.PI * ratio);
      const curve = bend * wave / Math.max(1, distance);
      this.draw({ x: start.x + dx * ratio - dy * curve,
        y: start.y + dy * ratio + dx * curve },
        bend ? (Math.atan2(dy, dx) * 180 / Math.PI + 135) * wave : 0, 1 - 0.12 * wave);
      this.frame = progress < 1 ? requestAnimationFrame(step) : 0;
    };
    this.frame = requestAnimationFrame(step);
    if (this.point.kind === "click") this.pulse = this.element.querySelector("svg")!.animate(
      [{ transform: "scale(0.86)" }, { transform: "scale(1)" }], { duration, easing: "ease-out" });
  }

  private draw(point: { x: number; y: number }, rotation: number, scale: number) {
    this.shown = point;
    this.element.style.transform = `translate(${point.x - 2}px, ${point.y - 2}px) rotate(${rotation}deg) scale(${scale})`;
  }

  hide() {
    this.point = null;
    this.shown = null;
    window.clearTimeout(this.expiry);
    window.cancelAnimationFrame(this.frame);
    this.pulse?.cancel();
    this.pulse = null;
    this.element.hidden = true;
  }

  private unavailable(message: string) {
    this.hide();
    this.notice.textContent = message;
    this.notice.hidden = false;
  }

  destroy() {
    this.closed = true;
    this.hide();
    this.socket.close();
    this.resize.disconnect();
    this.element.remove();
    this.notice.remove();
  }
}
