import type { ComputerEvents } from "./computer-events";
import { Motion, glowFor, rippleAt, squishAt, RIPPLE_MS, type Vec } from "./cursor-motion";

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
function checkState(raw: unknown): CursorState {
  const value = raw as Partial<CursorState> | null;
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

const NAME_MS = 2000;
const LEAVE_MS = 200;
const CLICK_TARGET_PX = 24;

/** 光标只拥有临时画面、订阅和动画；人工输入或断线立即清除。运动数学在 cursor-motion。 */
export class AgentCursor {
  private readonly layer = document.createElement("div");
  private readonly pointer = document.createElement("div");
  private readonly glyph = document.createElement("div");
  private readonly glow = document.createElement("div");
  private readonly ripple = document.createElement("div");
  private readonly badge = document.createElement("div");
  private readonly notice = document.createElement("div");
  private readonly resize: ResizeObserver;
  private readonly reduced = matchMedia("(prefers-reduced-motion: reduce)");
  private point: Point | null = null;
  private shown: Vec | null = null;
  private motion: Motion | null = null;
  private motionStart = 0;
  private clickAt: { at: number; x: number; y: number } | null = null;
  private revision = -1;
  private expiry = 0;
  private leaving = 0;
  private quiet = 0;
  private frame = 0;
  private closed = false;

  constructor(
    private readonly host: HTMLElement,
    private readonly events: ComputerEvents,
    private readonly viewport: () => DOMRect | null,
  ) {
    this.build();
    host.append(this.layer, this.notice);
    this.resize = new ResizeObserver(() => this.place(false));
    this.resize.observe(host);
    host.addEventListener("pointermove", this.reveal);
    events.addEventListener("cursor", this.onCursor);
    events.addEventListener("error", this.onLost);
    if (events.cursor) this.receive(events.cursor);
  }

  private readonly onCursor = (event: Event) => this.receive((event as CustomEvent<unknown>).detail);

  // 状态流重连后 gateway 可能已重启，revision 从头计数。
  private readonly onLost = () => {
    this.revision = -1;
    this.unavailable("操作位置标记已断开，正在重连…");
  };

  private receive(raw: unknown) {
    if (this.closed) return;
    try {
      const state = checkState(raw);
      if (state.revision <= this.revision) return;
      this.revision = state.revision;
      this.notice.textContent = state.error;
      this.notice.hidden = !state.error;
      if (!state.point || state.ttlMs === 0) { this.hide(); return; }
      this.point = state.point;
      this.place(true);
      window.clearTimeout(this.expiry);
      this.expiry = window.setTimeout(() => this.hide(), state.ttlMs);
    } catch (error) {
      this.unavailable(String(error));
    }
  }

  // 结构：光晕和涟漪在指针下层；箭头和名字徽章同在指针层，随指针移动。
  private build() {
    this.layer.className = "computer-agent-layer";
    this.layer.setAttribute("aria-hidden", "true");
    this.layer.hidden = true;
    this.glow.className = "computer-agent-glow";
    this.ripple.className = "computer-agent-ripple";
    this.pointer.className = "computer-agent-pointer";
    this.glyph.className = "computer-agent-glyph";
    this.glyph.innerHTML = `<svg viewBox="0 0 24 24"><path class="computer-agent-arrow"
      d="M3 2 L10 21 L12.6 13.4 L20 11 Z"/></svg>`;
    this.badge.className = "computer-agent-badge";
    this.badge.innerHTML = `<i></i><span>Akashic</span>`;
    this.pointer.append(this.glyph, this.badge);
    this.layer.append(this.glow, this.ripple, this.pointer);
    this.notice.className = "computer-cursor-notice";
    this.notice.setAttribute("role", "status");
    this.notice.hidden = true;
  }

  /** 坐标对齐实际视频区域，留白、全屏和窄屏缩放不改变热点。 */
  private place(animate: boolean) {
    if (!this.point || this.closed) return;
    const view = this.viewport();
    if (!view) { this.hide(); return; }
    const bounds = this.host.getBoundingClientRect();
    const end = { x: view.x - bounds.x + this.point.x * view.width / this.point.width,
      y: view.y - bounds.y + this.point.y * view.height / this.point.height };
    // 1. 首次出现：直接落在目标处并亮出名字，2 秒后收成圆点。
    if (!this.shown || this.layer.hidden) {
      this.layer.hidden = false;
      this.showName(NAME_MS);
    }
    window.clearTimeout(this.leaving);
    this.layer.removeAttribute("data-leaving");
    const still = !animate || !this.shown || this.reduced.matches;
    // 2. 减少动态效果或尺寸变化时直接对齐；否则沿 Cua 弧线移动，点击在到达后触发。
    this.motion = still ? null : new Motion(this.shown!, end, CLICK_TARGET_PX);
    this.motionStart = performance.now();
    this.clickAt = null;
    if (still) {
      this.draw(end, 0, 0);
      if (animate && this.point.kind === "click") this.clicked(end);
      return;
    }
    this.run();
  }

  private run() {
    if (!this.frame) this.frame = requestAnimationFrame(this.tick);
  }

  private readonly tick = (now: number) => {
    this.frame = 0;
    if (this.closed || this.layer.hidden) return;
    let speed = 0;
    if (this.motion) {
      const pose = this.motion.poseAt(now - this.motionStart);
      speed = pose.speed;
      const glow = glowFor(pose);
      this.glow.style.opacity = glow ? String(glow.alpha) : "0";
      if (glow) this.glow.style.transform = `translate(${glow.x}px, ${glow.y}px) scale(${glow.scale})`;
      // 移动时按水平速度轻微倾斜，静止时回正。
      this.draw(pose, Math.max(-12, Math.min(12, pose.vx * 0.012)), 0);
      if (pose.done) {
        this.motion = null;
        this.glow.style.opacity = "0";
        if (this.point?.kind === "click") this.clicked(pose);
      }
    }
    if (this.clickAt) {
      const age = now - this.clickAt.at;
      const ripple = rippleAt(age);
      this.ripple.style.opacity = ripple ? String(ripple.alpha) : "0";
      if (ripple) {
        this.ripple.style.transform = `translate(${this.clickAt.x}px, ${this.clickAt.y}px) scale(${ripple.radius / 26})`;
        this.ripple.style.borderWidth = `${ripple.width * 26 / ripple.radius}px`;
      }
      if (!this.motion) this.draw(this.shown!, 0, squishAt(age));
      if (age > RIPPLE_MS) this.clickAt = null;
    }
    if (this.motion || this.clickAt || speed > 0) this.run();
  };

  private clicked(at: Vec) {
    if (this.reduced.matches) return;
    this.clickAt = { at: performance.now(), x: at.x, y: at.y };
    this.run();
  }

  private draw(point: Vec, tilt: number, squish: number) {
    this.shown = { x: point.x, y: point.y };
    // 位移在外层，倾斜和挤压只作用于箭头，名字徽章保持水平。
    this.pointer.style.transform = `translate(${point.x - 3}px, ${point.y - 2}px)`;
    this.glyph.style.transform = `rotate(${tilt}deg) scale(${1 - squish}, ${1 - squish * 0.6})`;
  }

  private showName(ms: number) {
    this.badge.removeAttribute("data-quiet");
    window.clearTimeout(this.quiet);
    this.quiet = window.setTimeout(() => this.badge.setAttribute("data-quiet", ""), ms);
  }

  // 指针靠近光标时重新亮出名字，方便确认这是 Agent 而不是自己的鼠标。
  private readonly reveal = (event: PointerEvent) => {
    if (!this.shown || this.layer.hidden) return;
    const bounds = this.host.getBoundingClientRect();
    if (Math.hypot(event.clientX - bounds.x - this.shown.x, event.clientY - bounds.y - this.shown.y) < 36)
      this.showName(1200);
  };

  hide() {
    this.point = null;
    this.motion = null;
    this.clickAt = null;
    window.clearTimeout(this.expiry);
    window.cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.glow.style.opacity = "0";
    this.ripple.style.opacity = "0";
    if (this.layer.hidden) { this.shown = null; return; }
    // 淡出后再移除，下一次出现从目标处重新开始。
    this.layer.setAttribute("data-leaving", "");
    this.leaving = window.setTimeout(() => {
      this.layer.hidden = true;
      this.layer.removeAttribute("data-leaving");
      this.shown = null;
    }, this.reduced.matches ? 0 : LEAVE_MS);
  }

  private unavailable(message: string) {
    this.hide();
    this.notice.textContent = message;
    this.notice.hidden = false;
  }

  destroy() {
    this.closed = true;
    this.hide();
    window.clearTimeout(this.leaving);
    window.clearTimeout(this.quiet);
    this.host.removeEventListener("pointermove", this.reveal);
    this.events.removeEventListener("cursor", this.onCursor);
    this.events.removeEventListener("error", this.onLost);
    this.resize.disconnect();
    this.layer.remove();
    this.notice.remove();
  }
}
