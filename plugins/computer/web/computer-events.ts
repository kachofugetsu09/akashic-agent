// 浏览器面板唯一的状态来源：(重)连前用一次 HTTP 读取识别页面过期和接口错误，之后只收推送。
// 状态与光标共用一条只读流；命令（唤醒、接管、交还）仍走各自的入口。
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import type { Target } from "./panel-view";

export interface ComputerState {
  readonly noticeId: number;
  readonly active: boolean;
  readonly turns: readonly string[];
  readonly browser: { readonly state: string; readonly error: string };
  readonly targets: Target[];
  readonly control: boolean;
}

const RUNTIME_STATES: readonly string[] = ["sleeping", "starting", "ready", "stopping", "failed"];
const RETRY_MS = 1000;

// 格式不对就整体拒绝，不渲染半截状态。
function checkState(value: Partial<ComputerState>): ComputerState {
  if (!Number.isInteger(value.noticeId) || typeof value.active !== "boolean"
    || !Array.isArray(value.turns) || value.turns.some(turn => typeof turn !== "string")
    || !RUNTIME_STATES.includes(value.browser?.state ?? "") || typeof value.browser?.error !== "string"
    || !Array.isArray(value.targets) || typeof value.control !== "boolean"
    || value.targets.some(target => typeof target.id !== "string" || typeof target.title !== "string"
      || typeof target.url !== "string" || typeof target.anonymous !== "boolean" || typeof target.front !== "boolean"
      || typeof target.agent !== "boolean" || (target.turn !== null && typeof target.turn !== "string")))
    throw new Error("Computer 状态回执无效");
  return value as ComputerState;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** 事件：state（完整快照）、cursor（光标原始回执）、error（读取失败，自动重试）、stale（页面过期，停止）。 */
export class ComputerEvents extends EventTarget {
  private latestCursor: { value: object; expiresAt: number } | null = null;
  private socket: WebSocket | null = null;
  private timer = 0;
  private closed = false;

  constructor(private readonly ctx: WebHostContextV1) {
    super();
    void this.open();
  }

  /** 最近一次光标回执，ttlMs 换算成剩余寿命；新建的光标层从这里补齐当前位置，不会让过期光标复活。 */
  get cursor(): unknown {
    if (!this.latestCursor) return null;
    const ttlMs = Math.max(0, this.latestCursor.expiresAt - Date.now());
    return { ...this.latestCursor.value, ttlMs };
  }

  close() {
    this.closed = true;
    window.clearTimeout(this.timer);
    const socket = this.socket;
    this.socket = null;
    socket?.close();
  }

  private async open() {
    // 1. 握手前的 HTTP 读取：WebSocket 被拒时拿不到状态码和过期标记。
    let response: Response;
    try {
      response = await this.ctx.http.request("/api/dashboard/computer/activity", { cache: "no-store" });
    } catch (error) {
      this.retry(message(error));
      return;
    }
    if (this.closed) return;
    if (response.headers.get("X-Akashic-Web-Stale") === "1") { this.emit("stale", null); return; }
    if (!response.ok) { this.retry(`状态接口返回 ${response.status}`); return; }
    // 2. 推送：首条是完整快照，之后只在变化时到达。
    const socket = new WebSocket(this.ctx.http.webSocketUrl("/api/dashboard/computer/events"));
    this.socket = socket;
    socket.addEventListener("message", event => {
      try {
        const value = JSON.parse(String(event.data)) as Partial<ComputerState> & { type?: string; ttlMs?: unknown };
        if (value?.type === "cursor") { this.remember(value); this.emit("cursor", value); }
        else if (value?.type === "state") this.emit("state", checkState(value));
        else throw new Error("Computer 状态回执无效");
      } catch (error) {
        this.socket = null;
        socket.close(1002, "Invalid Computer event");
        this.retry(message(error));
      }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.retry("状态连接中断");
    });
  }

  private remember(value: { ttlMs?: unknown }) {
    const ttlMs = typeof value.ttlMs === "number" && Number.isFinite(value.ttlMs) ? value.ttlMs : 0;
    this.latestCursor = { value, expiresAt: Date.now() + ttlMs };
  }

  private retry(reason: string) {
    if (this.closed) return;
    this.emit("error", reason);
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => void this.open(), RETRY_MS);
  }

  private emit(type: string, detail: unknown) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}
