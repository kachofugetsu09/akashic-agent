// 面板的唯一呈现规则：把运行时、连接、Agent 活动和观看目标这几组事实映射成界面。
// 纯函数，不碰 DOM；index.tsx 只负责采集事实和执行动作，文案与优先级都在这里。

export type RuntimeState = "loading" | "sleeping" | "starting" | "ready" | "stopping" | "failed";
export type LinkState = "idle" | "connecting" | "connected" | "lost" | "failed" | "stale";

export interface PanelFacts {
  readonly runtime: RuntimeState;
  readonly runtimeError: string;
  readonly link: LinkState;
  readonly linkError: string;
  /** Agent 有进行中的浏览器调用：你仍然可以操作，它的下一步会等你停手。 */
  readonly agentActive: boolean;
  /** Agent 正在执行单个浏览器操作：这一两秒里挡住你的键鼠，做完立即放开。 */
  readonly agentBusy: boolean;
  /** 最前的标签属于匿名 Context：不带你的登录。 */
  readonly anonymous: boolean;
  readonly wakeRequested: boolean;
  readonly restartRequested: boolean;
  /** 最近一次状态读取失败的原因；成功读取后清空。 */
  readonly statusError: string;
}

export type ActionId = "wake" | "restart" | "reconnect" | "reload" | "copy-error";
export type Tone = "idle" | "busy" | "ok" | "warn" | "error";
export type Icon = "moon" | "spinner" | "alert" | "refresh" | "unplug";
export type FrameKind = "none" | "anonymous";

export interface ActionView {
  readonly id: ActionId;
  readonly label: string;
  readonly emphasis: "primary" | "plain";
  /** 有值时按钮不可用，原因替代胶囊里的说明。 */
  readonly blockedBy?: string;
}

export interface OverlayView {
  readonly icon: Icon;
  readonly text: string;
  readonly actions: readonly ActionView[];
  readonly details?: string;
}

export interface PanelView {
  readonly tone: Tone;
  readonly title: string;
  readonly detail: string;
  /** 有画面时的唯一主操作，放在状态胶囊里；没有画面时为 null，操作在遮罩里。 */
  readonly action: ActionView | null;
  readonly overlay: OverlayView | null;
  readonly frame: { readonly kind: FrameKind };
  /** 你可以直接操作：不显示状态胶囊，像本地浏览器一样。 */
  readonly direct: boolean;
}

/** 谁在动谁用：画面连上后随时可以操作，只有 Agent 正在执行一步时挡住。 */
export function canInputDirectly(f: PanelFacts): boolean {
  return f.link === "connected" && !f.agentBusy;
}

const NO_FRAME = { kind: "none" } as const;

// 没有画面的情况：运行时未就绪、连接未建立或已失效。按严重程度从高到低判断。
function withoutScreen(f: PanelFacts): PanelView | null {
  if (f.link === "stale") return blank("warn", "Akashic 已更新", "刷新页面后才能继续看画面",
    { icon: "refresh", text: "刷新页面后才能继续看画面，浏览器里的任务不受影响。",
      actions: [{ id: "reload", label: "刷新页面", emphasis: "primary" }] });
  // 读不到状态时不猜运行时和连接；已连上的画面不依赖轮询，继续显示。
  if (f.statusError && f.link !== "connected") return blank("warn", "读不到浏览器状态", "每秒自动重试",
    { icon: "alert", text: "Akashic 暂时没有返回浏览器状态，正在每秒重试。持续出现时请确认浏览器容器在运行。",
      actions: [], details: f.statusError });
  if (f.restartRequested && f.runtime !== "ready") return blank("busy", "正在重启浏览器", "会先结束卡住的进程，再重新启动",
    { icon: "spinner", text: "正在结束卡住的进程并重新启动浏览器…", actions: [] });
  if (f.runtime === "failed") return blank("error", "浏览器出错了", "可以重启浏览器；反复出现时请复制详情",
    { icon: "alert", text: "浏览器的运行环境异常退出。重启会结束卡住的进程再重新启动；出错前最后一刻的改动可能没有保存。如果反复出现，请把详情发给维护者。",
      actions: [{ id: "restart", label: "重启浏览器", emphasis: "primary" }, { id: "copy-error", label: "复制详情", emphasis: "plain" }],
      details: f.runtimeError });
  if (f.runtime === "loading") return blank("idle", "正在读取浏览器状态", "",
    { icon: "spinner", text: "正在读取浏览器状态…", actions: [] });
  if (f.runtime === "starting" || (f.wakeRequested && f.runtime === "sleeping" && !f.runtimeError))
    return blank("busy", "正在唤醒", "通常一秒左右",
      { icon: "spinner", text: "正在启动浏览器…", actions: [] });
  if (f.runtime === "stopping") return blank("idle", "浏览器正在休息", "",
    { icon: "moon", text: "浏览器正在进入休息状态…", actions: [] });
  if (f.runtime === "sleeping" && f.runtimeError) return blank("error", "没能唤醒浏览器", "可以重试；反复失败时请复制详情",
    { icon: "alert", text: "唤醒时出错了。可以重试；如果反复失败，请把详情发给维护者。",
      actions: [{ id: "wake", label: "重试", emphasis: "primary" }, { id: "copy-error", label: "复制详情", emphasis: "plain" }],
      details: f.runtimeError });
  if (f.runtime === "sleeping") return blank("idle", "浏览器在休息", "空闲 10 分钟后自动休息，登录过的网站都会保留",
    { icon: "moon", text: "空闲 10 分钟后自动休息。你登录过的网站都还在，Agent 需要时会自己唤醒。",
      actions: [{ id: "wake", label: "唤醒浏览器", emphasis: "primary" }] });
  if (f.link === "failed") return blank("error", "画面连接失败", "浏览器本身仍在运行",
    { icon: "unplug", text: f.linkError || "没能连接到浏览器画面。", actions: [{ id: "reconnect", label: "重新连接", emphasis: "primary" }] });
  if (f.link === "lost") return blank("warn", "连接中断，正在重连", "浏览器本身没有停止",
    { icon: "spinner", text: "画面连接中断，正在重新连接…", actions: [{ id: "reconnect", label: "立即重连", emphasis: "plain" }] });
  if (f.link === "idle" || f.link === "connecting") return blank("busy", "正在连接画面", "",
    { icon: "spinner", text: "正在连接画面…", actions: [] });
  return null;
}

function blank(tone: Tone, title: string, detail: string, overlay: OverlayView): PanelView {
  return { tone, title, detail, action: null, overlay, frame: NO_FRAME, direct: false };
}

// 有画面时没有接管：Agent 在用就用胶囊轻声说明，你随时可以动手；它不在用时不显示胶囊。
function withScreen(f: PanelFacts): PanelView {
  const frame = { kind: f.anonymous ? "anonymous" : "none" } as const;
  if (f.agentActive || f.agentBusy) return { tone: "busy", title: f.agentBusy ? "Agent 正在操作…" : "Agent 正在使用浏览器",
    detail: f.anonymous ? "匿名浏览器 · 你动手时它会先停下" : "你可以随时操作，它会等你停手", action: null, overlay: null,
    frame, direct: false };
  return { tone: "ok", title: "", detail: "", action: null, overlay: null, frame, direct: true };
}

export function present(facts: PanelFacts): PanelView {
  return withoutScreen(facts) ?? withScreen(facts);
}
