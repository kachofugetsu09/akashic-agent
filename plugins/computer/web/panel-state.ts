// 面板的唯一呈现规则：把运行时、连接、控制权和观看目标这几组事实映射成界面。
// 纯函数，不碰 DOM；index.tsx 只负责采集事实和执行动作，文案与优先级都在这里。

export type RuntimeState = "loading" | "sleeping" | "starting" | "ready" | "stopping" | "failed";
export type LinkState = "idle" | "connecting" | "connected" | "lost" | "failed" | "stale";
export type ControlState = "none" | "taking" | "mine" | "other";

export interface PanelFacts {
  readonly runtime: RuntimeState;
  readonly runtimeError: string;
  readonly link: LinkState;
  readonly linkError: string;
  readonly control: ControlState;
  readonly agentActive: boolean;
  /** Agent 有进行中的回合：回合结束前它随时会继续操作，人要先接管。 */
  readonly agentTurn: boolean;
  /** 最前的标签属于匿名 Context：不带你的登录。 */
  readonly anonymous: boolean;
  readonly wakeRequested: boolean;
  readonly restartRequested: boolean;
  /** 最近一次状态读取失败的原因；成功读取后清空。 */
  readonly statusError: string;
}

export type ActionId = "take" | "release" | "cancel" | "wake" | "restart" | "reconnect" | "reload" | "copy-error";
export type Tone = "idle" | "busy" | "ok" | "you" | "warn" | "error";
export type Icon = "moon" | "spinner" | "alert" | "refresh" | "unplug";
export type FrameKind = "none" | "anonymous" | "control";

export interface ActionView {
  readonly id: ActionId;
  readonly label: string;
  readonly emphasis: "primary" | "control" | "plain";
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

/** 画面已连上、没人接管、Agent 也不在用时，点击和按键直接送到浏览器（参考 Cursor：Agent 加锁时才挡住）。 */
export function canInputDirectly(f: PanelFacts): boolean {
  return f.link === "connected" && f.control === "none" && !f.agentActive && !f.agentTurn;
}

const take: ActionView = { id: "take", label: "接管", emphasis: "primary" };
const release: ActionView = { id: "release", label: "交还给 Agent", emphasis: "control" };
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
  if ((f.link === "idle" || f.link === "connecting") && f.control === "taking")
    return blank("busy", "正在接管…", "Agent 做完当前这一步就会停下",
      { icon: "spinner", text: "等 Agent 做完当前这一步，再把操作交给你。", actions: [{ id: "cancel", label: "取消", emphasis: "plain" }] });
  if (f.link === "idle" || f.link === "connecting") return blank("busy", "正在连接画面", "",
    { icon: "spinner", text: "正在连接画面…", actions: [] });
  return null;
}

function blank(tone: Tone, title: string, detail: string, overlay: OverlayView): PanelView {
  return { tone, title, detail, action: null, overlay, frame: NO_FRAME, direct: false };
}

// 有画面时：Agent 不在用就直接操作、不显示胶囊；Agent 在用时胶囊说明情况并给出接管。外框颜色表达谁在操作。
function withScreen(f: PanelFacts): PanelView {
  const anonymous = f.anonymous;
  if (f.control === "mine") return { tone: "you", title: "你正在操作",
    detail: anonymous ? "匿名浏览器 · Agent 已暂停" : "Agent 已暂停，交还后会继续", action: release, overlay: null,
    frame: { kind: "control" }, direct: false };
  const frame = { kind: anonymous ? "anonymous" : "none" } as const;
  if (f.control === "taking") return { tone: "busy", title: "正在接管…", detail: "Agent 做完当前这一步就会停下",
    action: { id: "cancel", label: "取消", emphasis: "plain" }, overlay: null, frame, direct: false };
  if (f.control === "other") return { tone: "you", title: "另一个窗口正在操作", detail: "Agent 已暂停，你仍然可以观看",
    action: { ...take, blockedBy: "等对方交还后才能接管" }, overlay: null, frame, direct: false };
  if (f.agentActive || f.agentTurn) return { tone: "busy", title: "Agent 正在使用浏览器",
    detail: anonymous ? "匿名浏览器 · 接管后 Agent 会暂停" : "接管后 Agent 会暂停", action: take, overlay: null, frame, direct: false };
  return { tone: "ok", title: "", detail: "", action: null, overlay: null, frame, direct: true };
}

export function present(facts: PanelFacts): PanelView {
  return withoutScreen(facts) ?? withScreen(facts);
}
