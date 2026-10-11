// 面板的唯一呈现规则：把运行时、连接、控制权和观看目标这几组事实映射成界面。
// 纯函数，不碰 DOM；index.tsx 只负责采集事实和执行动作，文案与优先级都在这里。

export type RuntimeState = "loading" | "sleeping" | "starting" | "ready" | "stopping" | "failed";
export type LinkState = "idle" | "connecting" | "connected" | "lost" | "failed" | "stale" | "ended";
export type ControlState = "none" | "taking" | "mine" | "other";
export type TargetKind = "desktop" | "browser";

export interface PanelFacts {
  readonly runtime: RuntimeState;
  readonly runtimeError: string;
  readonly link: LinkState;
  readonly linkError: string;
  readonly control: ControlState;
  readonly agentActive: boolean;
  readonly targetKind: TargetKind;
  readonly targetTitle: string;
  readonly wakeRequested: boolean;
}

export type ActionId = "take" | "release" | "cancel" | "wake" | "reconnect" | "home" | "reload" | "copy-error";
export type Tone = "idle" | "busy" | "ok" | "you" | "warn" | "error";
export type Icon = "moon" | "spinner" | "alert" | "closed" | "refresh" | "unplug";
export type FrameKind = "none" | "anonymous" | "control";

export interface ActionView {
  readonly id: ActionId;
  readonly label: string;
  readonly emphasis: "primary" | "control" | "plain";
  /** 有值时按钮不可用，原因显示在按钮下方。 */
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
  /** 有画面时的唯一主操作，放在状态条右侧；没有画面时为 null，操作在遮罩里。 */
  readonly action: ActionView | null;
  readonly overlay: OverlayView | null;
  readonly frame: { readonly kind: FrameKind; readonly legend: string };
}

const take: ActionView = { id: "take", label: "接管操作", emphasis: "primary" };
const release: ActionView = { id: "release", label: "交还给 Agent", emphasis: "control" };
const NO_FRAME = { kind: "none", legend: "" } as const;

// 没有画面的情况：运行时未就绪、连接未建立或已失效。按严重程度从高到低判断。
function withoutScreen(f: PanelFacts): PanelView | null {
  if (f.link === "stale") return blank("warn", "Akashic 已更新", "刷新页面后才能继续看画面",
    { icon: "refresh", text: "刷新页面后才能继续看画面，电脑里的任务不受影响。",
      actions: [{ id: "reload", label: "刷新页面", emphasis: "primary" }] });
  if (f.runtime === "failed") return blank("error", "电脑出错了", "需要重启 Akashic 后才能再次唤醒",
    { icon: "alert", text: "电脑的运行环境异常退出。重启 Akashic 后可以重新唤醒；如果反复出现，请把详情发给维护者。",
      actions: [{ id: "copy-error", label: "复制详情", emphasis: "plain" }], details: f.runtimeError });
  if (f.runtime === "loading") return blank("idle", "正在读取电脑状态", "",
    { icon: "spinner", text: "正在读取电脑状态…", actions: [] });
  if (f.runtime === "starting" || (f.wakeRequested && f.runtime === "sleeping" && !f.runtimeError))
    return blank("busy", "正在唤醒", "通常需要几秒钟",
      { icon: "spinner", text: "正在启动电脑和浏览器…", actions: [] });
  if (f.runtime === "stopping") return blank("idle", "电脑正在休息", "",
    { icon: "moon", text: "电脑正在进入休息状态…", actions: [] });
  if (f.runtime === "sleeping" && f.runtimeError) return blank("error", "没能唤醒电脑", "可以重试；反复失败时请复制详情",
    { icon: "alert", text: "唤醒时出错了。可以重试；如果反复失败，请把详情发给维护者。",
      actions: [{ id: "wake", label: "重试", emphasis: "primary" }, { id: "copy-error", label: "复制详情", emphasis: "plain" }],
      details: f.runtimeError });
  if (f.runtime === "sleeping") return blank("idle", "电脑在休息", "空闲 10 分钟后自动休息，登录过的网站都会保留",
    { icon: "moon", text: "空闲 10 分钟后自动休息。你登录过的网站都还在，Agent 需要时会自己唤醒。",
      actions: [{ id: "wake", label: "唤醒电脑", emphasis: "primary" }] });
  if (f.link === "ended") return blank("idle", "这个浏览器已关闭", "任务结束后，匿名浏览器会自动关闭",
    { icon: "closed", text: "匿名浏览器只在一轮任务里存在，任务结束后会自动关闭。",
      actions: [{ id: "home", label: "回到主桌面", emphasis: "primary" }] });
  if (f.link === "failed") return blank("error", "画面连接失败", "电脑本身仍在运行",
    { icon: "unplug", text: f.linkError || "没能连接到电脑画面。", actions: [{ id: "reconnect", label: "重新连接", emphasis: "primary" }] });
  if (f.link === "lost") return blank("warn", "连接中断，正在重连", "电脑本身没有停止",
    { icon: "spinner", text: "画面连接中断，正在重新连接…", actions: [{ id: "reconnect", label: "立即重连", emphasis: "plain" }] });
  if ((f.link === "idle" || f.link === "connecting") && f.control === "taking")
    return blank("busy", "正在接管…", "Agent 做完当前这一步就会停下",
      { icon: "spinner", text: "等 Agent 做完当前这一步，再把操作交给你。", actions: [{ id: "cancel", label: "取消", emphasis: "plain" }] });
  if (f.link === "idle" || f.link === "connecting") return blank("busy", "正在连接画面", "",
    { icon: "spinner", text: "正在连接画面…", actions: [] });
  return null;
}

function blank(tone: Tone, title: string, detail: string, overlay: OverlayView): PanelView {
  return { tone, title, detail, action: null, overlay, frame: NO_FRAME };
}

// 有画面时：状态条说明谁在操作，外框表达这是谁的浏览器、谁在操作。
function withScreen(f: PanelFacts): PanelView {
  const anonymous = f.targetKind === "browser";
  const where = anonymous && f.targetTitle ? `${f.targetTitle} · ` : "";
  if (f.control === "mine") return {
    tone: "you", title: "你正在操作", detail: `${where}Agent 已暂停，交还后会继续`, action: release, overlay: null,
    frame: { kind: "control", legend: anonymous ? "你正在操作 · 匿名浏览器" : "你正在操作 · 键盘和鼠标会发送到电脑" },
  };
  const frame = anonymous ? { kind: "anonymous", legend: "匿名浏览器 · 不带你的登录" } as const : NO_FRAME;
  if (f.control === "taking") return { tone: "busy", title: "正在接管…", detail: "Agent 做完当前这一步就会停下",
    action: { id: "cancel", label: "取消", emphasis: "plain" }, overlay: null, frame };
  if (f.control === "other") return { tone: "you", title: "另一个窗口正在操作", detail: `${where}Agent 已暂停；你仍然可以观看`,
    action: { ...take, blockedBy: "等对方交还后可接管" }, overlay: null, frame };
  if (f.agentActive) return { tone: "busy", title: "Agent 正在操作", detail: `${where}你可以随时接管`,
    action: take, overlay: null, frame };
  return { tone: "ok", title: "只读观看", detail: `${where}你的点击和按键不会发送到电脑`, action: take, overlay: null, frame };
}

export function present(facts: PanelFacts): PanelView {
  return withoutScreen(facts) ?? withScreen(facts);
}
