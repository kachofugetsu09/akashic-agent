import type { WebEntry, WebUiDisposer } from "@akashic/web-ui-v1";

/**
 * 对话右侧工具区的一个标签（conversation.tools.v2）。render 收到的第三个参数是
 * conversation-ui 给这个标签的父视图；标签拥有自己的内容，父级拥有展开、当前标签、宽度与窄屏布局。
 */
export interface ConversationTab extends WebEntry {
  label: string;
  /** 16px 线性 SVG 字符串，启动器菜单使用。 */
  icon?: string;
  /**
   * 默认首次被选中才 render；需要在未打开时用 requestAttention 提示活动的标签设为 true，挂载即渲染。
   * 未选中时标签应按 onActiveChange 暂停昂贵工作。
   */
  background?: boolean;
  /**
   * 能处理的意图名，形如 "workspace.changes.open.v1"。payload 结构由意图的领域 owner 在自己的合同里定义；
   * 宿主按标签顺序交给第一个接受者，不广播，也不认识意图含义。
   */
  accepts?: readonly string[];
  /** 可以多开：每次打开都是新实例（如终端），各自 render 一次、各占一个标签。默认单开，已开时只切过去。 */
  multiple?: boolean;
  /** 新标签页里工具名后面的一句说明，≤ 40 字符。 */
  hint?: string;
  /**
   * 切到（或打开）这个工具并把焦点移进去的快捷键："Ctrl+`" 或 "Mod+Shift+字母"（Mod 在 macOS 是 ⌘）。
   * 浏览器、对话页与宿主占用的组合（Mod+Shift+A/B/C/I/J/N/O/P/Q/R/T/V/W/Z）不能用，工具之间不能重复；
   * 标签自己处理过的按键（preventDefault）、按住不放的自动重复都不触发。宿主另占 Mod+J（开关工具区）与 Mod+\（上下分屏）。
   */
  shortcut?: string;
}

/** 来自用户点击的意图；宿主已校验名字、大小（payload JSON ≤ 16 KiB）与会话是当前会话。 */
export interface ConversationIntent {
  readonly id: string;
  readonly name: string;
  readonly session: string;
  readonly payload: unknown;
}

export type ConversationIntentStatus = "handled" | "invalid" | "failed";

/** 引用回输入框；session 是发起时记下的会话，text ≤ 8192 字符，source ≤ 200 字符。 */
export interface ConversationQuote {
  session: string;
  text: string;
  source: string;
}

export type ConversationQuoteStatus = "inserted" | "stale" | "rejected";

/**
 * 父视图只报告父级拥有的可见事实与聊天报告的当前会话，不授予子标签写父状态的能力。
 * session 是聊天 iframe 报告的当前会话 ID，新建对话尚未发送时为 null；父级只转交，不保存、不推导。
 */
export interface ConversationTabView {
  readonly active: boolean;
  onActiveChange(listener: (active: boolean) => void): WebUiDisposer;
  readonly session: string | null;
  onSessionChange(listener: (session: string | null) => void): WebUiDisposer;
  /** 新的、未处理的 noticeId 只亮徽标，不打开工具区；重放同一 ID 无效。 */
  requestAttention(noticeId: string): void;
  /** 宿主先展开并激活本标签再投递；10 秒内没有结果按 failed 回执。 */
  onIntent(listener: (intent: ConversationIntent) => ConversationIntentStatus | Promise<ConversationIntentStatus>): WebUiDisposer;
  /** 只在标签自己的点击或键盘处理函数里调用；会话已切走返回 stale，不自动发送。 */
  quote(request: ConversationQuote): ConversationQuoteStatus;
  /**
   * 多开工具的实例 ID（宿主生成，随会话布局恢复）；单开工具为 null。实例属于创建它的会话，session 不再变化，
   * 换会话时宿主卸下实例（调用 disposer），回到该会话再 render 一次。
   */
  readonly instance: string | null;
  /**
   * 只对多开工具的实例（instance 非 null）生效：用户关闭了这个实例的标签（先于 disposer 调用）；会话切换引起的卸载不算关闭。
   * 单开工具关闭标签后仍保持挂载、不会收到这个通知，按 onActiveChange 暂停即可。
   */
  onClose(listener: () => void): WebUiDisposer;
  /**
   * 本会话里这个工具当前打开的全部实例 ID（含自己，含还没渲染过的）；单开工具为空。
   * 多开工具据此判断自己记下的“实例 → 后端资源”对应里哪些已经过期（标签已关）。
   */
  readonly instances: readonly string[];
  /** 新开这个工具的一个实例，放在本标签之后并聚焦；只用于多开工具（如终端里新开终端的快捷键）。 */
  openNew(): void;
  /** 切到本组里相邻的标签（delta 为 1 或 -1）并聚焦；本组只有一个标签时返回 false，什么也不做。 */
  cycle(delta: 1 | -1): boolean;
  /** 标签标题，≤ 80 字符，null 恢复为 label。 */
  setTitle(title: string | null): void;
  /** 标签自报的状态：busy 在标签上亮琥珀色点；detail（≤ 24 字符）只出现在提示与新标签页里，不占标签宽度。 */
  setIndicator(indicator: ConversationTabIndicator | null): void;
}

export interface ConversationTabIndicator {
  busy?: boolean;
  detail?: string;
}
