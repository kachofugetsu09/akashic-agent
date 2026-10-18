import type { ComponentType, ReactNode } from "react";

/** 聊天宿主提供的插槽；插件只在目录声明过的插槽里挂载。 */
export type PluginUiSlotName =
  | "turn.before_reasoning"
  | "turn.before_tool"
  | "turn.after_answer"
  | "message.attachment"
  | "drawer.panel"
  | "chat.notice"
  | "header.end"
  | "composer.suggest"
  | "dashboard.main";

/** 宿主挂载插件元素的插槽；composer.suggest 是数据源，不挂载。 */
export type PluginUiMountSlotName = Exclude<PluginUiSlotName, "dashboard.main" | "composer.suggest">;

export type WorkbenchIntentStatus = "handled" | "unhandled" | "invalid" | "failed";

export interface PluginUiQueryOptions {
  cache?: "none" | "memory" | "immutable";
}

export interface PluginUiContext {
  slot: PluginUiSlotName;
  sessionId?: string;
  messageId?: string;
  turnId?: string;
  block?: unknown;
  /** 挂载内的事实失效提示；不改变挂载身份。 */
  onInvalidate(callback: () => void): () => void;
  /** 异步展示失败时交还宿主降级内容；不改变消息。 */
  reportError(message: string): void;
  capabilities: {
    queryCacheModes?: readonly ("none" | "memory" | "immutable")[];
  };
  /**
   * 本插件后端 follow 推送的当前会话状态：先回放最近一帧，之后每帧回调；
   * null 表示旧状态已失效（断线、插件换代），插件应丢弃它。返回取消订阅。
   */
  subscribe(callback: (value: Record<string, unknown> | null) => void): () => void;
  /** 只查询本插件注册的方法；宿主不替插件选择其他插件。 */
  query(method: string, payload?: Record<string, unknown>, options?: PluginUiQueryOptions): Promise<Record<string, unknown>>;
  /**
   * 在右侧工具区打开：只在点击或键盘激活的处理函数里调用；没有接受者时得到 unhandled，插件回到自己的原有行为。
   * accepts 的结果会随宿主广播变化，变化时经 onInvalidate 通知，插件在回调里重新判断是否显示入口。
   */
  workbench: {
    accepts(name: string): boolean;
    open(name: string, payload: unknown): Promise<WorkbenchIntentStatus>;
  };
}

export interface PluginUiRenderer {
  accepts?(block: unknown): boolean;
  prefetch?(context: PluginUiContext): Promise<void>;
  mount(host: HTMLElement, context: PluginUiContext): void | (() => void);
}

/** 输入框建议的一项：选中后写入 `${prefix}${token} `；发出的消息里同样写法显示为提及。 */
export interface ComposerSuggestion {
  token: string;
  label: string;
  description: string;
  /** 有值时照样列出但不能选，显示这个原因。 */
  unavailable?: string;
  /** 同名项并存时用来区分的来源。 */
  badge?: string;
}

/** 输入框建议的数据源：触发、菜单、键盘、IME、无障碍与气泡里的提及显示都由宿主负责。 */
export interface ComposerSuggestProvider {
  /** 菜单分组标题。 */
  label: string;
  /** 行首或空白后输入这些符号时打开菜单。 */
  sigils: readonly ("/" | "@")[];
  /** 写入草稿、识别提及用的前缀；与插件后端识别的写法一致。 */
  prefix: "/" | "@";
  /** 没有任何条目时的说明。 */
  empty: string;
  /** 每次打开菜单重新读取；只经 query 访问本插件。 */
  load(context: Pick<PluginUiContext, "query">): Promise<readonly ComposerSuggestion[]>;
}

/** 插件界面模块的默认导出。 */
export interface PluginUiDefinition {
  slots: Partial<Record<PluginUiMountSlotName, PluginUiRenderer>> & { "composer.suggest"?: ComposerSuggestProvider };
  dashboard?: PluginUiRenderer;
}

/** 在插槽元素里挂一个共享 React 的根；返回值可直接作为 mount() 的清理函数。 */
export function render(host: HTMLElement, node: ReactNode): () => void;
/** 输入框上方的一行状态提示。 */
export const Notice: ComponentType<{ children: ReactNode }>;
/**
 * 顶栏右侧的标记：图标加短名，alert 时变色；点开显示 children 面板，点外部或 Esc 收起。
 * 面板里的 code 显示为整块路径，带 data-tone="error" 的元素显示为错误色。
 */
export const HeaderMarker: ComponentType<{
  icon: ReactNode;
  label: string;
  alert?: boolean;
  ariaLabel: string;
  panelLabel: string;
  children: ReactNode;
}>;
/** 标记面板底部的一排文字按钮。 */
export const HeaderMarkerActions: ComponentType<{ children: ReactNode }>;
/** 过长时滚动显示的单行文字。 */
export const MarqueeText: ComponentType<{ children: ReactNode; still?: boolean }>;
export function copyText(text: string): Promise<void>;
