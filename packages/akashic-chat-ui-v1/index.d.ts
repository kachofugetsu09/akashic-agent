import type { ComponentType, ReactNode } from "react";

/** 聊天宿主提供的插槽；插件只在目录声明过的插槽里挂载。 */
export type PluginUiSlotName =
  | "turn.before_reasoning"
  | "turn.before_tool"
  | "turn.after_answer"
  | "message.attachment"
  | "drawer.panel"
  | "chat.notice"
  | "dashboard.main";

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

/** 插件界面模块的默认导出。 */
export interface PluginUiDefinition {
  slots: Partial<Record<Exclude<PluginUiSlotName, "dashboard.main">, PluginUiRenderer>>;
  dashboard?: PluginUiRenderer;
}

/** 在插槽元素里挂一个共享 React 的根；返回值可直接作为 mount() 的清理函数。 */
export function render(host: HTMLElement, node: ReactNode): () => void;
/** 输入框上方的一行状态提示。 */
export const Notice: ComponentType<{ children: ReactNode }>;
/** 过长时滚动显示的单行文字。 */
export const MarqueeText: ComponentType<{ children: ReactNode; still?: boolean }>;
export function copyText(text: string): Promise<void>;
