import type { WebEntry, WebUiDisposer } from "@akashic/web-ui-v1";

/**
 * 对话右侧工具区的一个标签（conversation.tools.v2）。render 收到的第三个参数是
 * conversation-ui 给这个标签的父视图；标签拥有自己的内容，父级拥有展开、当前标签、宽度与窄屏布局。
 */
export interface ConversationTab extends WebEntry {
  label: string;
}

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
}
