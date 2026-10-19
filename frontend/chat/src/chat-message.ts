import type { FileUIPart } from "ai";
import type { SentAttachmentItem } from "./attachment-card";

export type ChatRole = "user" | "assistant";

export interface ThinkingBlock {
  kind: "thinking";
  content: string;
}

export interface ToolBlock {
  kind: "tool";
  callId: string;
  name: string;
  /** 注册该工具的插件；只有它能接手结果展示，归档里的调用没有。 */
  owner?: string;
  status: "input-available" | "output-available" | "output-error";
  input: unknown;
  output: unknown;
  errorText: string | undefined;
  durationMs?: number;
  /** 结果附带的图片（截图、读图）；与文字结果一起在调用卡片里展示。 */
  attachments?: SentAttachmentItem[];
}

export type AgentBlock = ThinkingBlock | ToolBlock;

export type MessageAttachment = FileUIPart & {
  id: string;
  path?: string;
  sizeBytes?: number;
};

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  attachments?: MessageAttachment[];
  blocks: AgentBlock[];
  streaming?: boolean;
  interrupted?: boolean;
  startedAt?: number;
  durationMs?: number;
  createdAt?: string;
  canonical?: boolean;
  /** 本地发送中的消息：sending 等待服务端确认，retrying 正在换连接重发；确认后移除。 */
  delivery?: "sending" | "retrying";
  controlTurnId?: string;
  reply?: {
    messageId: string;
    role: ChatRole;
    preview: string;
  };
}
