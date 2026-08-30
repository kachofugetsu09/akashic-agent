import type { FileUIPart } from "ai";

export type ChatRole = "user" | "assistant";

export interface ThinkingBlock {
  kind: "thinking";
  content: string;
}

export interface ToolBlock {
  kind: "tool";
  callId: string;
  name: string;
  status: "input-available" | "output-available" | "output-error";
  input: unknown;
  output: unknown;
  errorText: string | undefined;
  durationMs?: number;
}

export type AgentBlock = ThinkingBlock | ToolBlock;

export type MessageAttachment = FileUIPart & {
  id: string;
  path?: string;
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
  controlTurnId?: string;
  reply?: {
    messageId: string;
    role: ChatRole;
    preview: string;
  };
}
