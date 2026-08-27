import type { ChatMessage, ToolBlock } from "./chat-message";
import type { ChatStatus } from "./web-chat-status";
import { blocksWithFinalThinking, mediaToAttachments, mergeAttachments } from "./web-chat-message-data.ts";
import type { WebTurnTraceKind } from "./web-turn-trace";

export type ChatFrame =
  | { type: "session.created"; request_id: string; session_id: string }
  | { type: "turn.started"; session_id: string; turn_id: string; control_turn_id: string; client_message_id: string; content: string }
  | { type: "react.thinking.delta"; session_id: string; turn_id: string; delta: string }
  | { type: "react.tool.started"; session_id: string; turn_id: string; call_id: string; tool_name: string; arguments: unknown }
  | { type: "react.tool.completed"; session_id: string; turn_id: string; call_id: string; tool_name: string; status: string; result_preview: string }
  | { type: "answer.delta"; session_id: string; turn_id: string; delta: string }
  | {
    type: "message.final";
    session_id: string;
    turn_id: string;
    content: string;
    thinking?: string;
    media?: ArtifactDescriptor[];
    duration_ms?: number;
    metadata?: Record<string, unknown>;
    reply_to?: string;
    session_message_id?: string;
    control_turn_id?: string;
    execution_attempt_id?: string;
    terminal_status?: "completed" | "failed" | "interrupted" | "cancelled";
  }
  | { type: "turn.output.completed"; session_id: string; turn_id: string; client_message_id?: string }
  | { type: "turn.interrupted"; request_id: string; session_id: string; status: string; message: string }
  | { type: "error"; request_id: string; message: string }
  | { type: "pong"; request_id: string };

export interface WebChatFrameContext {
  activeSessionId: () => string;
  activateSession: (sessionId: string) => void;
  setError: (message: string) => void;
  setMessages: (updater: (messages: ChatMessage[]) => ChatMessage[]) => void;
  getStatus: () => ChatStatus;
  setStatus: (status: ChatStatus) => void;
  getActiveTurnId: () => string | null;
  setActiveTurnId: (turnId: string | null) => void;
  loadSessions: () => Promise<void>;
  loadMessages: (sessionId: string) => Promise<void>;
}

function parseDurationMs(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string") return null;

  const normalized = Number(value.trim());
  if (!Number.isFinite(normalized)) return null;
  return normalized;
}

export function parseChatFrame(value: unknown): ChatFrame {
  const frame = recordValue(value);
  if (!frame || typeof frame.type !== "string") throw new Error("WebSocket 返回了无效消息");
  switch (frame.type) {
    case "session.created":
      requireStrings(frame, ["request_id", "session_id"]);
      break;
    case "turn.started":
      requireStrings(frame, ["session_id", "turn_id", "control_turn_id", "client_message_id", "content"]);
      break;
    case "react.thinking.delta":
      requireStrings(frame, ["session_id", "turn_id", "delta"]);
      break;
    case "react.tool.started":
      requireStrings(frame, ["session_id", "turn_id", "call_id", "tool_name"]);
      break;
    case "react.tool.completed":
      requireStrings(frame, ["session_id", "turn_id", "call_id", "tool_name", "status", "result_preview"]);
      break;
    case "answer.delta":
      requireStrings(frame, ["session_id", "turn_id", "delta"]);
      break;
    case "message.final":
      requireStrings(frame, ["session_id", "turn_id", "content"]);
      if (frame.thinking !== undefined && typeof frame.thinking !== "string") throw new Error("message.final.thinking 格式无效");
      if (frame.media !== undefined && (!Array.isArray(frame.media) || frame.media.some((item) => !artifactDescriptor(item)))) {
        throw new Error("message.final.media 格式无效");
      }
      if (frame.duration_ms != null) {
        const durationMs = parseDurationMs(frame.duration_ms);
        if (durationMs === null) {
          console.debug(
            "[chat-transport] message.final duration invalid",
            {
              session_id: frame.session_id,
              turn_id: frame.turn_id,
              duration_ms: frame.duration_ms,
            },
          );
          throw new Error("message.final.duration_ms 格式无效");
        }
        frame.duration_ms = durationMs;
      }
      if (frame.metadata !== undefined && !recordValue(frame.metadata)) throw new Error("message.final.metadata 格式无效");
      if (
        frame.terminal_status !== undefined
        && !["completed", "failed", "interrupted", "cancelled"].includes(frame.terminal_status as string)
      ) {
        throw new Error("message.final.terminal_status 格式无效");
      }
      if (frame.execution_attempt_id !== undefined || frame.control_turn_id !== undefined) {
        requireStrings(frame, ["execution_attempt_id", "control_turn_id"]);
        if (frame.turn_id !== frame.execution_attempt_id) {
          throw new Error("message.final.turn_id 必须等于 execution_attempt_id");
        }
      }
      break;
    case "turn.output.completed":
      requireStrings(frame, ["session_id", "turn_id"]);
      if (frame.client_message_id !== undefined && typeof frame.client_message_id !== "string") {
        throw new Error("turn.output.completed.client_message_id 格式无效");
      }
      break;
    case "turn.interrupted":
      requireStrings(frame, ["request_id", "session_id", "status", "message"]);
      break;
    case "error":
      requireStrings(frame, ["request_id", "message"]);
      break;
    case "pong":
      requireStrings(frame, ["request_id"]);
      break;
    default:
      throw new Error(`WebSocket 返回了未知消息类型: ${frame.type}`);
  }
  return frame as unknown as ChatFrame;
}

type ArtifactDescriptor = {
  artifact_id: string;
  kind: "file" | "image";
  filename: string | null;
  media_type: string | null;
  size_bytes: number;
  sha256: string;
  url: string;
};

function artifactDescriptor(value: unknown): value is ArtifactDescriptor {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  return typeof item.artifact_id === "string"
    && item.artifact_id.length > 0
    && (item.kind === "file" || item.kind === "image")
    && (item.filename === null || typeof item.filename === "string")
    && (item.media_type === null || typeof item.media_type === "string")
    && typeof item.size_bytes === "number"
    && Number.isInteger(item.size_bytes)
    && item.size_bytes >= 0
    && typeof item.sha256 === "string"
    && /^[0-9a-f]{64}$/.test(item.sha256)
    && typeof item.url === "string"
    && item.url.startsWith("/api/chat/artifacts/");
}

export function traceKindForChatFrame(frame: ChatFrame): WebTurnTraceKind | undefined {
  if (frame.type === "react.thinking.delta" && frame.delta !== "") return "thinking";
  if (frame.type === "answer.delta" && frame.delta !== "") return "answer";
  if (frame.type === "message.final") return "terminal";
  return undefined;
}

export function applyChatFrame(frame: ChatFrame, context: WebChatFrameContext): void {
  console.debug("[chat-transport] apply frame", {
    type: frame.type,
    session_id: "session_id" in frame ? frame.session_id : undefined,
    turn_id: "turn_id" in frame ? frame.turn_id : undefined,
  });
  if (frame.type === "session.created") {
    context.activateSession(frame.session_id);
    return;
  }
  if (frame.type === "error") {
    console.debug("[chat-transport] error frame", frame.message);
    context.setError(frame.message);
    context.setStatus("error");
    return;
  }
  if (!("session_id" in frame)) return;
  if (frame.session_id !== context.activeSessionId()) {
    console.debug("[chat-transport] skip frame for inactive session", {
      type: frame.type,
      frameSessionId: frame.session_id,
      activeSessionId: context.activeSessionId(),
    });
    return;
  }

  if (frame.type === "turn.interrupted") {
    console.debug("[chat-transport] turn.interrupted", {
      status: frame.status,
      message: frame.message,
    });
    context.setError(frame.status === "idle" ? frame.message : "");
    context.setStatus("idle");
    context.setActiveTurnId(null);
    return;
  }
  if (frame.type === "turn.started") {
    console.debug("[chat-transport] turn.started", { turn_id: frame.turn_id });
    context.setStatus("streaming");
    context.setActiveTurnId(frame.turn_id);
    context.setMessages((messages) => {
      const next = [...messages];
      if (
        frame.content !== ""
        && !next.some((message) => message.id === frame.client_message_id)
      ) {
        next.push({
          id: frame.client_message_id,
          role: "user",
          content: frame.content,
          blocks: [],
          createdAt: new Date().toISOString(),
          canonical: false,
        });
      }
      if (!next.some((message) => message.id === frame.turn_id)) {
        next.push({
          id: frame.turn_id,
          role: "assistant",
          content: "",
          blocks: [],
          streaming: true,
          startedAt: Date.now(),
        });
      }
      return next;
    });
    return;
  }
  if (frame.type === "react.thinking.delta") {
    context.setStatus("streaming");
    context.setMessages((messages) => updateAssistantById(messages, frame.turn_id, (message) => {
      const blocks = [...message.blocks];
      const last = blocks.at(-1);
      if (last?.kind === "thinking") blocks[blocks.length - 1] = { ...last, content: last.content + frame.delta };
      else blocks.push({ kind: "thinking", content: frame.delta });
      return { ...message, blocks, streaming: true };
    }));
    return;
  }
  if (frame.type === "react.tool.started") {
    context.setMessages((messages) => updateAssistantById(messages, frame.turn_id, (message) => ({
      ...message,
      blocks: [...message.blocks, {
        kind: "tool",
        callId: frame.call_id,
        name: frame.tool_name,
        status: "input-available",
        input: frame.arguments,
        output: undefined,
        errorText: undefined,
      }],
      streaming: true,
    })));
    return;
  }
  if (frame.type === "react.tool.completed") {
    const succeeded = frame.status === "success";
    context.setMessages((messages) => updateTool(messages, frame.turn_id, frame.call_id, {
      status: succeeded ? "output-available" : "output-error",
      output: frame.result_preview,
      errorText: succeeded ? undefined : frame.result_preview,
    }));
    return;
  }
  if (frame.type === "answer.delta") {
    console.debug("[chat-transport] answer.delta", { turn_id: frame.turn_id });
    context.setMessages((messages) => updateAssistantById(messages, frame.turn_id, (message) => ({
      ...message,
      content: message.content + frame.delta,
      streaming: true,
    })));
    return;
  }
  if (frame.type === "turn.output.completed") {
    // 只有属于当前 active turn 且仍处于生成中才进入 finalizing；
    // 迟到/跨 turn 的 completion 直接忽略，避免污染下一轮或把 idle 改回 finalizing。
    const current = context.getStatus();
    if (
      frame.turn_id === context.getActiveTurnId() &&
      (current === "streaming" || current === "submitted")
    ) {
      context.setStatus("finalizing");
    }
    return;
  }
  if (frame.type !== "message.final") return;

  if (frame.metadata?.source === "message_push") {
    if (!frame.session_message_id) {
      throw new Error("message_push final 缺少 canonical session_message_id");
    }
    console.debug("[chat-transport] message_push final", {
      session_id: frame.session_id,
      message_id: frame.session_message_id,
    });
    void context.loadMessages(frame.session_id);
    void context.loadSessions();
    return;
  }
  const isActiveTerminal = frame.turn_id === context.getActiveTurnId();
  const isRecoveredTerminal = context.getActiveTurnId() === null;
  const failed = frame.terminal_status === "failed";
  if (isActiveTerminal) {
    context.setStatus(failed ? "error" : "idle");
    context.setActiveTurnId(null);
  }
  if (failed && (isActiveTerminal || isRecoveredTerminal)) {
    context.setError(frame.content);
    context.setStatus("error");
  }
  context.setMessages((messages) => updateAssistantById(messages, frame.turn_id, (message) => ({
    ...message,
    content: frame.content || message.content,
    attachments: frame.media?.length
      ? mergeAttachments(message.attachments, mediaToAttachments(frame.media))
      : message.attachments,
    blocks: blocksWithFinalThinking(message.blocks, frame.thinking),
    durationMs: frame.duration_ms ?? (message.startedAt ? Date.now() - message.startedAt : message.durationMs),
    streaming: false,
  })));
  void context.loadMessages(frame.session_id);
  void context.loadSessions();
}

export function sendWhenOpen(
  socket: WebSocket,
  payload: Record<string, unknown>,
  signal?: AbortSignal,
  timeoutMs = 10_000,
): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("请求已取消", "AbortError"));
  if (socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify(payload));
      return Promise.resolve();
    } catch (error) {
      return Promise.reject(error);
    }
  }
  if (socket.readyState !== WebSocket.CONNECTING) return Promise.reject(new Error("聊天连接尚未建立"));

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = globalThis.setTimeout(() => {
      fail(new Error("聊天连接超时"));
      if (socket.readyState === WebSocket.CONNECTING) socket.close();
    }, timeoutMs);

    function cleanup(): void {
      globalThis.clearTimeout(timeout);
      socket.removeEventListener("open", onOpen);
      socket.removeEventListener("error", onError);
      socket.removeEventListener("close", onClose);
      signal?.removeEventListener("abort", onAbort);
    }

    function fail(error: Error): void {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    }

    function onOpen(): void {
      if (settled) return;
      try {
        if (socket.readyState !== WebSocket.OPEN) throw new Error("聊天连接未能打开");
        socket.send(JSON.stringify(payload));
        settled = true;
        cleanup();
        resolve();
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)));
      }
    }

    function onError(): void {
      fail(new Error("聊天连接失败"));
    }

    function onClose(): void {
      fail(new Error("聊天连接在发送前关闭"));
    }

    function onAbort(): void {
      fail(new DOMException("请求已取消", "AbortError"));
    }

    socket.addEventListener("open", onOpen, { once: true });
    socket.addEventListener("error", onError, { once: true });
    socket.addEventListener("close", onClose, { once: true });
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

function updateAssistantById(
  messages: ChatMessage[],
  messageId: string,
  updater: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const index = messages.findIndex((message) => message.role === "assistant" && message.id === messageId);
  if (index < 0) {
    return [...messages, updater({ id: messageId, role: "assistant", content: "", blocks: [] })];
  }
  const next = [...messages];
  next[index] = updater(next[index]);
  return next;
}

function updateTool(
  messages: ChatMessage[],
  messageId: string,
  callId: string,
  patch: Pick<ToolBlock, "status" | "output" | "errorText">,
): ChatMessage[] {
  return updateAssistantById(messages, messageId, (message) => ({
    ...message,
    blocks: message.blocks.map((block) => block.kind === "tool" && block.callId === callId
      ? { ...block, ...patch }
      : block),
  }));
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function requireStrings(record: Record<string, unknown>, keys: string[]): void {
  for (const key of keys) {
    if (typeof record[key] !== "string") throw new Error(`WebSocket 消息缺少字符串字段: ${key}`);
  }
}
