import { Cause, Effect, Exit } from "effect";

export type ChatFrame =
  | { type: "sessions.changed"; version: 2 }
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
  | { type: "message.accepted"; request_id: string; session_id: string }
  | { type: "message.steer.result"; request_id: string; staged?: boolean; withdrawn?: boolean }
  | { type: "error"; request_id: string; message: string }
  | { type: "pong"; request_id: string };

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
    case "sessions.changed":
      if (frame.version !== 2) throw new Error("会话目录通知版本无效");
      break;
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
    case "message.accepted":
      requireStrings(frame, ["request_id", "session_id"]);
      break;
    case "message.steer.result":
      requireStrings(frame, ["request_id"]);
      if (frame.staged !== undefined && typeof frame.staged !== "boolean") throw new Error("message.steer.result.staged 格式无效");
      if (frame.withdrawn !== undefined && typeof frame.withdrawn !== "boolean") throw new Error("message.steer.result.withdrawn 格式无效");
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

/** 等待连接后发送一次；Effect 负责超时、取消和竞争结果。 */
export async function sendWhenOpen(
  socket: WebSocket,
  payload: Record<string, unknown>,
  signal?: AbortSignal,
  timeoutMs = 10_000,
): Promise<void> {
  if (signal?.aborted) throw new DOMException("请求已取消", "AbortError");
  const wait = Effect.async<void, Error>((resume) => {
    // timeout 的竞争任务开始前，浏览器可能已经发出了 open 或 close。
    if (socket.readyState !== WebSocket.CONNECTING) {
      resume(socket.readyState === WebSocket.OPEN ? Effect.void : Effect.fail(new Error("聊天连接尚未建立")));
      return;
    }
    const finish = (result: Effect.Effect<void, Error>) => { cleanup(); resume(result); };
    const onOpen = () => finish(Effect.void);
    const onError = () => finish(Effect.fail(new Error("聊天连接失败")));
    const onClose = () => finish(Effect.fail(new Error("聊天连接在发送前关闭")));
    const cleanup = () => {
      socket.removeEventListener("open", onOpen);
      socket.removeEventListener("error", onError);
      socket.removeEventListener("close", onClose);
    };
    socket.addEventListener("open", onOpen);
    socket.addEventListener("error", onError);
    socket.addEventListener("close", onClose);
    return Effect.sync(cleanup);
  }).pipe(Effect.timeoutFail({
    duration: timeoutMs,
    onTimeout: () => {
      if (socket.readyState === WebSocket.CONNECTING) socket.close();
      return new Error("聊天连接超时");
    },
  }));
  const sent = await Effect.runPromiseExit((socket.readyState === WebSocket.OPEN ? Effect.void : wait).pipe(
    Effect.andThen(Effect.try({ try: () => {
      if (socket.readyState !== WebSocket.OPEN) throw new Error("聊天连接未能打开");
      socket.send(JSON.stringify(payload));
    }, catch: (error) => error })),
  ), { signal });
  if (Exit.isFailure(sent)) {
    if (signal?.aborted && Cause.isInterruptedOnly(sent.cause)) throw new DOMException("请求已取消", "AbortError");
    throw Cause.squash(sent.cause);
  }
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
