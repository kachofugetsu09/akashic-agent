import React, { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { FileUIPart } from "ai";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  CircleStop,
  Pencil,
  Plus,
  SendHorizontal,
  Smartphone,
} from "lucide-react";
import {
  Attachment,
  AttachmentHoverCard,
  AttachmentHoverCardContent,
  AttachmentHoverCardTrigger,
  AttachmentInfo,
  AttachmentPreview,
  AttachmentRemove,
  Attachments,
  getAttachmentLabel,
  getMediaCategory,
} from "@/components/ai-elements/attachments";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { MobilePairingDialog } from "./mobile-pairing-dialog";
import { MobileShowcase } from "./mobile-showcase";
import "./styles.css";

type ChatStatus = "idle" | "submitted" | "streaming" | "error";
type Role = "user" | "assistant";

interface SessionRow {
  key: string;
  updated_at?: string;
  message_count?: number;
  first_message_content?: string;
}

interface MessageRow {
  id: number | string;
  role: string;
  content: string;
  timestamp?: string;
  media?: unknown;
  tool_chain?: unknown;
  reasoning_content?: unknown;
  turn_duration_ms?: unknown;
  extra?: Record<string, unknown>;
}

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

type ComposerFile = {
  filename?: string;
  mediaType?: string;
  url?: string;
};

type UploadedFile = {
  filename: string;
  upload_path: string;
  upload_url?: string;
};

export type MessageAttachment = FileUIPart & {
  id: string;
  path?: string;
};

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  attachments?: MessageAttachment[];
  blocks: AgentBlock[];
  streaming?: boolean;
  interrupted?: boolean;
  startedAt?: number;
  durationMs?: number;
}

const LazyChatMessageView = lazy(() =>
  import("./message-view").then(({ ChatMessageView }) => ({ default: ChatMessageView })),
);

type ChatFrame =
  | { type: "session.created"; request_id: string; session_id: string }
  | { type: "turn.started"; session_id: string; turn_id: string; content: string }
  | { type: "react.thinking.delta"; session_id: string; turn_id: string; delta: string }
  | { type: "react.tool.started"; session_id: string; turn_id: string; call_id: string; tool_name: string; arguments: unknown }
  | { type: "react.tool.completed"; session_id: string; turn_id: string; call_id: string; tool_name: string; status: string; result_preview: string }
  | { type: "answer.delta"; session_id: string; turn_id: string; delta: string }
  | { type: "message.final"; session_id: string; turn_id: string; content: string; thinking?: string; media?: string[]; duration_ms?: number; metadata?: Record<string, unknown> }
  | { type: "turn.interrupted"; request_id: string; session_id: string; status: string; message: string }
  | { type: "error"; request_id: string; message: string }
  | { type: "pong"; request_id: string };

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function fetchChatJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(response.ok ? "服务器返回了无效 JSON" : `请求失败: ${response.status}`);
    }
  }
  if (!response.ok) {
    const body = recordValue(payload);
    const detail = typeof body?.detail === "string" ? body.detail : typeof body?.message === "string" ? body.message : "";
    throw new Error(detail || `请求失败: ${response.status}`);
  }
  if (payload === null) throw new Error("服务器返回空响应");
  return payload as T;
}

function responseItems(payload: unknown, endpoint: string): Record<string, unknown>[] {
  const body = recordValue(payload);
  if (!body || !Array.isArray(body.items) || body.items.some((item) => !recordValue(item))) {
    throw new Error(`${endpoint} 返回格式无效`);
  }
  return body.items as Record<string, unknown>[];
}

function sessionRows(payload: unknown): SessionRow[] {
  const items = responseItems(payload, "/api/chat/sessions");
  if (items.some((item) => (
    typeof item.key !== "string"
    || !item.key.trim()
    || (item.first_message_content !== undefined && typeof item.first_message_content !== "string")
  ))) {
    throw new Error("/api/chat/sessions 返回了无效 session 行");
  }
  return items as unknown as SessionRow[];
}

function messageRows(payload: unknown, endpoint: string): MessageRow[] {
  const items = responseItems(payload, endpoint);
  if (items.some((item) => (
    (typeof item.id !== "string" && (typeof item.id !== "number" || !Number.isFinite(item.id)))
    || typeof item.role !== "string"
    || typeof item.content !== "string"
  ))) {
    throw new Error(`${endpoint} 返回了无效 message 行`);
  }
  return items as unknown as MessageRow[];
}

function uploadedFileResponse(payload: unknown): UploadedFile {
  const body = recordValue(payload);
  if (!body || typeof body.filename !== "string" || typeof body.upload_path !== "string" || !body.upload_path) {
    throw new Error("上传接口返回格式无效");
  }
  if (body.upload_url !== undefined && typeof body.upload_url !== "string") {
    throw new Error("上传接口返回了无效 URL");
  }
  return body as unknown as UploadedFile;
}

function App() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState("");
  const [mobilePairingOpen, setMobilePairingOpen] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");
  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const sendRequestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const reportError = useCallback((error: unknown, nextStatus?: ChatStatus): void => {
    if (isAbortError(error)) return;
    console.error("[chat] request failed", error);
    setError(errorMessage(error));
    if (nextStatus) setStatus(nextStatus);
  }, []);

  const loadSessions = useCallback(async () => {
    sessionsRequestRef.current?.abort();
    const controller = new AbortController();
    sessionsRequestRef.current = controller;
    try {
      const payload = await fetchChatJson<unknown>("/api/chat/sessions?page=1&page_size=80", { signal: controller.signal });
      const items = sessionRows(payload);
      setSessions(items.filter((session) => session.first_message_content?.trim()));
    } finally {
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null;
    }
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    messagesRequestRef.current?.abort();
    const controller = new AbortController();
    messagesRequestRef.current = controller;
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;
    try {
      const payload = await fetchChatJson<unknown>(`${endpoint}?page=1&page_size=100&sort_by=seq&sort_order=asc`, { signal: controller.signal });
      setMessages(messageRows(payload, endpoint).filter(isVisibleChatRow).map(rowToMessage));
    } finally {
      if (messagesRequestRef.current === controller) messagesRequestRef.current = null;
    }
  }, []);

  const loadSessionsSafely = useCallback(() => loadSessions().catch((error: unknown) => reportError(error)), [loadSessions, reportError]);
  const loadMessagesSafely = useCallback((sessionId: string) => loadMessages(sessionId).catch((error: unknown) => reportError(error)), [loadMessages, reportError]);

  const connect = useCallback(() => {
    const current = socketRef.current;
    if (current && current.readyState <= WebSocket.OPEN) {
      return current;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return;
      try {
        const frame = parseChatFrame(JSON.parse(String(event.data)));
        handleFrame(frame, {
          activeSessionRef,
          setActiveSessionId,
          setError,
          setMessages,
          setStatus,
          loadSessions: loadSessionsSafely,
        });
      } catch (error) {
        reportError(error, "error");
      }
    };
    socket.onerror = () => {
      if (socketRef.current === socket) reportError(new Error("聊天连接失败"), "error");
    };
    socket.onclose = (event) => {
      if (socketRef.current !== socket) return;
      socketRef.current = null;
      if (event.code !== 1000 || statusRef.current !== "idle") {
        reportError(new Error("聊天连接已关闭"), "error");
      }
    };
    return socket;
  }, [loadSessionsSafely, reportError]);

  useEffect(() => {
    void loadSessionsSafely();
    const socket = connect();
    return () => {
      sessionsRequestRef.current?.abort();
      messagesRequestRef.current?.abort();
      sendRequestRef.current?.abort();
      if (socketRef.current === socket) socketRef.current = null;
      socket.close(1000, "component unmounted");
    };
  }, [connect, loadSessionsSafely]);

  const ensureSession = useCallback(async () => {
    if (activeSessionRef.current) return activeSessionRef.current;
    const sessionId = `web:${crypto.randomUUID().replaceAll("-", "")}`;
    activeSessionRef.current = sessionId;
    setActiveSessionId(sessionId);
    return sessionId;
  }, []);

  const sendMessage = useCallback(async (text: string, files: ComposerFile[]) => {
    const cleanText = text.trim();
    if (!cleanText && files.length === 0) return;
    setError("");
    setStatus("submitted");
    setInput("");
    sendRequestRef.current?.abort();
    const controller = new AbortController();
    sendRequestRef.current = controller;
    const optimisticId = crypto.randomUUID();
    try {
      const sessionId = await ensureSession();
      const media = await uploadFiles(files, controller.signal);
      const attachments = media.map((item) => uploadedFileToAttachment(item));
      setMessages((current) => [
        ...current,
        {
          id: optimisticId,
          role: "user",
          content: cleanText || media.map((item) => item.filename).join("\n"),
          attachments,
          blocks: [],
        },
      ]);
      await sendWhenOpen(connect(), {
        type: "message.send",
        request_id: crypto.randomUUID(),
        session_id: sessionId,
        text: cleanText,
        media: media.map((item) => item.upload_path),
      }, controller.signal);
    } catch (error) {
      if (isAbortError(error)) throw error;
      setMessages((current) => current.filter((message) => message.id !== optimisticId));
      setInput(text);
      reportError(error, "error");
      throw error;
    } finally {
      if (sendRequestRef.current === controller) sendRequestRef.current = null;
    }
  }, [connect, ensureSession, reportError]);

  const stopTurn = useCallback(() => {
    if (!activeSessionId) return;
    void sendWhenOpen(connect(), {
      type: "turn.stop",
      request_id: crypto.randomUUID(),
      session_id: activeSessionId,
    }).then(() => setStatus("idle")).catch((error: unknown) => reportError(error, "error"));
  }, [activeSessionId, connect, reportError]);

  return (
    <main className="chat-shell dark">
      <aside className="chat-sidebar">
        <section className="session-section">
          <div className="session-title-row">
            <div className="session-title">最近</div>
            <button
              className="icon-button"
              type="button"
              aria-label="新聊天"
              onClick={() => {
                activeSessionRef.current = "";
                messagesRequestRef.current?.abort();
                sendRequestRef.current?.abort();
                setActiveSessionId("");
                setMessages([]);
                setStatus("idle");
              }}
            >
              <Pencil size={18} />
            </button>
          </div>
          <div className="session-list">
            {sessions.map((session) => (
              <button
                key={session.key}
                className={`session-button ${activeSessionId === session.key ? "active" : ""}`}
                type="button"
                onClick={() => {
                  setActiveSessionId(session.key);
                  void loadMessagesSafely(session.key);
                }}
              >
                {sessionLabel(session)}
              </button>
            ))}
          </div>
        </section>
        <div className="sidebar-mobile-action">
          <button className="mobile-connect-trigger" type="button" onClick={() => setMobilePairingOpen(true)}>
            <span className="mobile-connect-icon" aria-hidden="true"><Smartphone size={19} /></span>
            <span>连接手机</span>
          </button>
        </div>
      </aside>

      <section className="chat-main">
        <Conversation className="conversation">
          <ConversationContent className={messages.length ? "conversation-content" : "conversation-content empty"}>
            {messages.length === 0 ? (
              <ConversationEmptyState className="home-state">
                <h1>今天有什么计划?</h1>
              </ConversationEmptyState>
            ) : (
              <MessageRendererErrorBoundary>
                <Suspense fallback={<div className="message-row message-loading">正在加载消息渲染器…</div>}>
                  {messages.map((message, index) => (
                    <React.Fragment key={message.id}>
                      {messages[index - 1]?.role === message.role ? <RoleDivider role={message.role} /> : null}
                      <LazyChatMessageView message={message} />
                    </React.Fragment>
                  ))}
                </Suspense>
              </MessageRendererErrorBoundary>
            )}
          </ConversationContent>
          <AutoScroll messages={messages} status={status} />
          <ConversationScrollButton />
        </Conversation>

        <div className={`composer-wrap ${messages.length === 0 ? "home" : ""}`}>
            <PromptInput
              className="composer"
              multiple
              onSubmit={(message) => sendMessage(message.text, message.files)}
            >
              <PromptInputBody>
                <ComposerAttachments />
                <PromptInputTextarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                placeholder="有问题，尽管问"
              />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger className="composer-tool" tooltip="添加文件">
                    <Plus size={20} />
                  </PromptInputActionMenuTrigger>
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments label="上传文件" />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
              </PromptInputTools>
              <PromptInputTools>
                <ComposerSubmit input={input} status={status} onStop={stopTurn} />
              </PromptInputTools>
            </PromptInputFooter>
          </PromptInput>
          {error && <div className="error-line">{error}</div>}
        </div>
      </section>
      <MobilePairingDialog open={mobilePairingOpen} onOpenChange={setMobilePairingOpen} />
    </main>
  );
}

class MessageRendererErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("消息渲染器加载失败", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return <div className="message-row message-renderer-error">消息渲染器加载失败，请刷新页面</div>;
    }
    return this.props.children;
  }
}

function ComposerAttachments() {
  const attachments = usePromptInputAttachments();
  if (attachments.files.length === 0) {
    return null;
  }

  return (
    <Attachments className="composer-attachments" variant="inline">
      {attachments.files.map((attachment) => (
        <AttachmentHoverCard key={attachment.id}>
          <AttachmentHoverCardTrigger asChild>
            <Attachment
              data={attachment}
              onRemove={() => attachments.remove(attachment.id)}
            >
              <div className="attachment-preview-slot">
                <div className="attachment-preview-icon">
                  <AttachmentPreview />
                </div>
                <AttachmentRemove className="attachment-remove-inline" />
              </div>
              <AttachmentInfo />
            </Attachment>
          </AttachmentHoverCardTrigger>
          <AttachmentHoverCardContent>
            <AttachmentHover attachment={attachment} />
          </AttachmentHoverCardContent>
        </AttachmentHoverCard>
      ))}
    </Attachments>
  );
}

function AttachmentHover({ attachment }: { attachment: ReturnType<typeof usePromptInputAttachments>["files"][number] }) {
  const category = getMediaCategory(attachment);
  const label = getAttachmentLabel(attachment);
  return (
    <div className="attachment-hover">
      {category === "image" && attachment.url ? (
        <img alt={label} className="attachment-hover-image" src={attachment.url} />
      ) : (
        <div className="attachment-hover-file">
          <Attachment data={attachment}>
            <AttachmentPreview />
          </Attachment>
        </div>
      )}
      <div className="attachment-hover-title">{label}</div>
      {attachment.mediaType && <div className="attachment-hover-type">{attachment.mediaType}</div>}
    </div>
  );
}

function ComposerSubmit({
  input,
  status,
  onStop,
}: {
  input: string;
  status: ChatStatus;
  onStop: () => void;
}) {
  const attachments = usePromptInputAttachments();
  return (
    <PromptInputSubmit
      className="send-button"
      status={status === "idle" ? undefined : status}
      onStop={onStop}
      disabled={status === "idle" && !input.trim() && attachments.files.length === 0}
    >
      {status === "idle" ? <SendHorizontal size={18} /> : <CircleStop size={18} />}
    </PromptInputSubmit>
  );
}

function RoleDivider({ role }: { role: Role }) {
  return <div aria-hidden="true" className={`role-divider ${role}-divider`} />;
}

function AutoScroll({ messages, status }: { messages: ChatMessage[]; status: ChatStatus }) {
  const { escapedFromLock, isAtBottom, scrollToBottom } = useStickToBottomContext();
  const lastMessageCountRef = useRef(messages.length);
  const scrollKey = messages.map((message) => {
    const lastBlock = message.blocks.at(-1);
    return [
      message.id,
      message.content.length,
      message.blocks.length,
      lastBlock?.kind === "thinking" ? lastBlock.content.length : "",
    ].join(":");
  }).join("|");

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const hasNewUserMessage = messages.length > lastMessageCountRef.current && lastMessage?.role === "user";
    lastMessageCountRef.current = messages.length;

    if (hasNewUserMessage) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
      return;
    }

    if ((status === "streaming" || status === "submitted") && isAtBottom && !escapedFromLock) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: false });
    }
  }, [escapedFromLock, isAtBottom, messages, scrollKey, status, scrollToBottom]);

  return null;
}

function parseChatFrame(value: unknown): ChatFrame {
  const frame = recordValue(value);
  if (!frame || typeof frame.type !== "string") throw new Error("WebSocket 返回了无效消息");
  switch (frame.type) {
    case "session.created":
      requireStrings(frame, ["request_id", "session_id"]);
      break;
    case "turn.started":
      requireStrings(frame, ["session_id", "turn_id", "content"]);
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
      if (frame.media !== undefined && (!Array.isArray(frame.media) || frame.media.some((item) => typeof item !== "string"))) {
        throw new Error("message.final.media 格式无效");
      }
      if (frame.metadata !== undefined && !recordValue(frame.metadata)) throw new Error("message.final.metadata 格式无效");
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

function requireStrings(record: Record<string, unknown>, keys: string[]): void {
  for (const key of keys) {
    if (typeof record[key] !== "string") throw new Error(`WebSocket 消息缺少字符串字段: ${key}`);
  }
}

function handleFrame(
  frame: ChatFrame,
  ctx: {
    activeSessionRef: React.MutableRefObject<string>;
    setActiveSessionId: React.Dispatch<React.SetStateAction<string>>;
    setError: React.Dispatch<React.SetStateAction<string>>;
    setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
    setStatus: React.Dispatch<React.SetStateAction<ChatStatus>>;
    loadSessions: () => Promise<void>;
  },
) {
  if (frame.type === "session.created") {
    ctx.activeSessionRef.current = frame.session_id;
    ctx.setActiveSessionId(frame.session_id);
    return;
  }
  if (frame.type === "error") {
    ctx.setError(frame.message);
    ctx.setStatus("error");
    return;
  }
  if (!("session_id" in frame)) return;
  if (ctx.activeSessionRef.current && frame.session_id !== ctx.activeSessionRef.current) return;

  if (frame.type === "turn.interrupted") {
    ctx.setError(frame.status === "idle" ? frame.message : "");
    ctx.setStatus("idle");
    return;
  }

  if (frame.type === "turn.started") {
    ctx.setStatus("streaming");
    ctx.setMessages((messages) => [
      ...messages,
      {
        id: frame.turn_id,
        role: "assistant",
        content: "",
        blocks: [],
        streaming: true,
        startedAt: Date.now(),
      },
    ]);
    return;
  }
  if (frame.type === "react.thinking.delta") {
    ctx.setStatus("streaming");
    ctx.setMessages((messages) => updateLastAssistant(messages, (message) => {
      const blocks = [...message.blocks];
      const last = blocks[blocks.length - 1];
      if (last?.kind === "thinking") {
        blocks[blocks.length - 1] = { ...last, content: last.content + frame.delta };
      } else {
        blocks.push({ kind: "thinking", content: frame.delta });
      }
      return { ...message, blocks, streaming: true };
    }));
    return;
  }
  if (frame.type === "react.tool.started") {
    ctx.setMessages((messages) => updateLastAssistant(messages, (message) => ({
      ...message,
      blocks: [
        ...message.blocks,
        {
          kind: "tool",
          callId: frame.call_id,
          name: frame.tool_name,
          status: "input-available",
          input: frame.arguments,
          output: undefined,
          errorText: undefined,
        },
      ],
      streaming: true,
    })));
    return;
  }
  if (frame.type === "react.tool.completed") {
    const succeeded = frame.status === "success";
    ctx.setMessages((messages) => updateTool(messages, frame.call_id, {
      status: succeeded ? "output-available" : "output-error",
      output: frame.result_preview,
      errorText: succeeded ? undefined : frame.result_preview,
    }));
    return;
  }
  if (frame.type === "answer.delta") {
    ctx.setMessages((messages) => updateLastAssistant(messages, (message) => ({
      ...message,
      content: message.content + frame.delta,
      streaming: true,
    })));
    return;
  }
  if (frame.type === "message.final") {
    if (frame.metadata?.source === "message_push") {
      ctx.setMessages((messages) => updateLastAssistant(messages, (message) => ({
        ...message,
        content: message.content || frame.content,
        attachments: mergeAttachments(message.attachments, mediaToAttachments(frame.media)),
        blocks: blocksWithFinalThinking(message.blocks, frame.thinking),
        streaming: message.streaming,
      })));
      void ctx.loadSessions();
      return;
    }
    ctx.setStatus("idle");
    ctx.setMessages((messages) => updateLastAssistant(messages, (message) => ({
      ...message,
      content: frame.content || message.content,
      attachments: frame.media?.length
        ? mergeAttachments(message.attachments, mediaToAttachments(frame.media))
        : message.attachments,
      blocks: blocksWithFinalThinking(message.blocks, frame.thinking),
      durationMs: frame.duration_ms ?? (
        message.startedAt ? Date.now() - message.startedAt : message.durationMs
      ),
      streaming: false,
    })));
    void ctx.loadSessions();
  }
}

function updateLastAssistant(
  messages: ChatMessage[],
  updater: (message: ChatMessage) => ChatMessage,
) {
  const next = [...messages];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (next[index].role === "assistant") {
      next[index] = updater(next[index]);
      return next;
    }
  }
  return [...messages, updater({ id: crypto.randomUUID(), role: "assistant", content: "", blocks: [] })];
}

function updateTool(
  messages: ChatMessage[],
  callId: string,
  patch: Pick<ToolBlock, "status" | "output" | "errorText">,
) {
  return updateLastAssistant(messages, (message) => ({
    ...message,
    blocks: message.blocks.map((block) => {
      if (block.kind !== "tool" || block.callId !== callId) return block;
      return { ...block, ...patch };
    }),
  }));
}

function sendWhenOpen(socket: WebSocket, payload: Record<string, unknown>, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("请求已取消", "AbortError"));
  if (socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify(payload));
      return Promise.resolve();
    } catch (error) {
      return Promise.reject(error);
    }
  }
  if (socket.readyState !== WebSocket.CONNECTING) {
    return Promise.reject(new Error("聊天连接尚未建立"));
  }
  return new Promise((resolve, reject) => {
    let settled = false;

    function cleanup(): void {
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

async function uploadFiles(files: ComposerFile[], signal: AbortSignal) {
  const result: UploadedFile[] = [];
  for (const file of files) {
    if (!file.url) throw new Error(`附件 ${file.filename || "未命名"} 缺少内容 URL`);
    const sourceResponse = await fetch(file.url, { signal });
    if (!sourceResponse.ok) throw new Error(`读取附件失败: ${sourceResponse.status}`);
    const blob = await sourceResponse.blob();
    const filename = file.filename || "upload.bin";
    const payload = await fetchChatJson<unknown>(`/api/chat/uploads?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      body: blob,
      signal,
    });
    result.push(uploadedFileResponse(payload));
  }
  return result;
}

function rowToMessage(row: MessageRow): ChatMessage {
  const role: Role = row.role === "user" ? "user" : "assistant";
  return {
    id: String(row.id),
    role,
    content: row.content,
    attachments: mediaToAttachments(row.media),
    blocks: role === "assistant" ? rowBlocks(row) : [],
    durationMs: numberValue(row.turn_duration_ms),
  };
}

function isVisibleChatRow(row: MessageRow) {
  return !(row.role === "user" && row.content.startsWith("[后台任务完成]"));
}

function rowBlocks(row: MessageRow): AgentBlock[] {
  const blocks = toolChainToBlocks(row.tool_chain);
  const finalThinking = stringValue(row.reasoning_content);
  if (finalThinking) {
    blocks.push({ kind: "thinking", content: finalThinking });
  }
  return blocks;
}

function toolChainToBlocks(toolChain: unknown): AgentBlock[] {
  if (!Array.isArray(toolChain)) return [];
  const blocks: AgentBlock[] = [];
  toolChain.forEach((item, groupIndex) => {
    const group = recordValue(item);
    if (!group) return;
    const thinking = stringValue(group.reasoning_content) || stringValue(group.text);
    if (thinking) {
      blocks.push({ kind: "thinking", content: thinking });
    }
    const calls = Array.isArray(group.calls) ? group.calls : [];
    calls.forEach((call, callIndex) => {
      const block = toolCallToBlock(call, groupIndex, callIndex);
      if (block) blocks.push(block);
    });
  });
  return blocks;
}

function toolCallToBlock(call: unknown, groupIndex: number, callIndex: number): ToolBlock | null {
  const item = recordValue(call);
  if (!item) return null;
  const name = stringValue(item.name);
  if (!name) return null;
  const rawStatus = stringValue(item.status);
  const status = !rawStatus || rawStatus === "success" ? "output-available" : "output-error";
  return {
    kind: "tool",
    callId: stringValue(item.call_id) || `${groupIndex}-${callIndex}-${name}`,
    name,
    status,
    input: item.final_arguments ?? item.arguments,
    output: item.result,
    errorText: status === "output-error" ? stringValue(item.result) : undefined,
  };
}

function blocksWithFinalThinking(blocks: AgentBlock[], thinking: string | undefined): AgentBlock[] {
  const text = thinking?.trim();
  if (!text || blocks.some((block) => block.kind === "thinking")) return blocks;
  return [{ kind: "thinking", content: text } satisfies ThinkingBlock, ...blocks];
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function numberValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function uploadedFileToAttachment(file: UploadedFile): MessageAttachment {
  const filename = file.filename || filenameFromPath(file.upload_path);
  return {
    id: file.upload_path,
    type: "file",
    filename,
    mediaType: guessMediaType(filename),
    url: file.upload_url || mediaUrl(file.upload_path),
    path: file.upload_path,
  };
}

function mergeAttachments(
  current: MessageAttachment[] | undefined,
  incoming: MessageAttachment[],
) {
  if (!incoming.length) return current;
  const merged = [...(current ?? [])];
  const seen = new Set(merged.map((item) => item.path || item.id));
  incoming.forEach((item) => {
    const key = item.path || item.id;
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(item);
  });
  return merged;
}

function mediaToAttachments(media: unknown): MessageAttachment[] {
  if (!Array.isArray(media)) return [];
  return media
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .map((path, index) => {
      const filename = filenameFromPath(path);
      return {
        id: `${path}:${index}`,
        type: "file",
        filename,
        mediaType: guessMediaType(filename),
        url: mediaUrl(path),
        path,
      };
    });
}

function mediaUrl(path: string) {
  return `/api/chat/media?path=${encodeURIComponent(path)}`;
}

function filenameFromPath(path: string) {
  return path.split(/[\\/]/).pop() || "附件";
}

function guessMediaType(filename: string) {
  const suffix = filename.split(".").pop()?.toLowerCase() || "";
  if (["apng", "avif", "gif", "jpg", "jpeg", "png", "svg", "webp"].includes(suffix)) {
    return `image/${suffix === "jpg" ? "jpeg" : suffix}`;
  }
  if (["mp4", "webm", "mov"].includes(suffix)) return `video/${suffix}`;
  if (["mp3", "ogg", "wav", "m4a"].includes(suffix)) return `audio/${suffix}`;
  if (suffix === "txt") return "text/plain";
  if (suffix === "pdf") return "application/pdf";
  return "application/octet-stream";
}

function sessionLabel(session: SessionRow) {
  const title = session.first_message_content?.trim() || "未命名对话";
  return title.length > 28 ? `${title.slice(0, 28)}...` : title;
}

const isMobileShowcase = new URLSearchParams(window.location.search).get("preview") === "mobile";

createRoot(document.getElementById("root")!).render(
  <TooltipProvider>
    {isMobileShowcase ? <MobileShowcase /> : <App />}
  </TooltipProvider>,
);
