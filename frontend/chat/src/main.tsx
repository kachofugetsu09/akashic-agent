import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { FileUIPart } from "ai";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  ChevronDown,
  CircleStop,
  Pencil,
  Plus,
  SendHorizontal,
  Wrench,
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
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
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
import {
  Reasoning,
  ReasoningTrigger,
  useReasoning,
} from "@/components/ai-elements/reasoning";
import { CollapsibleContent } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { TooltipProvider } from "@/components/ui/tooltip";
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

interface ThinkingBlock {
  kind: "thinking";
  content: string;
}

interface ToolBlock {
  kind: "tool";
  callId: string;
  name: string;
  status: "input-available" | "output-available" | "output-error";
  input: unknown;
  output: unknown;
  errorText: string | undefined;
}

type AgentBlock = ThinkingBlock | ToolBlock;

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

type MessageAttachment = FileUIPart & {
  id: string;
  path?: string;
};

interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  attachments?: MessageAttachment[];
  blocks: AgentBlock[];
  streaming?: boolean;
  startedAt?: number;
  durationMs?: number;
}

type ChatFrame =
  | { type: "session.created"; request_id: string; session_id: string }
  | { type: "turn.started"; session_id: string; turn_id: string; content: string }
  | { type: "react.thinking.delta"; session_id: string; turn_id: string; delta: string }
  | { type: "react.tool.started"; session_id: string; turn_id: string; call_id: string; tool_name: string; arguments: unknown }
  | { type: "react.tool.completed"; session_id: string; turn_id: string; call_id: string; tool_name: string; status: string; result_preview: string }
  | { type: "answer.delta"; session_id: string; turn_id: string; delta: string }
  | { type: "message.final"; session_id: string; turn_id: string; content: string; thinking?: string; media?: string[]; duration_ms?: number; metadata?: Record<string, unknown> }
  | { type: "turn.interrupted"; session_id: string; status: string; message: string }
  | { type: "error"; request_id: string; message: string }
  | { type: "pong"; request_id: string };

function App() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const loadSessions = useCallback(async () => {
    const response = await fetch("/api/chat/sessions?page=1&page_size=80");
    const data = await response.json() as { items?: SessionRow[] };
    setSessions((data.items ?? []).filter((session) => session.first_message_content?.trim()));
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    const response = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/messages?page=1&page_size=100&sort_by=seq&sort_order=asc`);
    const data = await response.json() as { items?: MessageRow[] };
    setMessages((data.items ?? []).filter(isVisibleChatRow).map(rowToMessage));
  }, []);

  const connect = useCallback(() => {
    const current = socketRef.current;
    if (current && current.readyState <= WebSocket.OPEN) {
      return current;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      const frame = JSON.parse(String(event.data)) as ChatFrame;
      handleFrame(frame, {
        activeSessionRef,
        setActiveSessionId,
        setError,
        setMessages,
        setStatus,
        loadSessions,
      });
    };
    socket.onclose = () => {
      if (statusRef.current !== "idle") setStatus("error");
    };
    return socket;
  }, [loadSessions]);

  useEffect(() => {
    void loadSessions();
    const socket = connect();
    return () => socket.close();
  }, [connect, loadSessions]);

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
    const sessionId = await ensureSession();
    const media = await uploadFiles(files);
    const attachments = media.map((item, index) => uploadedFileToAttachment(item, files[index]));
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: cleanText || media.map((item) => item.filename).join("\n"),
        attachments,
        blocks: [],
      },
    ]);
    sendWhenOpen(connect(), {
      type: "message.send",
      request_id: crypto.randomUUID(),
      session_id: sessionId,
      text: cleanText,
      media: media.map((item) => item.upload_path),
    });
  }, [connect, ensureSession]);

  const stopTurn = useCallback(() => {
    if (!activeSessionId) return;
    sendWhenOpen(connect(), {
      type: "turn.stop",
      request_id: crypto.randomUUID(),
      session_id: activeSessionId,
    });
    setStatus("idle");
  }, [activeSessionId, connect]);

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
                  void loadMessages(session.key);
                }}
              >
                {sessionLabel(session)}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <section className="chat-main">
        <Conversation className="conversation">
          <ConversationContent className={messages.length ? "conversation-content" : "conversation-content empty"}>
            {messages.length === 0 ? (
              <ConversationEmptyState className="home-state">
                <h1>今天有什么计划?</h1>
              </ConversationEmptyState>
            ) : (
              messages.map((message, index) => (
                <React.Fragment key={message.id}>
                  {messages[index - 1]?.role === message.role ? <RoleDivider role={message.role} /> : null}
                  <ChatMessageView message={message} />
                </React.Fragment>
              ))
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
    </main>
  );
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

function ChatMessageView({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <Message from="user" className="message-row user-row">
        <MessageContent className="user-bubble">
          {message.attachments?.length ? <MessageAttachments attachments={message.attachments} /> : null}
          {message.content ? <MessageResponse>{message.content}</MessageResponse> : null}
        </MessageContent>
      </Message>
    );
  }

  return (
    <Message from="assistant" className="message-row agent-row">
      <MessageContent className="agent-content">
        {message.blocks.length ? <ProcessTrace message={message} /> : null}
        {message.attachments?.length ? <MessageAttachments attachments={message.attachments} /> : null}
        {message.content && <MessageResponse>{message.content}</MessageResponse>}
      </MessageContent>
    </Message>
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

function MessageAttachments({ attachments }: { attachments: MessageAttachment[] }) {
  return (
    <Attachments className="message-attachments" variant="grid">
      {attachments.map((attachment) => (
        <MessageAttachmentItem attachment={attachment} key={attachment.id} />
      ))}
    </Attachments>
  );
}

function MessageAttachmentItem({ attachment }: { attachment: MessageAttachment }) {
  const isImage = getMediaCategory(attachment) === "image" && attachment.url;
  const label = getAttachmentLabel(attachment);

  if (isImage) {
    return (
      <Dialog>
        <DialogTrigger asChild>
          <Attachment
            className="previewable-attachment"
            data={attachment}
            role="button"
            tabIndex={0}
            title="点击预览图片"
          >
            <AttachmentPreview />
          </Attachment>
        </DialogTrigger>
        <DialogContent className="image-preview-dialog">
          <DialogTitle className="sr-only">{label}</DialogTitle>
          <img alt={label} className="image-preview-full" src={attachment.url} />
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <AttachmentHoverCard>
      <AttachmentHoverCardTrigger asChild>
        <Attachment data={attachment}>
          <AttachmentPreview />
        </Attachment>
      </AttachmentHoverCardTrigger>
      <AttachmentHoverCardContent>
        <AttachmentHover attachment={attachment} />
      </AttachmentHoverCardContent>
    </AttachmentHoverCard>
  );
}

function ProcessTrace({ message }: { message: ChatMessage }) {
  return (
    <Reasoning
      className="process-trace"
      isStreaming={!!message.streaming}
      defaultOpen={!!message.streaming}
      duration={message.durationMs ? Math.max(1, Math.round(message.durationMs / 1000)) : undefined}
    >
      <ProcessTraceTrigger />
      <CollapsibleContent className="process-content">
        <div className="process-line" aria-hidden="true" />
        <div className="process-items">
          {message.blocks.map((block, index) => (
            block.kind === "thinking"
              ? <ThinkingStep key={`thinking-${index}`} block={block} active={!!message.streaming && index === message.blocks.length - 1} />
              : <ToolStep key={block.callId} block={block} active={block.status === "input-available"} />
          ))}
        </div>
      </CollapsibleContent>
    </Reasoning>
  );
}

function ProcessTraceTrigger() {
  const { isOpen, isStreaming, duration } = useReasoning();
  return (
    <ReasoningTrigger className="process-trigger">
      <span>{isStreaming ? "正在思考" : `已思考${duration ? ` ${duration}s` : ""}`}</span>
      <ChevronDown className={`process-chevron ${isOpen ? "open" : ""}`} size={15} />
    </ReasoningTrigger>
  );
}

function ThinkingStep({ block, active }: { block: ThinkingBlock; active: boolean }) {
  return (
    <div className={`process-item thinking-step ${active ? "active" : ""}`}>
      <span className="process-node circle" />
      <div className="process-text">{block.content}</div>
    </div>
  );
}

function ToolStep({ block, active }: { block: ToolBlock; active: boolean }) {
  const description = toolDescription(block.input);
  return (
    <div className={`process-item tool-step ${active ? "active" : ""} ${block.status === "output-error" ? "error" : ""}`}>
      <span className="process-node diamond" />
      <div className="tool-step-body">
        <div className="tool-step-title">
          <Wrench className="tool-step-icon" size={14} />
          <span>{block.name}</span>
        </div>
        {description ? <div className="tool-step-description">{description}</div> : null}
      </div>
    </div>
  );
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
    ctx.setMessages((messages) => updateTool(messages, frame.call_id, {
      status: frame.status === "error" ? "output-error" : "output-available",
      output: frame.result_preview,
      errorText: frame.status === "error" ? frame.result_preview : undefined,
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

function sendWhenOpen(socket: WebSocket, payload: Record<string, unknown>) {
  const send = () => socket.send(JSON.stringify(payload));
  if (socket.readyState === WebSocket.OPEN) {
    send();
    return;
  }
  socket.addEventListener("open", send, { once: true });
}

async function uploadFiles(files: ComposerFile[]) {
  const result: UploadedFile[] = [];
  for (const file of files) {
    if (!file.url) continue;
    const blob = await fetch(file.url).then((response) => response.blob());
    const filename = file.filename || "upload.bin";
    const response = await fetch(`/api/chat/uploads?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      body: blob,
    });
    result.push(await response.json() as UploadedFile);
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
  const status = stringValue(item.status) === "error" ? "output-error" : "output-available";
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

function toolDescription(input: unknown) {
  const inputRecord = recordValue(input);
  return stringValue(inputRecord?.description);
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

function uploadedFileToAttachment(file: UploadedFile, source?: ComposerFile): MessageAttachment {
  const filename = file.filename || filenameFromPath(file.upload_path);
  return {
    id: file.upload_path,
    type: "file",
    filename,
    mediaType: source?.mediaType || guessMediaType(filename),
    url: source?.url || file.upload_url || mediaUrl(file.upload_path),
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

createRoot(document.getElementById("root")!).render(
  <TooltipProvider>
    <App />
  </TooltipProvider>,
);
