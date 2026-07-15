import "./mobile-polyfills";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  AlertCircle,
  Check,
  ChevronDown,
  FileText,
  Menu,
  MessageSquarePlus,
  Paperclip,
  RefreshCw,
  RotateCcw,
  SendHorizontal,
  Share2,
  Square,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ChatMessageView } from "./message-view";
import {
  MobilePluginSlot,
  receiveMobilePluginAssets,
  settleMobilePluginResponses,
  type MobilePluginAsset,
} from "./mobile-plugin-runtime";
import type { AgentBlock, ChatMessage } from "./main";
import "./mobile-native.css";

type ConnectionStatus = "connecting" | "ready" | "degraded" | "reconnecting" | "disconnected";
type ProcessState = "completed" | "running" | "failed";

interface MobileAttachment {
  id: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  transferredBytes: number;
  state: string;
  canRemove?: boolean;
  contentUrl?: string;
}

interface MobileProcessBlock {
  id: string;
  kind: "thinking" | "tool";
  title: string;
  detail: string;
  state: ProcessState;
}

interface MobileMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  deliveryLabel?: string;
  blocks: MobileProcessBlock[];
  streaming: boolean;
  interrupted: boolean;
  durationSeconds?: number;
  attachments: MobileAttachment[];
}

export interface MobileSnapshot {
  protocolVersion: 1;
  connection: {
    label: string;
    status: ConnectionStatus;
    notice?: string;
    error?: string;
  };
  sessions: { id: string; title: string }[];
  selectedSessionId?: string;
  messages: MobileMessage[];
  pluginResponses: { requestId: string; resultJson?: string; error?: string }[];
  composer: {
    attachments: MobileAttachment[];
    commands: { command: string; description: string }[];
    isStreaming: boolean;
    isResyncing: boolean;
    canResync: boolean;
    isStopping: boolean;
    canStop: boolean;
    canSend: boolean;
  };
}

interface NativeBridge {
  reportReady(): void;
  requestSnapshot(): void;
  selectSession(sessionId: string): void;
  createSession(): void;
  restartPairing(): void;
  reloadFromServer(): void;
  chooseAttachments(): void;
  removeAttachment(attachmentId: string): void;
  retryAttachment(attachmentId: string): void;
  retryDownloadedAttachment(attachmentId: string): void;
  touchDownloadedAttachment(attachmentId: string): void;
  openDownloadedAttachment(attachmentId: string): void;
  shareDownloadedAttachment(attachmentId: string): void;
  dismissError(): void;
  sendMessage(text: string): void;
  sendCommand(command: string): void;
  stopTurn(): void;
  callPluginUi(
    requestId: string,
    sessionId: string | null,
    turnId: string | null,
    pluginId: string,
    method: string,
    payloadJson: string,
  ): void;
  acknowledgePluginUiResponses(requestIdsJson: string): void;
}

type SnapshotRecord = Record<string, unknown>;

function requireRecord(value: unknown, label: string): SnapshotRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} 不是对象`);
  return value as SnapshotRecord;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} 不是字符串`);
  return value;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${label} 不是布尔值`);
  return value;
}

function requireNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} 不是有效数字`);
  return value;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  return requireString(value, label);
}

function requireArray<T>(value: unknown, label: string, parse: (item: unknown, index: number) => T): T[] {
  if (!Array.isArray(value)) throw new Error(`${label} 不是数组`);
  return value.map(parse);
}

function optionalArray<T>(value: unknown, label: string, parse: (item: unknown, index: number) => T): T[] {
  return value === undefined ? [] : requireArray(value, label, parse);
}

function parseAttachment(value: unknown, index: number): MobileAttachment {
  const raw = requireRecord(value, `attachments[${index}]`);
  return {
    id: requireString(raw.id, `attachments[${index}].id`),
    filename: requireString(raw.filename, `attachments[${index}].filename`),
    contentType: requireString(raw.contentType, `attachments[${index}].contentType`),
    sizeBytes: requireNumber(raw.sizeBytes, `attachments[${index}].sizeBytes`),
    transferredBytes: requireNumber(raw.transferredBytes, `attachments[${index}].transferredBytes`),
    state: requireString(raw.state, `attachments[${index}].state`),
    canRemove: raw.canRemove === undefined ? false : requireBoolean(raw.canRemove, `attachments[${index}].canRemove`),
    contentUrl: optionalString(raw.contentUrl, `attachments[${index}].contentUrl`),
  };
}

function parseProcessBlock(value: unknown, index: number): MobileProcessBlock {
  const raw = requireRecord(value, `blocks[${index}]`);
  const kind = requireString(raw.kind, `blocks[${index}].kind`);
  const state = requireString(raw.state, `blocks[${index}].state`);
  if (kind !== "thinking" && kind !== "tool") throw new Error(`blocks[${index}].kind 不受支持`);
  if (state !== "completed" && state !== "running" && state !== "failed") {
    throw new Error(`blocks[${index}].state 不受支持`);
  }
  return {
    id: requireString(raw.id, `blocks[${index}].id`),
    kind,
    title: requireString(raw.title, `blocks[${index}].title`),
    detail: requireString(raw.detail, `blocks[${index}].detail`),
    state,
  };
}

function parseMessage(value: unknown, index: number): MobileMessage {
  const raw = requireRecord(value, `messages[${index}]`);
  const role = requireString(raw.role, `messages[${index}].role`);
  if (role !== "user" && role !== "assistant") throw new Error(`messages[${index}].role 不受支持`);
  return {
    id: requireString(raw.id, `messages[${index}].id`),
    sessionId: requireString(raw.sessionId, `messages[${index}].sessionId`),
    role,
    content: requireString(raw.content, `messages[${index}].content`),
    deliveryLabel: optionalString(raw.deliveryLabel, `messages[${index}].deliveryLabel`),
    blocks: optionalArray(raw.blocks, `messages[${index}].blocks`, parseProcessBlock),
    streaming: raw.streaming === undefined ? false : requireBoolean(raw.streaming, `messages[${index}].streaming`),
    interrupted: raw.interrupted === undefined ? false : requireBoolean(raw.interrupted, `messages[${index}].interrupted`),
    durationSeconds: raw.durationSeconds === undefined ? undefined : requireNumber(raw.durationSeconds, `messages[${index}].durationSeconds`),
    attachments: optionalArray(raw.attachments, `messages[${index}].attachments`, parseAttachment),
  };
}

/** 在 native 协议边界校验完整快照，并只补齐 Kotlin 明确定义的默认字段。 */
function parseMobileSnapshot(value: unknown): MobileSnapshot {
  // 1. 校验协议版本与根对象
  const raw = requireRecord(value, "snapshot");
  if (raw.protocolVersion !== 1) throw new Error(`不支持的移动端协议版本: ${String(raw.protocolVersion)}`);
  const connection = requireRecord(raw.connection, "connection");
  const status = requireString(connection.status, "connection.status");
  if (!["connecting", "ready", "degraded", "reconnecting", "disconnected"].includes(status)) {
    throw new Error(`connection.status 不受支持: ${status}`);
  }

  // 2. 校验会话、消息和插件响应
  const sessions = requireArray(raw.sessions, "sessions", (item, index) => {
    const session = requireRecord(item, `sessions[${index}]`);
    return { id: requireString(session.id, `sessions[${index}].id`), title: requireString(session.title, `sessions[${index}].title`) };
  });
  const messages = requireArray(raw.messages, "messages", parseMessage);
  const pluginResponses = requireArray(raw.pluginResponses, "pluginResponses", (item, index) => {
    const response = requireRecord(item, `pluginResponses[${index}]`);
    return {
      requestId: requireString(response.requestId, `pluginResponses[${index}].requestId`),
      resultJson: optionalString(response.resultJson, `pluginResponses[${index}].resultJson`),
      error: optionalString(response.error, `pluginResponses[${index}].error`),
    };
  });

  // 3. 校验输入区状态并构造内部类型
  const composer = requireRecord(raw.composer, "composer");
  return {
    protocolVersion: 1,
    connection: {
      label: requireString(connection.label, "connection.label"),
      status: status as ConnectionStatus,
      notice: optionalString(connection.notice, "connection.notice"),
      error: optionalString(connection.error, "connection.error"),
    },
    sessions,
    selectedSessionId: optionalString(raw.selectedSessionId, "selectedSessionId"),
    messages,
    pluginResponses,
    composer: {
      attachments: requireArray(composer.attachments, "composer.attachments", parseAttachment),
      commands: requireArray(composer.commands, "composer.commands", (item, index) => {
        const command = requireRecord(item, `composer.commands[${index}]`);
        return {
          command: requireString(command.command, `composer.commands[${index}].command`),
          description: requireString(command.description, `composer.commands[${index}].description`),
        };
      }),
      isStreaming: requireBoolean(composer.isStreaming, "composer.isStreaming"),
      isResyncing: requireBoolean(composer.isResyncing, "composer.isResyncing"),
      canResync: requireBoolean(composer.canResync, "composer.canResync"),
      isStopping: requireBoolean(composer.isStopping, "composer.isStopping"),
      canStop: requireBoolean(composer.canStop, "composer.canStop"),
      canSend: requireBoolean(composer.canSend, "composer.canSend"),
    },
  };
}

declare global {
  interface Window {
    AkashicNative?: NativeBridge;
    AkashicMobile?: {
      receiveSnapshot(snapshot: unknown): void;
      receivePluginAssets(assets: MobilePluginAsset[]): void;
    };
  }
}

function MobileNativeApp() {
  const [snapshot, setSnapshot] = useState<MobileSnapshot | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [commandsOpen, setCommandsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [stopRequested, setStopRequested] = useState(false);
  const [pluginLoadError, setPluginLoadError] = useState<string | null>(null);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const drawerToggleRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let pending: MobileSnapshot | null = null;
    let frame: number | null = null;
    let requestTimer: number | null = null;
    let snapshotAccepted = false;
    const requestSnapshot = () => {
      if (snapshotAccepted) return;
      window.AkashicNative?.requestSnapshot();
      requestTimer = window.setTimeout(requestSnapshot, 250);
    };
    window.AkashicMobile = {
      receiveSnapshot(next) {
        try {
          const parsed = parseMobileSnapshot(next);
          settleMobilePluginResponses(parsed.pluginResponses);
          pending = parsed;
          setSnapshotError(null);
        } catch (error) {
          console.error("[mobile] rejected native snapshot", error);
          setSnapshotError(error instanceof Error ? error.message : "原生快照无效");
          return;
        }
        if (frame !== null) return;
        frame = requestAnimationFrame(() => {
          frame = null;
          setSnapshot(pending);
          pending = null;
          snapshotAccepted = true;
          if (requestTimer !== null) window.clearTimeout(requestTimer);
        });
      },
      receivePluginAssets(assets) {
        void receiveMobilePluginAssets(assets).then(
          () => setPluginLoadError(null),
          (error: unknown) => {
            console.error("[mobile] failed to load plugin assets", error);
            setPluginLoadError(error instanceof Error ? error.message : "插件界面加载失败");
          },
        );
      },
    };
    window.AkashicNative?.reportReady();
    requestSnapshot();
    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      if (requestTimer !== null) window.clearTimeout(requestTimer);
      delete window.AkashicMobile;
    };
  }, []);

  useEffect(() => {
    if (snapshot?.composer.isStopping || !snapshot?.composer.canStop || snapshot?.connection.error) {
      setStopRequested(false);
    }
  }, [snapshot?.composer.canStop, snapshot?.composer.isStopping, snapshot?.connection.error]);

  const messages = useMemo(
    () => snapshot?.messages.map(toChatMessage) ?? [],
    [snapshot?.messages],
  );
  if (!snapshot) {
    if (snapshotError) {
      return (
        <main className="mobile-fatal" role="alert">
          <AlertCircle className="mobile-fatal__mark" size={28} />
          <h1>会话数据没有通过校验</h1>
          <p>{snapshotError}</p>
          <button type="button" onClick={() => window.location.reload()}>
            <RefreshCw size={18} />
            重新载入
          </button>
        </main>
      );
    }
    return (
      <main className="mobile-loading" aria-live="polite">
        <span className="mobile-loading__mark" />
        <span>正在载入对话</span>
      </main>
    );
  }

  const send = () => {
    const text = input.trim();
    if (!text && snapshot.composer.attachments.length === 0) return;
    window.AkashicNative?.sendMessage(text);
    setInput("");
    setCommandsOpen(false);
    textareaRef.current?.blur();
  };
  const stop = () => {
    if (!snapshot.composer.canStop || stopRequested) return;
    setStopRequested(true);
    window.AkashicNative?.stopTurn();
  };
  const closeDrawer = () => {
    setDrawerOpen(false);
    requestAnimationFrame(() => drawerToggleRef.current?.focus());
  };
  const toggleDrawer = () => {
    if (drawerOpen) closeDrawer();
    else setDrawerOpen(true);
  };
  const closeCommands = (restoreFocus = true) => {
    setCommandsOpen(false);
    if (restoreFocus) requestAnimationFrame(() => textareaRef.current?.focus());
  };
  const toggleCommands = () => {
    if (commandsOpen) closeCommands();
    else setCommandsOpen(true);
  };

  return (
    <TooltipProvider>
      <main className="mobile-shell">
        <MobileTopBar
          status={snapshot.connection.status}
          label={snapshot.connection.label}
          drawerOpen={drawerOpen}
          toggleRef={drawerToggleRef}
          onToggleDrawer={toggleDrawer}
        />
        <MobileDrawer
          open={drawerOpen}
          snapshot={snapshot}
          onClose={closeDrawer}
        />

        <div className="mobile-main-content" inert={drawerOpen ? true : undefined}>
          {pluginLoadError ? (
            <div className="mobile-plugin-load-error" role="status">
              插件界面暂不可用 · {pluginLoadError}
            </div>
          ) : null}
          <Conversation className="mobile-conversation">
            <ConversationContent className="mobile-conversation__content">
            {messages.length === 0 ? (
              <ConversationEmptyState className="mobile-empty">
                <h1>开始一段新对话</h1>
                <p>消息会通过电脑上的 Akashic 实时处理。</p>
              </ConversationEmptyState>
            ) : (
              messages.map((message, index) => (
                <React.Fragment key={message.id}>
                  {messages[index - 1]?.role === message.role ? (
                    <div className={`mobile-role-divider ${message.role}`} />
                  ) : null}
                  <ChatMessageView
                    message={message}
                    attachmentContent={<MobileMessageAttachments attachments={snapshot.messages[index].attachments} />}
                    processStartContent={isPluginTurnMessage(message) ? (
                      <MobilePluginSlot
                        name="turn.before_reasoning"
                        sessionId={snapshot.messages[index].sessionId}
                        messageId={message.id}
                        turnId={pluginTurnId(message)}
                      />
                    ) : undefined}
                    beforeProcessBlock={(block) => isPluginTurnMessage(message) && block.kind === "tool" ? (
                      <MobilePluginSlot
                        name="turn.before_tool"
                        sessionId={snapshot.messages[index].sessionId}
                        messageId={message.id}
                        turnId={pluginTurnId(message)}
                        block={block}
                      />
                    ) : null}
                    answerEndContent={isPluginTurnMessage(message) ? (
                      <MobilePluginSlot
                        name="turn.after_answer"
                        sessionId={snapshot.messages[index].sessionId}
                        messageId={message.id}
                        turnId={pluginTurnId(message)}
                      />
                    ) : undefined}
                  />
                  <MessageMeta source={snapshot.messages[index]} />
                </React.Fragment>
              ))
            )}
            </ConversationContent>
            <MobileAutoScroll messages={messages} streaming={snapshot.composer.isStreaming} />
            <ConversationScrollButton className="mobile-scroll-button" />
          </Conversation>

          <MobileComposer
            snapshot={snapshot}
            input={input}
            textareaRef={textareaRef}
            commandsOpen={commandsOpen}
            stopRequested={stopRequested}
            onInput={setInput}
            onToggleCommands={toggleCommands}
            onCloseCommands={closeCommands}
            onSend={send}
            onStop={stop}
          />
        </div>
      </main>
    </TooltipProvider>
  );
}

function MobileTopBar({
  status,
  label,
  drawerOpen,
  toggleRef,
  onToggleDrawer,
}: {
  status: ConnectionStatus;
  label: string;
  drawerOpen: boolean;
  toggleRef: React.RefObject<HTMLButtonElement | null>;
  onToggleDrawer: () => void;
}) {
  const online = status === "ready";
  return (
    <header className={`mobile-topbar ${drawerOpen ? "drawer-open" : ""}`}>
      <button ref={toggleRef} className="mobile-icon-button drawer-toggle" type="button" onClick={onToggleDrawer} aria-label={drawerOpen ? "收起会话" : "打开会话"} aria-expanded={drawerOpen}>
        {drawerOpen ? <X size={25} /> : <Menu size={25} />}
      </button>
      <div className={`connection-state ${status}`} aria-live="polite">
        {online ? <Wifi size={19} /> : status === "disconnected" ? <WifiOff size={19} /> : <RefreshCw className="connection-spinner" size={18} />}
        <span>{label}</span>
      </div>
    </header>
  );
}

function MobileDrawer({ open, snapshot, onClose }: { open: boolean; snapshot: MobileSnapshot; onClose: () => void }) {
  const drawerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) requestAnimationFrame(() => drawerRef.current?.focus());
  }, [open]);
  return (
    <div className={`mobile-drawer-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="mobile-drawer-scrim" type="button" onClick={onClose} aria-label="关闭会话抽屉" tabIndex={open ? 0 : -1} />
      <aside ref={drawerRef} className="mobile-drawer" role="dialog" aria-modal="true" aria-label="会话列表" tabIndex={-1}>
        <div className="mobile-drawer__heading">会话</div>
        <nav className="mobile-session-list">
          {snapshot.sessions.map((session) => (
            <button
              className={`mobile-session-row ${session.id === snapshot.selectedSessionId ? "active" : ""}`}
              type="button"
              key={session.id}
              onClick={() => {
                window.AkashicNative?.selectSession(session.id);
                onClose();
              }}
            >
              <span>{session.title || "未命名会话"}</span>
              {session.id === snapshot.selectedSessionId ? <Check size={18} /> : null}
            </button>
          ))}
        </nav>
        <MobilePluginSlot name="drawer.panel" sessionId={snapshot.selectedSessionId} />
        <div className="mobile-drawer__actions">
          <button className="drawer-action" type="button" disabled={!snapshot.composer.canResync} onClick={() => {
            if (window.confirm("清除本机已同步消息和附件缓存，并从电脑重新拉取？连接状态会保留。")) {
              window.AkashicNative?.reloadFromServer();
              onClose();
            }
          }}>
            <RotateCcw size={18} />
            <span>{snapshot.composer.isResyncing ? "正在重新同步" : "清理缓存并同步"}</span>
          </button>
          <button className="drawer-action" type="button" onClick={() => window.AkashicNative?.restartPairing()}>
            <RefreshCw size={18} />
            <span>重新扫码</span>
          </button>
          <button className="new-chat-action" type="button" onClick={() => {
            window.AkashicNative?.createSession();
            onClose();
          }}>
            <MessageSquarePlus size={18} />
            <span>新聊天</span>
          </button>
        </div>
      </aside>
    </div>
  );
}

function MobileComposer({
  snapshot,
  input,
  textareaRef,
  commandsOpen,
  stopRequested,
  onInput,
  onToggleCommands,
  onCloseCommands,
  onSend,
  onStop,
}: {
  snapshot: MobileSnapshot;
  input: string;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  commandsOpen: boolean;
  stopRequested: boolean;
  onInput: (value: string) => void;
  onToggleCommands: () => void;
  onCloseCommands: (restoreFocus?: boolean) => void;
  onSend: () => void;
  onStop: () => void;
}) {
  const stopping = stopRequested || snapshot.composer.isStopping;
  const hasDraft = snapshot.composer.attachments.length > 0;
  const canSubmit = snapshot.composer.canSend && (!!input.trim() || hasDraft);

  return (
    <div className="mobile-composer-zone">
      <CommandSheet open={commandsOpen} commands={snapshot.composer.commands} onClose={onCloseCommands} />
      {snapshot.connection.notice ? <div className="connection-notice">{snapshot.connection.notice}</div> : null}
      {snapshot.connection.error ? (
        <button className="mobile-error" type="button" onClick={() => window.AkashicNative?.dismissError()}>
          <AlertCircle size={17} />
          <span>{snapshot.connection.error}</span>
          <X size={16} />
        </button>
      ) : null}
      {stopping ? <div className="stop-feedback" aria-live="polite">正在中止本轮处理…</div> : null}
      {hasDraft ? <DraftAttachments attachments={snapshot.composer.attachments} /> : null}
      <div className="mobile-composer">
        <button className={`mobile-icon-button command-toggle ${commandsOpen ? "active" : ""}`} type="button" onClick={onToggleCommands} aria-label={commandsOpen ? "关闭快捷命令" : "打开快捷命令"}>
          {commandsOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          placeholder="输入消息"
          onChange={(event) => onInput(event.target.value)}
          onFocus={() => commandsOpen && onCloseCommands(false)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <button className="mobile-icon-button" type="button" onClick={() => window.AkashicNative?.chooseAttachments()} aria-label="添加附件">
          <Paperclip size={22} />
        </button>
        {snapshot.composer.canStop || stopping ? (
          <button className={`mobile-send-button stop ${stopping ? "pending" : ""}`} type="button" onClick={onStop} aria-label={stopping ? "正在中止" : "中止回答"} disabled={stopping}>
            <Square size={17} fill="currentColor" />
          </button>
        ) : (
          <button className="mobile-send-button" type="button" onClick={onSend} aria-label="发送消息" disabled={!canSubmit}>
            <SendHorizontal size={22} />
          </button>
        )}
      </div>
    </div>
  );
}

function CommandSheet({ open, commands, onClose }: { open: boolean; commands: MobileSnapshot["composer"]["commands"]; onClose: (restoreFocus?: boolean) => void }) {
  const dispatchedRef = useRef(false);
  useEffect(() => {
    if (open) dispatchedRef.current = false;
  }, [open]);
  return (
    <section className={`command-sheet ${open ? "open" : ""}`} aria-label="快捷命令" aria-hidden={!open}>
      <div className="command-sheet__handle" />
      <div className="command-sheet__header">
        <div>
          <h2>快捷命令</h2>
          <p>选择后立即发送</p>
        </div>
        <button className="mobile-icon-button" type="button" onClick={() => onClose()} aria-label="关闭快捷命令" tabIndex={open ? 0 : -1}><ChevronDown size={22} /></button>
      </div>
      <div className="command-sheet__list">
        {commands.map((item) => (
          <button type="button" className="command-row" key={item.command} tabIndex={open ? 0 : -1} onClick={() => {
            if (dispatchedRef.current) return;
            dispatchedRef.current = true;
            window.AkashicNative?.sendCommand(`/${item.command}`);
            onClose();
          }}>
            <span className="command-row__description">{item.description}</span>
            <span className="command-row__name">/{item.command}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function DraftAttachments({ attachments }: { attachments: MobileAttachment[] }) {
  return (
    <div className="draft-attachments">
      {attachments.map((attachment) => {
        const progress = attachment.sizeBytes > 0
          ? Math.min(100, Math.round(attachment.transferredBytes / attachment.sizeBytes * 100))
          : 0;
        return (
          <div className="draft-attachment" key={attachment.id}>
            <FileText size={19} />
            <div className="draft-attachment__body">
              <div className="draft-attachment__name">{attachment.filename}</div>
              <div className="draft-attachment__status">{attachment.state === "ready" ? "上传完成" : attachment.state === "failed" ? "上传失败" : `${progress}%`}</div>
              <div className="draft-attachment__track"><span style={{ inlineSize: `${progress}%` }} /></div>
            </div>
            {attachment.state === "failed" ? (
              <button type="button" onClick={() => window.AkashicNative?.retryAttachment(attachment.id)}>重试</button>
            ) : attachment.canRemove ? (
              <button type="button" onClick={() => window.AkashicNative?.removeAttachment(attachment.id)} aria-label={`移除 ${attachment.filename}`}><X size={18} /></button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function MobileMessageAttachments({ attachments }: { attachments: MobileAttachment[] }) {
  if (attachments.length === 0) return null;
  return (
    <div className="mobile-message-attachments">
      {attachments.map((attachment) => (
        <MobileMessageAttachment attachment={attachment} key={attachment.id} />
      ))}
    </div>
  );
}

function MobileMessageAttachment({ attachment }: { attachment: MobileAttachment }) {
  const progress = attachment.sizeBytes > 0
    ? Math.min(100, Math.round(attachment.transferredBytes / attachment.sizeBytes * 100))
    : 0;
  const cached = attachment.state === "cached";
  const image = cached && attachment.contentType.startsWith("image/") && attachment.contentUrl;
  const status = attachment.state === "pending"
    ? "等待下载"
    : attachment.state === "downloading"
      ? `下载中 ${progress}%`
      : attachment.state === "failed"
        ? "下载失败"
        : attachment.state === "evicted"
          ? "缓存已清理"
          : "已下载";

  return (
    <div className={`mobile-message-attachment ${attachment.state}`}>
      {image ? (
        <Dialog onOpenChange={(open) => open && window.AkashicNative?.touchDownloadedAttachment(attachment.id)}>
          <DialogTrigger asChild>
            <button className="message-attachment-preview" type="button" aria-label={`预览 ${attachment.filename}`}>
              <img alt="" src={attachment.contentUrl} />
            </button>
          </DialogTrigger>
          <DialogContent className="image-preview-dialog">
            <DialogTitle className="sr-only">{attachment.filename}</DialogTitle>
            <img alt={attachment.filename} className="image-preview-full" src={attachment.contentUrl} />
          </DialogContent>
        </Dialog>
      ) : (
        <button
          className="message-attachment-main"
          type="button"
          disabled={!cached}
          onClick={() => window.AkashicNative?.openDownloadedAttachment(attachment.id)}
        >
          <FileText size={20} />
          <span>
            <strong>{attachment.filename}</strong>
            <small>{status}</small>
          </span>
        </button>
      )}
      {image ? (
        <div className="message-attachment-caption">
          <strong>{attachment.filename}</strong>
          <small>{status}</small>
        </div>
      ) : null}
      {attachment.state === "downloading" ? (
        <div className="message-attachment-progress" aria-label={status}><span style={{ inlineSize: `${progress}%` }} /></div>
      ) : null}
      {attachment.state === "failed" || attachment.state === "evicted" ? (
        <button className="message-attachment-action" type="button" onClick={() => window.AkashicNative?.retryDownloadedAttachment(attachment.id)}>重试</button>
      ) : cached ? (
        <button className="message-attachment-action icon" type="button" onClick={() => window.AkashicNative?.shareDownloadedAttachment(attachment.id)} aria-label={`分享 ${attachment.filename}`}><Share2 size={18} /></button>
      ) : null}
    </div>
  );
}

function MessageMeta({ source }: { source: MobileMessage }) {
  if (source.role === "user") return <div className="mobile-message-meta user">{source.deliveryLabel}</div>;
  if (source.interrupted) return <div className="interrupted-notice">本轮已中止</div>;
  return null;
}

function MobileAutoScroll({ messages, streaming }: { messages: ChatMessage[]; streaming: boolean }) {
  const { escapedFromLock, isAtBottom, scrollToBottom } = useStickToBottomContext();
  const lastMessageCountRef = useRef(messages.length);
  const lastMessage = messages[messages.length - 1];
  const lastBlock = lastMessage?.blocks[lastMessage.blocks.length - 1];
  const scrollKey = `${messages.length}:${lastMessage?.id}:${lastMessage?.content.length}:${lastMessage?.blocks.length}:${lastBlock?.kind === "thinking" ? lastBlock.content.length : ""}`;

  useEffect(() => {
    const newUserMessage = messages.length > lastMessageCountRef.current && lastMessage?.role === "user";
    lastMessageCountRef.current = messages.length;
    if (newUserMessage) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
    } else if (streaming && isAtBottom && !escapedFromLock) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: false });
    }
  }, [escapedFromLock, isAtBottom, lastMessage?.role, messages, scrollKey, scrollToBottom, streaming]);
  return null;
}

function toChatMessage(message: MobileMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    attachments: [],
    blocks: message.blocks.map(toAgentBlock),
    streaming: message.streaming,
    interrupted: message.interrupted,
    durationMs: message.durationSeconds !== undefined ? message.durationSeconds * 1000 : undefined,
  };
}

function isPluginTurnMessage(message: ChatMessage): boolean {
  return message.role === "assistant" && !message.id.startsWith("proactive:");
}

function pluginTurnId(message: ChatMessage): string | undefined {
  if (!message.streaming || !message.id.startsWith("assistant:")) return undefined;
  return message.id.slice("assistant:".length);
}

function toAgentBlock(block: MobileProcessBlock): AgentBlock {
  if (block.kind === "thinking") return { kind: "thinking", content: block.detail || block.title };
  return {
    kind: "tool",
    callId: block.id,
    name: block.title,
    status: block.state === "running" ? "input-available" : block.state === "failed" ? "output-error" : "output-available",
    input: block.detail ? { description: block.detail } : {},
    output: null,
    errorText: block.state === "failed" ? block.detail : undefined,
  };
}

class MobileErrorBoundary extends React.Component<React.PropsWithChildren, { message: string | null }> {
  state: { message: string | null } = { message: null };

  static getDerivedStateFromError(error: unknown) {
    return { message: error instanceof Error ? error.message : "会话界面发生未知错误" };
  }

  componentDidCatch(error: unknown) {
    console.error("[mobile] render failed", error);
  }

  render() {
    if (!this.state.message) return this.props.children;
    return (
      <main className="mobile-fatal" role="alert">
        <AlertCircle className="mobile-fatal__mark" size={28} />
        <h1>会话界面没有正常载入</h1>
        <p>{this.state.message}</p>
        <button type="button" onClick={() => window.location.reload()}>
          <RefreshCw size={18} />
          重新载入
        </button>
      </main>
    );
  }
}

// Android WebView 首帧可能把百分比和视口单位高度冻结为 0，显式像素值同时覆盖键盘和旋转变化。
function syncMobileViewportHeight() {
  const viewportHeight = Math.max(1, Math.round(window.visualViewport?.height ?? window.innerHeight));
  document.documentElement.style.setProperty("--mobile-viewport-height", `${viewportHeight}px`);
}

syncMobileViewportHeight();
window.addEventListener("resize", syncMobileViewportHeight);
window.visualViewport?.addEventListener("resize", syncMobileViewportHeight);

const root = document.getElementById("root");
if (!root) throw new Error("Mobile Web root 不存在");
createRoot(root).render(
  <MobileErrorBoundary>
    <MobileNativeApp />
  </MobileErrorBoundary>,
);
