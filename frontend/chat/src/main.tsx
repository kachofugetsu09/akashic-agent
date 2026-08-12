import React, { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { cycleTheme, initializeTheme, setTheme, startCrossPortThemeSync, useTheme } from "../../theme/src/theme-runtime";
import { MaterialButton } from "../../theme/src/material-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { TooltipProvider } from "@/components/ui/tooltip";
import type {
  ChatMessage,
  ToolBlock,
} from "./chat-message";
import { DesktopConversationMessages } from "./desktop-conversation";
import { DesktopComposer, desktopComposerReplyPreview, type ComposerFile } from "./desktop-composer";
import { DesktopMobileNavigation } from "./desktop-mobile-navigation";
import { DesktopSidebar } from "./desktop-sidebar";
import { loadWebPluginCatalog } from "./mobile-plugin-runtime";
import { StreamProjectionStore, attachReducedMotionFlush } from "./stream-projection";
import {
  advanceWebStreamPresentation,
  canProjectWebStreamWithoutRoot,
  publishWebStreamChanges,
} from "./web-stream-projection";
import type { ChatStatus } from "./web-chat-status";
import {
  chatModelState,
  errorMessage,
  fetchChatJson,
  isAbortError,
  messageRows,
  sessionRows,
  uploadFiles,
  webShellState,
  type ChatModelState,
  type SessionRow,
  type WebShellState,
} from "./web-chat-data";
import {
  blocksWithFinalThinking,
  formatNavigationTime,
  isVisibleChatRow,
  mediaToAttachments,
  mergeAttachments,
  rowToMessage,
  sessionLabel,
  uploadedFileToAttachment,
} from "./web-chat-message-data";
import { webTurnTrace, type WebTurnTraceKind } from "./web-turn-trace";
import { WebUiErrorBoundary } from "./webui-error-boundary";
import "./styles.css";
import "./message-view.css";

export type { AgentBlock, ChatMessage, MessageAttachment, ThinkingBlock, ToolBlock } from "./chat-message";

const LazyMobileShowcase = lazy(() =>
  import("./mobile-showcase").then(({ MobileShowcase }) => ({ default: MobileShowcase })),
);
const LazySharedChatShowcase = lazy(() =>
  import("./shared-chat-showcase").then(({ SharedChatShowcase }) => ({ default: SharedChatShowcase })),
);
const LazyTraceMotionShowcase = lazy(() =>
  import("./trace-motion-showcase").then(({ TraceMotionShowcase }) => ({ default: TraceMotionShowcase })),
);
const LazyDrawerIslandShowcase = lazy(() =>
  import("./drawer-island-showcase").then(({ DrawerIslandShowcase }) => ({ default: DrawerIslandShowcase })),
);
const LazyModelExperienceShowcase = lazy(() =>
  import("./model-experience-showcase").then(({ ModelExperienceShowcase }) => ({ default: ModelExperienceShowcase })),
);
const LazySettingsApp = lazy(() =>
  import("./settings-app").then(({ SettingsApp }) => ({ default: SettingsApp })),
);
const LazyMobilePairingDialog = lazy(() =>
  import("./mobile-pairing-dialog").then(({ MobilePairingDialog }) => ({ default: MobilePairingDialog })),
);
const LazyRuntimeDashboard = lazy(() =>
  import("./runtime-dashboard").then(({ RuntimeDashboard }) => ({ default: RuntimeDashboard })),
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

function App() {
  const theme = useTheme();
  const [surface, setSurface] = useState<"chat" | "runtime">(
    () => new URLSearchParams(window.location.search).get("surface") === "runtime" ? "runtime" : "chat",
  );
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState("");
  const [streamStore] = useState(() => new StreamProjectionStore<ChatMessage>(
    {
      request: (callback) => window.requestAnimationFrame(callback),
      cancel: (handle) => window.cancelAnimationFrame(handle),
    },
    advanceWebStreamPresentation,
  ));
  const [messages, setMessagesState] = useState<ChatMessage[]>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  const commitMessages = useCallback((
    action: React.SetStateAction<ChatMessage[]>,
    revealImmediately: boolean,
  ) => {
    // 1. Resolve every WebSocket mutation against a synchronous immutable baseline.
    const previous = messagesRef.current;
    const next = typeof action === "function" ? action(previous) : action;
    messagesRef.current = next;
    const projectionOnly = canProjectWebStreamWithoutRoot(previous, next);

    // 2. Seed the row projection before React receives the authoritative chunk.
    if (next.length === 0) {
      streamStore.clear();
    } else {
      publishWebStreamChanges(
        previous,
        next,
        streamStore,
        revealImmediately || window.matchMedia("(prefers-reduced-motion: reduce)").matches,
      );
      const projectionMessage = next.at(-1);
      if (projectionMessage?.role === "assistant" && projectionMessage.streaming !== undefined) {
        webTurnTrace.markProjection(projectionMessage.id);
      }
    }
    if (!projectionOnly) setMessagesState(next);
  }, [streamStore]);
  const setMessages = useCallback<React.Dispatch<React.SetStateAction<ChatMessage[]>>>(
    (action) => commitMessages(action, false),
    [commitMessages],
  );
  const setMessagesImmediate = useCallback<React.Dispatch<React.SetStateAction<ChatMessage[]>>>(
    (action) => commitMessages(action, true),
    [commitMessages],
  );
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [stopPending, setStopPending] = useState(false);
  const [error, setError] = useState("");
  const [mobilePairingOpen, setMobilePairingOpen] = useState(false);
  const [shellState, setShellState] = useState<WebShellState | null>(null);
  const [replyTarget, setReplyTarget] = useState<ChatMessage | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const [modelState, setModelState] = useState<ChatModelState | null>(null);
  const [selectedRuntimeId, setSelectedRuntimeId] = useState("");
  const [selectedReasoningEffort, setSelectedReasoningEffort] = useState("");
  const [modelSelectionDirty, setModelSelectionDirty] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const messageElementsRef = useRef(new Map<string, HTMLDivElement>());
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");
  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const modelsRequestRef = useRef<AbortController | null>(null);
  const sendRequestRef = useRef<AbortController | null>(null);
  const stopRequestRef = useRef<AbortController | null>(null);
  const chatReady = shellState?.chatReady === true;

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
      streamStore.clear();
      setMessages(messageRows(payload, endpoint).filter(isVisibleChatRow).map(rowToMessage));
    } finally {
      if (messagesRequestRef.current === controller) messagesRequestRef.current = null;
    }
  }, [setMessages, streamStore]);

  useEffect(() => {
    streamStore.reconcileBaseline(messages);
  }, [messages, streamStore]);

  useEffect(() => () => streamStore.clear(), [streamStore]);

  // 切入 prefers-reduced-motion: reduce 时立即补齐积压，即使没有新 delta；
  // 卸载时移除 listener。初始化已 reduce 的行为由 publish 处的 matchMedia 判断保持即时。
  useEffect(() => attachReducedMotionFlush(streamStore), [streamStore]);

  const loadSessionsSafely = useCallback(() => loadSessions().catch((error: unknown) => reportError(error)), [loadSessions, reportError]);
  const loadMessagesSafely = useCallback((sessionId: string) => loadMessages(sessionId).catch((error: unknown) => reportError(error)), [loadMessages, reportError]);

  const loadModels = useCallback(async (sessionId: string) => {
    modelsRequestRef.current?.abort();
    const controller = new AbortController();
    modelsRequestRef.current = controller;
    const query = sessionId ? `?session_key=${encodeURIComponent(sessionId)}` : "";
    try {
      const next = chatModelState(await fetchChatJson<unknown>(`/api/chat/models${query}`, { signal: controller.signal }));
      setModelState(next);
      setSelectedRuntimeId(next.sessionOverride);
      setSelectedReasoningEffort(next.sessionSelection.reasoningEffort);
      setModelSelectionDirty(false);
    } finally {
      if (modelsRequestRef.current === controller) modelsRequestRef.current = null;
    }
  }, []);

  useEffect(() => {
    const handleModelsChanged = (event: MessageEvent<unknown>): void => {
      const payload = event.data;
      if (
        event.origin !== window.location.origin
        || event.source !== window.parent
        || typeof payload !== "object"
        || payload === null
        || !("type" in payload)
        || payload.type !== "akashic.models.changed"
      ) return;
      void loadModels(activeSessionRef.current).catch((error: unknown) => reportError(error));
    };
    window.addEventListener("message", handleModelsChanged);
    return () => window.removeEventListener("message", handleModelsChanged);
  }, [loadModels, reportError]);

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
        const traceKind = webFrameTraceKind(frame);
        if (traceKind !== undefined && "session_id" in frame && "turn_id" in frame) {
          webTurnTrace.observeFrame(frame.session_id, frame.turn_id, traceKind);
        }
        handleFrame(frame, {
          activeSessionRef,
          setActiveSessionId,
          setError,
          setMessages,
          setMessagesImmediate,
          setStatus,
          loadSessions: loadSessionsSafely,
          loadMessages: loadMessagesSafely,
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
  }, [loadMessagesSafely, loadSessionsSafely, reportError, setMessages, setMessagesImmediate]);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const next = webShellState(await fetchChatJson<unknown>("/api/shell/state"));
        if (active) setShellState(next);
      } catch (stateError) {
        if (active) reportError(stateError);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1200);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [reportError]);

  useEffect(() => {
    if (!chatReady) return;
    void loadSessionsSafely();
    void loadModels("").catch((error: unknown) => reportError(error));
    const socket = connect();
    return () => {
      sessionsRequestRef.current?.abort();
      messagesRequestRef.current?.abort();
      modelsRequestRef.current?.abort();
      sendRequestRef.current?.abort();
      stopRequestRef.current?.abort();
      if (socketRef.current === socket) socketRef.current = null;
      socket.close(1000, "component unmounted");
    };
  }, [chatReady, connect, loadModels, loadSessionsSafely, reportError]);

  useEffect(() => {
    if (!chatReady) return;
    const controller = new AbortController();
    void loadWebPluginCatalog(controller.signal).catch((error: unknown) => {
      if (!isAbortError(error)) reportError(error);
    });
    return () => controller.abort();
  }, [chatReady, reportError]);

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
    sendRequestRef.current?.abort();
    const controller = new AbortController();
    sendRequestRef.current = controller;
    const optimisticId = crypto.randomUUID();
    const reply = replyTarget;
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
          createdAt: new Date().toISOString(),
          canonical: false,
          reply: reply ? {
            messageId: reply.id,
            role: reply.role,
            preview: desktopComposerReplyPreview(reply),
          } : undefined,
        },
      ]);
      const payload: Record<string, unknown> = {
        type: "message.send",
        request_id: crypto.randomUUID(),
        session_id: sessionId,
        text: cleanText,
        media: media.map((item) => item.upload_path),
      };
      if (reply) payload.reply_to_message_id = reply.id;
      if (modelSelectionDirty) {
        payload.model_runtime_id = selectedRuntimeId;
        payload.model_reasoning_effort = selectedReasoningEffort;
      }
      await sendWhenOpen(connect(), payload, controller.signal);
      setModelSelectionDirty(false);
      setReplyTarget(null);
    } catch (error) {
      if (isAbortError(error)) throw error;
      setMessages((current) => current.filter((message) => message.id !== optimisticId));
      reportError(error, "error");
      throw error;
    } finally {
      if (sendRequestRef.current === controller) sendRequestRef.current = null;
    }
  }, [connect, ensureSession, modelSelectionDirty, replyTarget, reportError, selectedReasoningEffort, selectedRuntimeId, setMessages]);

  const stopTurn = useCallback(() => {
    if (!activeSessionId || stopRequestRef.current) return;
    const controller = new AbortController();
    stopRequestRef.current = controller;
    setStopPending(true);
    void sendWhenOpen(connect(), {
      type: "turn.stop",
      request_id: crypto.randomUUID(),
      session_id: activeSessionId,
    }, controller.signal)
      .then(() => setStatus("idle"))
      .catch((error: unknown) => reportError(error, "error"))
      .finally(() => {
        if (stopRequestRef.current === controller) stopRequestRef.current = null;
        setStopPending(false);
      });
  }, [activeSessionId, connect, reportError]);

  const startNewChat = useCallback(() => {
    setSurface("chat");
    window.history.replaceState(null, "", window.location.pathname);
    activeSessionRef.current = "";
    messagesRequestRef.current?.abort();
    modelsRequestRef.current?.abort();
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    setActiveSessionId("");
    setPendingSessionId("");
    setMessages([]);
    setReplyTarget(null);
    setStatus("idle");
    setStopPending(false);
    setSelectedRuntimeId("");
    setSelectedReasoningEffort("");
    setModelSelectionDirty(false);
    void loadModels("").catch((error: unknown) => reportError(error));
  }, [loadModels, reportError, setMessages]);

  const activateSession = useCallback((sessionId: string) => {
    if (surface === "chat" && activeSessionRef.current === sessionId) return;
    setSurface("chat");
    window.history.replaceState(null, "", window.location.pathname);
    activeSessionRef.current = sessionId;
    setActiveSessionId(sessionId);
    setPendingSessionId(sessionId);
    setReplyTarget(null);
    setModelState(null);
    setSelectedRuntimeId("");
    setModelSelectionDirty(false);
    void Promise.all([loadMessages(sessionId), loadModels(sessionId)])
      .catch((reason: unknown) => reportError(reason))
      .finally(() => {
        if (activeSessionRef.current === sessionId) setPendingSessionId("");
      });
  }, [loadMessages, loadModels, reportError, surface]);

  const openRuntime = useCallback(() => {
    setSurface("runtime");
    window.history.replaceState(null, "", `${window.location.pathname}?surface=runtime`);
  }, []);
  const handleReplyMessage = useCallback((message: ChatMessage) => setReplyTarget(message), []);
  const handleModelChange = useCallback((runtimeId: string, effort: string) => {
    setSelectedRuntimeId(runtimeId);
    setSelectedReasoningEffort(effort);
    setModelSelectionDirty(true);
  }, []);
  const cancelReply = useCallback(() => setReplyTarget(null), []);
  const handleCopiedMessage = useCallback((messageId: string) => {
    setCopiedMessageId(messageId);
    window.setTimeout(() => setCopiedMessageId(""), 1200);
  }, []);
  const sidebarSessions = useMemo(() => sessions.map((session) => ({
    id: session.key,
    title: sessionLabel(session),
    preview: `${session.message_count ?? 0} 条消息`,
    updatedLabel: formatNavigationTime(session.updated_at),
    active: activeSessionId === session.key,
  })), [activeSessionId, sessions]);
  const openPairing = useCallback(() => setMobilePairingOpen(true), []);

  return (
    <main className={`chat-shell ${isEmbeddedRuntime ? "embedded-runtime" : ""}`}>
      {!isEmbeddedRuntime ? (
        <>
          <DesktopSidebar
            embeddedShell={isEmbeddedShell} surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} onSelectSession={activateSession} onOpenRuntime={openRuntime}
            onCycleTheme={cycleTheme} onOpenPairing={openPairing} onNewChat={startNewChat}
          />
          <DesktopMobileNavigation
            embeddedShell={isEmbeddedShell} surface={surface} sessions={sidebarSessions}
            activeSessionId={activeSessionId} pendingSessionId={pendingSessionId} chatReady={chatReady}
            themeLabel={theme.label} onSelectSession={activateSession} onOpenRuntime={openRuntime}
            onCycleTheme={cycleTheme} onOpenPairing={openPairing} onNewChat={startNewChat}
          />
        </>
      ) : null}

      {surface === "runtime" ? (
        <Suspense fallback={<section className="runtime-dashboard" aria-busy="true">正在加载知识与运行…</section>}>
          <LazyRuntimeDashboard />
        </Suspense>
      ) : <section className="chat-main">
        <Conversation className="conversation" resize={status === "streaming" ? "smooth" : "instant"}>
          <ConversationContent className={messages.length ? "conversation-content" : "conversation-content empty"}>
            {messages.length === 0 ? (
              <ConversationEmptyState className="home-state">
                {shellState?.status === "needs_setup" ? (
                  <div className="model-connection-state">
                    <span>首次使用</span>
                    <h1>先连接一个模型</h1>
                    <p>绑定 Codex、OpenCode 或自己的 API Key 后，就可以在这里直接对话。</p>
                    <a href="/settings">连接模型</a>
                  </div>
                ) : shellState?.status === "starting" ? (
                  <div className="model-connection-state">
                    <span>正在启动</span>
                    <h1>模型已保存，Akashic 正在准备对话</h1>
                    <p>这个页面会自动恢复，不需要切换端口或刷新浏览器。</p>
                    <a href="/settings">查看模型设置</a>
                  </div>
                ) : shellState === null ? (
                  <h1>正在连接 Akashic…</h1>
                ) : (
                  <h1>今天有什么计划?</h1>
                )}
              </ConversationEmptyState>
            ) : (
              <MessageRendererErrorBoundary>
                <DesktopConversationMessages
                  messages={messages}
                  activeSessionId={activeSessionId}
                  status={status}
                  copiedMessageId={copiedMessageId}
                  streamStore={streamStore}
                  messageElementsRef={messageElementsRef}
                  onReply={handleReplyMessage}
                  onCopied={handleCopiedMessage}
                  onError={reportError}
                />
              </MessageRendererErrorBoundary>
            )}
          </ConversationContent>
          <AutoScroll messages={messages} status={status} streamStore={streamStore} />
          <ConversationScrollButton />
        </Conversation>

        <div className={`composer-wrap ${messages.length === 0 ? "home" : ""}`}>
          <DesktopComposer
            chatReady={chatReady}
            status={status}
            stopPending={stopPending}
            modelState={modelState}
            selectedRuntimeId={selectedRuntimeId}
            selectedEffort={selectedReasoningEffort}
            replyTarget={replyTarget}
            onModelChange={handleModelChange}
            onCancelReply={cancelReply}
            onSend={sendMessage}
            onStop={stopTurn}
          />
          {error && <div className="error-line" role="alert"><span>{error}</span><MaterialButton
            variant="danger"
            onClick={() => {
              setError("");
              void loadSessionsSafely();
              if (shellState?.chatReady) void loadModels(activeSessionRef.current).catch((reason: unknown) => reportError(reason));
            }}
          >重试</MaterialButton></div>}
        </div>
      </section>}
      {mobilePairingOpen ? <Suspense fallback={null}>
        <LazyMobilePairingDialog open onOpenChange={setMobilePairingOpen} />
      </Suspense> : null}
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
      return <div className="message-row message-renderer-error" role="alert">
        <span>消息渲染器加载失败</span>
        <button type="button" onClick={() => window.location.reload()}>重新加载页面</button>
      </div>;
    }
    return this.props.children;
  }
}

function AutoScroll({
  messages,
  status,
  streamStore,
}: {
  messages: ChatMessage[];
  status: ChatStatus;
  streamStore: StreamProjectionStore<ChatMessage>;
}) {
  const { escapedFromLock, isAtBottom, scrollToBottom } = useStickToBottomContext();
  const lastMessageCountRef = useRef(messages.length);
  const baselineLastMessage = messages.at(-1);
  const subscribe = useCallback(
    (listener: () => void) => baselineLastMessage
      ? streamStore.subscribe(baselineLastMessage.id, listener)
      : () => {},
    [baselineLastMessage?.id, streamStore],
  );
  const getSnapshot = useCallback(
    () => baselineLastMessage
      ? streamStore.read(baselineLastMessage.id, baselineLastMessage)
      : undefined,
    [baselineLastMessage, streamStore],
  );
  const lastMessage = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const lastBlock = lastMessage?.blocks.at(-1);
  const scrollKey = [
    messages.length,
    lastMessage?.id ?? "",
    lastMessage?.content.length ?? 0,
    lastMessage?.blocks.length ?? 0,
    lastBlock?.kind === "thinking" ? lastBlock.content.length : "",
  ].join(":");

  useEffect(() => {
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

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function webFrameTraceKind(frame: ChatFrame): WebTurnTraceKind | undefined {
  if (frame.type === "react.thinking.delta" && frame.delta !== "") return "thinking";
  if (frame.type === "answer.delta" && frame.delta !== "") return "answer";
  if (frame.type === "message.final") return "terminal";
  return undefined;
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
    setMessagesImmediate: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
    setStatus: React.Dispatch<React.SetStateAction<ChatStatus>>;
    loadSessions: () => Promise<void>;
    loadMessages: (sessionId: string) => Promise<void>;
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
      ctx.setMessagesImmediate((messages) => updateLastAssistant(messages, (message) => ({
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
    void ctx.loadMessages(frame.session_id);
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

const entryParams = new URLSearchParams(window.location.search);
const preview = entryParams.get("preview");
const isEmbeddedShell = entryParams.get("embedded") === "1";
const isEmbeddedRuntime = isEmbeddedShell && entryParams.get("surface") === "runtime";
if (isEmbeddedShell) document.documentElement.dataset.embeddedShell = "true";
initializeTheme();
startCrossPortThemeSync();
if (isEmbeddedShell) {
  const parentOrigins = new Set([
    window.location.origin,
    `${window.location.protocol}//${window.location.hostname}:5173`,
  ]);
  window.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (!parentOrigins.has(event.origin) || typeof event.data !== "object" || event.data === null) return;
    const message = event.data as Record<string, unknown>;
    if (message.type !== "akashic.theme" || typeof message.themeId !== "string") return;
    setTheme(message.themeId, false);
  });
}
const isMobileShowcase = preview === "mobile";
const isSharedChatShowcase = preview === "chat";
const isTraceMotionShowcase = preview === "trace-motion";
const isDrawerIslandShowcase = preview === "drawer-islands";
const isModelExperienceShowcase = preview === "model-experience";
const rootApp = window.location.pathname === "/settings" || window.location.pathname.startsWith("/settings/")
  ? <LazySettingsApp />
  : isMobileShowcase
    ? <LazyMobileShowcase />
    : isSharedChatShowcase
      ? <LazySharedChatShowcase />
      : isTraceMotionShowcase
        ? <LazyTraceMotionShowcase />
      : isDrawerIslandShowcase
        ? <LazyDrawerIslandShowcase />
      : isModelExperienceShowcase
        ? <LazyModelExperienceShowcase />
    : <App />;

createRoot(document.getElementById("root")!).render(
  <WebUiErrorBoundary>
    <TooltipProvider>
      <Suspense fallback={<div className="webui-entry-loading">正在载入界面…</div>}>
        {rootApp}
      </Suspense>
    </TooltipProvider>
  </WebUiErrorBoundary>,
);
