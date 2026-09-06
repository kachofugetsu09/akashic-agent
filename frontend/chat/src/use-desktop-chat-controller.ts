import { useCallback, useEffect, useMemo, useRef, useState, type SetStateAction } from "react";
import { Duration, Effect, Fiber, Schedule } from "effect";
import { createUuid, createUuidV7 } from "./browser-uuid.ts";
import type { ChatMessage } from "./chat-message";
import { desktopComposerReplyPreview, type ComposerFile } from "./desktop-composer";
import { loadWebPluginCatalog } from "./mobile-plugin-runtime";
import { StreamProjectionStore } from "./stream-projection";
import { canProjectWebStreamWithoutRoot, publishWebStreamChanges } from "./web-stream-projection";
import type { ChatStatus } from "./web-chat-status";
import {
  chatHistoryPage,
  chatModelState,
  errorMessage,
  fetchChatJson,
  isAbortError,
  sessionRows,
  uploadFiles,
  webShellState,
  type ChatModelState,
  type SessionRow,
  type WebShellState,
} from "./web-chat-data";
import {
  formatNavigationTime,
  isVisibleChatRow,
  rowToMessage,
  sessionLabel,
  uploadedFileToAttachment,
} from "./web-chat-message-data";
import { ClientTurnMetricsTracker } from "./stream-turn-metrics";
import { applyChatFrame, parseChatFrame, sendWhenOpen, traceKindForChatFrame, type ChatFrame } from "./web-chat-transport";
import { webTurnTrace } from "./web-turn-trace";

function outputTokensFromFrame(frame: ChatFrame): number | null {
  if (frame.type !== "message.final") return null;
  const metadata = frame.metadata;
  if (!metadata) return null;
  const usage = metadata.model_usage ?? metadata.usage;
  if (typeof usage !== "object" || usage === null) return null;
  const record = usage as Record<string, unknown>;
  const raw = record.completion_tokens ?? record.output_tokens ?? record.outputTokens;
  return typeof raw === "number" && Number.isFinite(raw) && raw >= 0 ? raw : null;
}

function observeTurnMetrics(tracker: ClientTurnMetricsTracker, frame: ChatFrame): void {
  if (frame.type === "turn.started") {
    tracker.onTurnStarted(frame.turn_id);
    return;
  }
  if (frame.type === "react.thinking.delta" || frame.type === "answer.delta") {
    tracker.onDelta(frame.turn_id, frame.delta);
    return;
  }
  if (frame.type === "message.final") {
    if (frame.metadata?.source === "message_push") return;
    tracker.onSettled(frame.turn_id, outputTokensFromFrame(frame));
    return;
  }
  if (frame.type === "turn.interrupted" || frame.type === "error") {
    tracker.onInterrupted();
  }
}

function attachSession(socket: WebSocket | null, sessionId: string): void {
  if (!sessionId || socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "session.attach",
    request_id: createUuid(),
    session_id: sessionId,
  }));
}

export function useDesktopChatController() {
  const [surface, setSurface] = useState<"chat" | "runtime">(
    () => new URLSearchParams(window.location.search).get("surface") === "runtime" ? "runtime" : "chat",
  );
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState("");
  const [streamStore] = useState(() => new StreamProjectionStore<ChatMessage>());
  const [turnMetrics] = useState(() => new ClientTurnMetricsTracker());
  const [messages, setMessagesState] = useState<ChatMessage[]>([]);
  const [historyBeforeSeq, setHistoryBeforeSeq] = useState<number | null>(null);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyLoadingOlder, setHistoryLoadingOlder] = useState(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const commitMessages = useCallback((action: SetStateAction<ChatMessage[]>) => {
    // 1. Resolve every WebSocket mutation against a synchronous immutable baseline.
    const previous = messagesRef.current;
    const next = typeof action === "function" ? action(previous) : action;
    messagesRef.current = next;
    const projectionOnly = canProjectWebStreamWithoutRoot(previous, next);

    // 2. Seed the row projection before React receives the authoritative chunk.
    if (next.length === 0) {
      streamStore.clear();
    } else {
      publishWebStreamChanges(previous, next, streamStore);
      const projectionMessage = next.at(-1);
      if (projectionMessage?.role === "assistant" && projectionMessage.streaming !== undefined) {
        webTurnTrace.markProjection(projectionMessage.id);
      }
    }
    if (!projectionOnly) setMessagesState(next);
  }, [streamStore]);
  const setMessages = useCallback<(action: SetStateAction<ChatMessage[]>) => void>(
    (action) => commitMessages(action),
    [commitMessages],
  );
  const [status, setStatus] = useState<ChatStatus>("idle");
  const setStatusLive = useCallback((next: ChatStatus): void => {
    statusLiveRef.current = next;
    setStatus(next);
  }, [setStatus]);
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
  const connectionTaskRef = useRef<Fiber.RuntimeFiber<void> | null>(null);
  const [reconnect] = useState(() => Effect.runSync(Schedule.driver(Schedule.exponential("1 second").pipe(
    Schedule.modifyDelay((_, delay) => Math.min(Duration.toMillis(delay), 30_000)),
    Schedule.jitteredWith({ min: 0, max: 1 }),
    Schedule.intersect(Schedule.recurs(12)),
  ))));
  const messageElementsRef = useRef(new Map<string, HTMLDivElement>());
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");
  const statusLiveRef = useRef<ChatStatus>("idle");
  const activeTurnIdRef = useRef<string | null>(null);
  const liveProjectionRevisionRef = useRef(0);
  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const olderMessagesRequestRef = useRef<AbortController | null>(null);
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
    const projectionRevision = liveProjectionRevisionRef.current;
    try {
      const page = chatHistoryPage(
        await fetchChatJson<unknown>(`${endpoint}?page_size=50`, { signal: controller.signal }),
        endpoint,
      );
      if (
        activeSessionRef.current !== sessionId
        || liveProjectionRevisionRef.current !== projectionRevision
      ) return;
      streamStore.clear();
      setMessages(page.items.filter(isVisibleChatRow).map(rowToMessage));
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
    } finally {
      if (messagesRequestRef.current === controller) messagesRequestRef.current = null;
    }
  }, [setMessages, streamStore]);

  const loadOlderMessages = useCallback(async () => {
    const sessionId = activeSessionRef.current;
    if (!sessionId || historyBeforeSeq === null || historyLoadingOlder) return;
    olderMessagesRequestRef.current?.abort();
    const controller = new AbortController();
    olderMessagesRequestRef.current = controller;
    setHistoryLoadingOlder(true);
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;
    try {
      const page = chatHistoryPage(await fetchChatJson<unknown>(
        `${endpoint}?page_size=50&before_seq=${historyBeforeSeq}`,
        { signal: controller.signal },
      ), endpoint);
      if (activeSessionRef.current !== sessionId) return;
      setMessages((current) => {
        const existingIds = new Set(current.map((message) => message.id));
        const older = page.items
          .filter(isVisibleChatRow)
          .map(rowToMessage)
          .filter((message) => !existingIds.has(message.id));
        return older.length ? [...older, ...current] : current;
      });
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
    } finally {
      if (olderMessagesRequestRef.current === controller) {
        olderMessagesRequestRef.current = null;
        setHistoryLoadingOlder(false);
      }
    }
  }, [historyBeforeSeq, historyLoadingOlder, setMessages]);

  useEffect(() => {
    streamStore.reconcileBaseline(messages);
  }, [messages, streamStore]);

  useEffect(() => () => streamStore.clear(), [streamStore]);

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

  /** 校验网络帧，再更新当前会话的消息投影。 */
  const receiveFrame = useCallback((event: MessageEvent) => {
    console.debug("[chat-ui] ws message", typeof event.data);
    try {
      const frame = parseChatFrame(JSON.parse(String(event.data)));
      if ("session_id" in frame && frame.session_id === activeSessionRef.current) {
        liveProjectionRevisionRef.current += 1;
      }
      const traceKind = traceKindForChatFrame(frame);
      if (traceKind !== undefined && "session_id" in frame && "turn_id" in frame) {
        webTurnTrace.observeFrame(frame.session_id, frame.turn_id, traceKind);
      }
      observeTurnMetrics(turnMetrics, frame);
      applyChatFrame(frame, {
        activeSessionId: () => activeSessionRef.current,
        activateSession: (sessionId) => {
          activeSessionRef.current = sessionId;
          setActiveSessionId(sessionId);
        },
        setError,
        setMessages,
        getStatus: () => statusLiveRef.current,
        setStatus: setStatusLive,
        getActiveTurnId: () => activeTurnIdRef.current,
        setActiveTurnId: (turnId) => { activeTurnIdRef.current = turnId; },
        loadSessions: loadSessionsSafely,
        loadMessages: loadMessagesSafely,
      });
    } catch (error) {
      reportError(error, "error");
    }
  }, [loadMessagesSafely, loadSessionsSafely, reportError, setMessages, setStatusLive, turnMetrics]);

  const closeConnection = useCallback(() => {
    // React 可立即重新挂载；先撤销旧连接，不能等异步 finalizer 才清引用。
    socketRef.current = null;
    if (connectionTaskRef.current) Effect.runFork(Fiber.interrupt(connectionTaskRef.current));
    connectionTaskRef.current = null;
  }, []);

  /** 复用当前连接，或启动一个可整体中断的重连任务。 */
  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return socketRef.current;
    closeConnection();
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${window.location.host}/ws`;
    const first = new WebSocket(url);
    socketRef.current = first;
    // 任务拥有整个重连循环；每次连接结束都先释放 socket 和监听器。
    connectionTaskRef.current = Effect.runFork(Effect.gen(function*() {
      let socket = first;
      while (true) {
        const current = socket;
        const event = yield* Effect.async<CloseEvent>((resume) => {
          current.onmessage = (event) => { if (socketRef.current === current) receiveFrame(event); };
          current.onopen = () => {
            if (socketRef.current !== current) return;
            Effect.runSync(reconnect.reset);
            const sessionId = activeSessionRef.current;
            attachSession(current, sessionId);
            if (sessionId) loadMessagesSafely(sessionId);
          };
          current.onerror = () => current.close();
          current.onclose = (event) => resume(Effect.succeed(event));
        }).pipe(Effect.ensuring(Effect.sync(() => {
          current.onmessage = current.onopen = current.onerror = current.onclose = null;
          if (socketRef.current === current) socketRef.current = null;
          current.close(1000, "connection released");
        })));
        console.warn("[chat-ui] ws close", { code: event.code, reason: event.reason });
        if (event.code !== 1000 && event.code !== 1013) reportError(new Error("聊天连接已关闭"), "error");
        yield* reconnect.next(undefined);
        socket = yield* Effect.sync(() => new WebSocket(url));
        socketRef.current = socket;
      }
    }).pipe(Effect.catchAll(() => Effect.sync(() => reportError(new Error("聊天连接已断开，请刷新页面重试"), "error")))));
    return first;
  }, [closeConnection, loadMessagesSafely, receiveFrame, reconnect, reportError]);

  useEffect(() => {
    connect();
    return closeConnection;
  }, [closeConnection, connect]);

  useEffect(() => {
    // 请求结束后再等待下一次轮询；中断任务会把 AbortSignal 传给 fetch。
    const task = Effect.runFork(Effect.tryPromise({
      try: async (signal) => webShellState(await fetchChatJson<unknown>("/api/shell/state", { signal })),
      catch: (error) => error,
    }).pipe(
      Effect.tap((next) => Effect.sync(() => setShellState(next))),
      Effect.catchAll((error) => Effect.sync(() => reportError(error))),
      Effect.repeat(Schedule.spaced("1200 millis")),
    ));
    return () => { Effect.runFork(Fiber.interrupt(task)); };
  }, [reportError]);

  useEffect(() => {
    if (!chatReady) return;
    void loadSessionsSafely();
    void loadModels("").catch((error: unknown) => reportError(error));
    return () => {
      sessionsRequestRef.current?.abort();
      messagesRequestRef.current?.abort();
      olderMessagesRequestRef.current?.abort();
      modelsRequestRef.current?.abort();
      sendRequestRef.current?.abort();
      stopRequestRef.current?.abort();
    };
  }, [chatReady, loadModels, loadSessionsSafely, reportError]);

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
    const sessionId = `akashic:${createUuid().replaceAll("-", "")}`;
    activeSessionRef.current = sessionId;
    attachSession(socketRef.current, sessionId);
    setActiveSessionId(sessionId);
    return sessionId;
  }, []);

  const sendMessage = useCallback(async (text: string, files: ComposerFile[]) => {
    const cleanText = text.trim();
    if (!cleanText && files.length === 0) return;
    setError("");
    setStatus("submitted");
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    sendRequestRef.current?.abort();
    const controller = new AbortController();
    sendRequestRef.current = controller;
    const clientMessageId = createUuidV7();
    const reply = replyTarget;
    try {
      const sessionId = await ensureSession();
      console.info("[chat-ui] sendMessage", {
        sessionId,
        textLength: cleanText.length,
        files: files.length,
      });
      const media = await uploadFiles(files, controller.signal);
      const attachments = media.map((item) => uploadedFileToAttachment(item));
      setMessages((current) => [
        ...current,
        {
          id: clientMessageId,
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
        request_id: clientMessageId,
        session_id: sessionId,
        text: cleanText,
        media: media.map((item) => item.artifact_id),
      };
      if (reply) payload.reply_to_message_id = reply.id;
      if (modelSelectionDirty) {
        payload.model_runtime_id = selectedRuntimeId;
        payload.model_reasoning_effort = selectedReasoningEffort;
      }
      await sendWhenOpen(connect(), payload, controller.signal);
      console.debug("[chat-ui] send frame delivered", { sessionId });
      setModelSelectionDirty(false);
      setReplyTarget(null);
    } catch (error) {
      setMessages((current) => current.filter((message) => message.id !== clientMessageId));
      if (isAbortError(error)) throw error;
      reportError(error, "error");
      throw error;
    } finally {
      if (sendRequestRef.current === controller) sendRequestRef.current = null;
    }
  }, [connect, ensureSession, modelSelectionDirty, replyTarget, reportError, selectedReasoningEffort, selectedRuntimeId, setMessages]);

  const stopTurn = useCallback(() => {
    if (sendRequestRef.current) {
      sendRequestRef.current.abort();
      if (socketRef.current?.readyState === WebSocket.CONNECTING) {
        closeConnection();
        statusRef.current = "idle";
      }
      setStatus("idle");
      return;
    }
    if (!activeSessionId || stopRequestRef.current) return;
    const controller = new AbortController();
    stopRequestRef.current = controller;
    setStopPending(true);
    void sendWhenOpen(connect(), {
      type: "turn.stop",
      request_id: createUuid(),
      session_id: activeSessionId,
    }, controller.signal)
      .then(() => {
        console.debug("[chat-ui] turn.stop acknowledged", { activeSessionId });
        setStatus("idle");
      })
      .catch((error: unknown) => reportError(error, "error"))
      .finally(() => {
        if (stopRequestRef.current === controller) stopRequestRef.current = null;
        setStopPending(false);
      });
  }, [activeSessionId, closeConnection, connect, reportError]);

  const startNewChat = useCallback(() => {
    setSurface("chat");
    window.history.replaceState(null, "", window.location.pathname);
    activeSessionRef.current = "";
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    modelsRequestRef.current?.abort();
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    setActiveSessionId("");
    setPendingSessionId("");
    setMessages([]);
    setHistoryBeforeSeq(null);
    setHistoryHasMore(false);
    setHistoryLoadingOlder(false);
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
    attachSession(socketRef.current, sessionId);
    olderMessagesRequestRef.current?.abort();
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
  const retry = useCallback(() => {
    setError("");
    void loadSessionsSafely();
    if (shellState?.chatReady) {
      void loadModels(activeSessionRef.current).catch((reason: unknown) => reportError(reason));
    }
  }, [loadModels, loadSessionsSafely, reportError, shellState?.chatReady]);

  return {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, status,
    streamStore, turnMetrics, messageElementsRef, copiedMessageId, shellState, stopPending, modelState,
    historyHasMore, historyLoadingOlder, loadOlderMessages,
    selectedRuntimeId, selectedReasoningEffort, replyTarget, error, mobilePairingOpen,
    activateSession, openRuntime, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, handleModelChange, cancelReply, sendMessage, stopTurn, retry,
    setMobilePairingOpen,
  };
}

export type DesktopChatController = ReturnType<typeof useDesktopChatController>;
