import { mergeTimelineMessages, readMessageLogFrame, type ReplyActivity, type TimelineMessage, type TimelineReply } from "./message-timeline";
import { useCallback, useEffect, useMemo, useRef, useState, type SetStateAction } from "react";
import { createUuid, createUuidV7 } from "./browser-uuid.ts";
import type { ChatMessage } from "./chat-message";
import type { ComposerFile } from "./desktop-composer";
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
  sessionPage,
  uploadFiles,
  webShellState,
  type ChatModelState,
  type SessionRow,
  type WebShellState,
} from "./web-chat-data";
import {
  formatNavigationTime,
  sessionLabel,
  uploadedFileToAttachment,
} from "./web-chat-message-data";
import { parseChatFrame, sendWhenOpen } from "./web-chat-transport";
import { webTurnTrace } from "./web-turn-trace";

function followSession(socket: WebSocket | null, sessionId: string, afterSeq: number): void {
  if (!sessionId || socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "session.follow", version: 2, after_seq: afterSeq,
    request_id: createUuid(),
    session_id: sessionId,
  }));
}

function replyChatStatus(items: ReplyActivity[], pending: number): ChatStatus {
  if (pending) return "submitted";
  if (items.some((item) => item.active)) return "streaming";
  return items.length ? "finalizing" : "idle";
}

export function useDesktopChatController() {
  const [surface, setSurface] = useState<"chat" | "runtime">(
    () => new URLSearchParams(window.location.search).get("surface") === "runtime" ? "runtime" : "chat",
  );
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState("");
  const [streamStore] = useState(() => new StreamProjectionStore<ChatMessage>());
  const [timelineMessages, setTimelineState] = useState<TimelineMessage[]>([]);
  const timelineRef = useRef<TimelineMessage[]>([]);
  const setTimelineMessages = useCallback((next: TimelineMessage[]) => {
    timelineRef.current = next;
    setTimelineState(next);
  }, []);
  const followAfterRef = useRef<number | null>(null);
  const [replyActivities, setReplyActivities] = useState<ReplyActivity[]>([]);
  const replyActivitiesRef = useRef<ReplyActivity[]>([]);
  const [replyAvailable, setReplyAvailable] = useState<boolean | null>(null);
  const [historyThroughSeq, setHistoryThroughSeq] = useState<number | null>(null);
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
  const [replyTarget, setReplyTarget] = useState<TimelineReply | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const [modelState, setModelState] = useState<ChatModelState | null>(null);
  const [selectedRuntimeId, setSelectedRuntimeId] = useState("");
  const [selectedReasoningEffort, setSelectedReasoningEffort] = useState("");
  const [modelSelectionDirty, setModelSelectionDirty] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const connectRef = useRef<(() => WebSocket) | null>(null);
  const messageElementsRef = useRef(new Map<string, HTMLDivElement>());
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");
  const statusLiveRef = useRef<ChatStatus>("idle");
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
      const items = new Map<string, SessionRow>();
      const cursors = new Set<string>();
      let query = "page_size=80";
      while (true) {
        const page = sessionPage(await fetchChatJson<unknown>(`/api/chat/sessions?${query}`, { signal: controller.signal }));
        if (controller.signal.aborted) return;
        page.items.forEach((session) => { if (!items.has(session.key)) items.set(session.key, session); });
        if (!page.nextCursor) break;
        query = `page_size=80&after_time=${encodeURIComponent(page.nextCursor.updated_at)}&after_key=${encodeURIComponent(page.nextCursor.session_id)}`;
        if (cursors.has(query)) throw new Error("会话目录游标未前进");
        cursors.add(query);
      }
      setSessions([...items.values()]);
    } finally {
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null;
    }
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    const controller = new AbortController();
    messagesRequestRef.current = controller;
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;
    try {
      const page = chatHistoryPage(
        await fetchChatJson<unknown>(`${endpoint}?page_size=50`, { signal: controller.signal }),
        endpoint,
      );
      if (
        controller.signal.aborted
        || activeSessionRef.current !== sessionId
      ) return;
      if (page.items.some((row) => row.session_id !== sessionId)) throw new Error("历史页属于其他会话");
      streamStore.clear();
      setMessages([]);
      setTimelineMessages(page.items);
      followAfterRef.current = page.throughSeq;
      setHistoryThroughSeq(page.throughSeq);
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
      followSession(socketRef.current ?? connectRef.current?.() ?? null, sessionId, page.throughSeq);
    } finally {
      if (messagesRequestRef.current === controller) messagesRequestRef.current = null;
    }
  }, [setMessages, setTimelineMessages, streamStore]);

  const loadOlderMessages = useCallback(async () => {
    const sessionId = activeSessionRef.current;
    if (!sessionId || historyBeforeSeq === null || historyThroughSeq === null || historyLoadingOlder) return;
    olderMessagesRequestRef.current?.abort();
    const controller = new AbortController();
    olderMessagesRequestRef.current = controller;
    setHistoryLoadingOlder(true);
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;
    try {
      const page = chatHistoryPage(await fetchChatJson<unknown>(
        `${endpoint}?page_size=50&before_seq=${historyBeforeSeq}&through_seq=${historyThroughSeq}`,
        { signal: controller.signal },
      ), endpoint);
      if (controller.signal.aborted || activeSessionRef.current !== sessionId) return;
      if (page.throughSeq !== historyThroughSeq || page.items.some((row) => row.session_id !== sessionId)) {
        throw new Error("历史页上界或会话发生变化");
      }
      setTimelineMessages(mergeTimelineMessages(timelineRef.current, page.items));
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
    } finally {
      if (olderMessagesRequestRef.current === controller) {
        olderMessagesRequestRef.current = null;
        setHistoryLoadingOlder(false);
      }
    }
  }, [historyBeforeSeq, historyThroughSeq, historyLoadingOlder, setTimelineMessages]);

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

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current !== null) return;
    const attempt = reconnectAttemptRef.current;
    if (attempt >= 12) {
      reportError(new Error("聊天连接已断开，请刷新页面重试"), "error");
      return;
    }
    const ceiling = Math.min(1000 * 2 ** attempt, 30000);
    const delay = Math.floor(Math.random() * ceiling);
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      if (!socketRef.current) connectRef.current?.();
    }, delay);
  }, [reportError]);

  const connect = useCallback(() => {
    const current = socketRef.current;
    if (current && current.readyState <= WebSocket.OPEN) {
      return current;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    console.info("[chat-ui] ws open", socket.url);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return;
      console.debug("[chat-ui] ws message", typeof event.data);
      try {
        const value: unknown = JSON.parse(String(event.data));
        const frame = readMessageLogFrame(value);
        if (frame) {
          if (frame.session_id !== activeSessionRef.current) return;
          if (frame.type === "messages.appended") {
            if (frame.after_seq !== followAfterRef.current) throw new Error("实时消息游标不连续，请重新连接");
            setTimelineMessages(mergeTimelineMessages(timelineRef.current, frame.items));
            followAfterRef.current = frame.next_after_seq;
            const saved = new Set(frame.items.map((item) => item.id));
            setMessages((current) => current.filter((item) => !saved.has(item.id)));
            setStatusLive(replyChatStatus(replyActivitiesRef.current, messagesRef.current.length));
            void loadSessionsSafely();
          } else if (frame.type === "reply.status") {
            replyActivitiesRef.current = frame.items;
            setReplyActivities(frame.items);
            setReplyAvailable(frame.available);
            setStatusLive(replyChatStatus(frame.items, messagesRef.current.length));
          }
          return;
        }
        const other = parseChatFrame(value);
        if (other.type === "error") {
          setMessages((current) => current.filter((item) => item.id !== other.request_id));
          reportError(new Error(other.message), "error");
        }
      } catch (error) {
        reportError(error, "error");
      }
    };
    socket.onopen = () => {
      console.info("[chat-ui] ws connected", socket.url);
      reconnectAttemptRef.current = 0;
      const sessionId = activeSessionRef.current;
      if (sessionId) {
        if (followAfterRef.current === null) void loadMessagesSafely(sessionId);
        else followSession(socket, sessionId, followAfterRef.current);
      }
    };
    socket.onerror = () => {
      if (socketRef.current === socket) socket.close();
    };
    socket.onclose = (event) => {
      if (socketRef.current !== socket) return;
      console.warn("[chat-ui] ws close", { code: event.code, reason: event.reason });
      socketRef.current = null;
      replyActivitiesRef.current = [];
      setReplyActivities([]);
      setReplyAvailable(null);
      if (event.code !== 1000 && event.code !== 1013) {
        reportError(new Error("聊天连接已关闭"), "error");
      }
      scheduleReconnect();
    };
    return socket;
  }, [loadMessagesSafely, loadSessionsSafely, reportError, scheduleReconnect, setMessages, setStatusLive, setTimelineMessages]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

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
    const socket = connect();
    return () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (socketRef.current === socket) socketRef.current = null;
      socket.close(1000, "component unmounted");
    };
  }, [connect]);

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
    if (activeSessionRef.current) {
      const sessionId = activeSessionRef.current;
      if (followAfterRef.current === null) await loadMessages(sessionId);
      return sessionId;
    }
    const sessionId = `akashic:${createUuid().replaceAll("-", "")}`;
    activeSessionRef.current = sessionId;
    followAfterRef.current = -1;
    followSession(socketRef.current, sessionId, -1);
    setActiveSessionId(sessionId);
    return sessionId;
  }, [loadMessages]);

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
      controller.signal.throwIfAborted();
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
      const socket = socketRef.current;
      if (socket?.readyState === WebSocket.CONNECTING) {
        socketRef.current = null;
        statusRef.current = "idle";
        socket.close(1000, "pending send cancelled");
      }
      setStatus("idle");
      return;
    }
    if (!activeSessionId || stopRequestRef.current) return;
    const controller = new AbortController();
    stopRequestRef.current = controller;
    setStopPending(true);
    void sendWhenOpen(connect(), {
      type: "message.send", text: "/stop", media: [],
      request_id: createUuid(),
      session_id: activeSessionId,
    }, controller.signal)
      .then(() => {
        console.debug("[chat-ui] stop request sent", { activeSessionId });
      })
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
    const socket = socketRef.current;
    socketRef.current = null;
    socket?.close(1000, "session changed");
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    modelsRequestRef.current?.abort();
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    setActiveSessionId("");
    setPendingSessionId("");
    setMessages([]);
    setTimelineMessages([]);
    followAfterRef.current = null;
    replyActivitiesRef.current = [];
    setReplyActivities([]);
    setReplyAvailable(null);
    setHistoryThroughSeq(null);
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
  }, [loadModels, reportError, setMessages, setTimelineMessages]);

  const activateSession = useCallback((sessionId: string) => {
    if (surface === "chat" && activeSessionRef.current === sessionId) return;
    setSurface("chat");
    window.history.replaceState(null, "", window.location.pathname);
    activeSessionRef.current = sessionId;
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    const socket = socketRef.current;
    socketRef.current = null;
    socket?.close(1000, "session changed");
    followAfterRef.current = null;
    replyActivitiesRef.current = [];
    setReplyActivities([]);
    setReplyAvailable(null);
    setStatusLive("idle");
    setStopPending(false);
    olderMessagesRequestRef.current?.abort();
    setActiveSessionId(sessionId);
    setPendingSessionId(sessionId);
    setTimelineMessages([]);
    setMessages([]);
    setHistoryBeforeSeq(null);
    setHistoryHasMore(false);
    setReplyTarget(null);
    setModelState(null);
    setSelectedRuntimeId("");
    setModelSelectionDirty(false);
    void Promise.all([loadMessages(sessionId), loadModels(sessionId)])
      .catch((reason: unknown) => reportError(reason))
      .finally(() => {
        if (activeSessionRef.current === sessionId) setPendingSessionId("");
      });
  }, [loadMessages, loadModels, reportError, setMessages, setTimelineMessages, setStatusLive, surface]);

  const openRuntime = useCallback(() => {
    setSurface("runtime");
    window.history.replaceState(null, "", `${window.location.pathname}?surface=runtime`);
  }, []);
  const handleReplyMessage = useCallback((reply: TimelineReply) => setReplyTarget(reply), []);
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
    const socket = socketRef.current;
    socketRef.current = null;
    socket?.close(1000, "retry");
    replyActivitiesRef.current = [];
    setReplyActivities([]);
    setReplyAvailable(null);
    const sessionId = activeSessionRef.current;
    if (sessionId && timelineRef.current.length) void loadMessagesSafely(sessionId);
    else connect();
    void loadSessionsSafely();
    if (shellState?.chatReady) {
      void loadModels(activeSessionRef.current).catch((reason: unknown) => reportError(reason));
    }
  }, [connect, loadMessagesSafely, loadModels, loadSessionsSafely, reportError, shellState?.chatReady]);

  return {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, timelineMessages, replyActivities, replyAvailable, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending, modelState,
    historyHasMore, historyLoadingOlder, loadOlderMessages,
    selectedRuntimeId, selectedReasoningEffort, replyTarget, error, mobilePairingOpen,
    activateSession, openRuntime, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, handleModelChange, cancelReply, sendMessage, stopTurn, retry,
    setMobilePairingOpen,
  };
}

export type DesktopChatController = ReturnType<typeof useDesktopChatController>;
