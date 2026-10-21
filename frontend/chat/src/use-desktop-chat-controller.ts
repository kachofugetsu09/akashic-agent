import { mergeTimelineMessages, readMessageLogFrame, type ReplyActivity, type TimelineMessage, type TimelineReply } from "./message-timeline";
import { useCallback, useEffect, useMemo, useRef, useState, type SetStateAction } from "react";
import { createUuid, createUuidV7 } from "./browser-uuid.ts";
import type { ChatMessage } from "./chat-message";
import type { ComposerFile } from "./desktop-composer";
import { loadWebPluginCatalog, receivePluginUiEvent, resetPluginUiEvents, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { groupKey, useNavigationGroupSections, type GroupRef, type NavigationGroupsState } from "./navigation-groups";
import { useNavigationPins } from "./use-navigation-pins";
import { readSessionActivityFrame, useSessionActivity, type SessionHead } from "./use-session-activity";
import { StreamProjectionStore } from "./stream-projection";
import { canProjectWebStreamWithoutRoot, publishWebStreamChanges } from "./web-stream-projection";
import { isGeneratingChatStatus, replyChatStatus, type ChatStatus } from "./web-chat-status";
import { reportShellSession } from "./shell-rail-actions";
import {
  ChatRequestError,
  chatHistoryPage,
  errorMessage,
  fetchChatJson,
  isAbortError,
  sessionPage,
  renameChatSession,
  setChatSessionDeleted,
  uploadFiles,
  webShellState,
  type SessionRow,
  type WebShellState,
} from "./web-chat-data";
import {
  sessionLabel,
  uploadedFileToAttachment,
} from "./web-chat-message-data";
import { parseChatFrame, sendWhenOpen } from "./web-chat-transport";
import type { DesktopSidebarSession } from "./desktop-sidebar";
import { useSteers, type SteerResult } from "./use-steers";
import { webTurnTrace } from "./web-turn-trace";
import { composerContributions, notifyComposerSent, useComposerContributions } from "./composer-contributions";

/** 回访会话时先展示的尾页快照；只做展示缓存，事实仍以服务端分页与实时帧为准。 */
interface SessionTail {
  items: TimelineMessage[];
  throughSeq: number;
  beforeSeq: number | null;
  hasMore: boolean;
  deleted: boolean;
  fetchedAt: number;
}

const SESSION_TAIL_CACHE_LIMIT = 8;
const RECONNECT_MAX_DELAY_MS = 30_000;
// 软删后的恢复入口：窗口关闭后 UI 不再提供恢复路径（无回收站视图）。
const DELETED_NOTICE_MS = 10_000;
// 新鲜快照激活时不再重复拉尾页；超过窗口或实时跟随断档仍回到权威分页。
const SESSION_TAIL_FRESH_MS = 30_000;
// 弱网下连接可能半开：socket.send 成功也不代表服务端收到。心跳发现死连接，发送以服务端确认为准，
// 超时换新连接用同一 request_id 重发（服务端按 provider message id 去重）。
const HEARTBEAT_INTERVAL_MS = 25_000;
const HEARTBEAT_TIMEOUT_MS = 10_000;
const SEND_ACK_TIMEOUT_MS = 10_000;
const SEND_ATTEMPTS = 3;

/** 连接任务被中断时等待/sleep 抛出的记号；外层 catch 识别后直接退出，不提示连接错误。 */
const CONNECTION_ABORTED = Symbol("connection-aborted");

/** 可被 AbortSignal 中断的睡眠；中断时 reject 出 CONNECTION_ABORTED 记号。 */
function sleepInterruptible(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(CONNECTION_ABORTED); return; }
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(CONNECTION_ABORTED);
    };
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * 原生重连退避驱动，等价原 Schedule.exponential(1s) 封顶 30s、jitteredWith(0..1)、
 * intersect(recurs(12))：第 n 次基础延迟 min(1000*2^n, 封顶) 再乘 [0,1) 随机抖动；
 * 成功 12 次后第 13 次立即 reject；reset 把计数归零。
 */
function createReconnectScheduler(maxDelayMs: number, signal: AbortSignal): { reset(): void; next(): Promise<void> } {
  let attempts = 0;
  return {
    reset(): void { attempts = 0; },
    next(): Promise<void> {
      const attempt = attempts;
      if (attempt >= 12) return Promise.reject(new Error("重连次数已用尽"));
      attempts = attempt + 1;
      const base = Math.min(1000 * 2 ** attempt, maxDelayMs);
      return sleepInterruptible(base * Math.random(), signal);
    },
  };
}

// 服务端明确拒绝的发送不重试，直接把原因交给用户。
class SendRejected extends Error {}

function followSession(socket: WebSocket | null, sessionId: string, afterSeq: number): void {
  if (!sessionId || socket?.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: "session.follow", version: 2, after_seq: afterSeq,
    request_id: createUuid(),
    session_id: sessionId,
  }));
}

/** 切换会话只清理一次性路由参数，保留嵌入模式和其他页面选项。 */
function clearChatRoute() {
  const url = new URL(window.location.href);
  url.searchParams.delete("session");
  url.searchParams.delete("surface");
  window.history.replaceState(null, "", url);
}

/** 重发用尽仍未送达：属于连接问题，连接恢复后提示自动撤下。 */
class DeliveryFailed extends Error {}
// 发送（上传附件或投递）失败：草稿与附件都还在输入框里，发送钮本身就是重试入口。
class SendFailed extends Error {}
// 历史没读到：原因由对话区的读取失败视图持有，不写全局错误（否则重试期间旧原因掉到输入框下，切会话还会残留）。
class HistoryUnavailable extends Error {}

const RECONNECTING_NOTICE = "连接已断开，正在重新连接…";

function sameScope(a: Readonly<Record<string, string>> | undefined, b: Readonly<Record<string, string>> | undefined): boolean {
  if (a === b) return true;
  const aEntries = Object.entries(a ?? {});
  const bRecord = b ?? {};
  const bEntries = Object.entries(bRecord);
  return aEntries.length === bEntries.length && aEntries.every(([key, value]) => bRecord[key] === value);
}

/** 目录行的结构共享缓存：字段全等的会话跨渲染复用同一对象，侧栏行 memo 才能命中。 */
const sidebarRowCache = new Map<string, DesktopSidebarSession>();

function shareSidebarRows(rows: DesktopSidebarSession[]): DesktopSidebarSession[] {
  const shared = rows.map((row) => {
    const kept = sidebarRowCache.get(row.id);
    const reusable = kept && kept.title === row.title && kept.preview === row.preview
      && kept.updatedAt === row.updatedAt && kept.createdAt === row.createdAt
      && kept.active === row.active && kept.status === row.status && kept.scoped === row.scoped
      && sameScope(kept.scope, row.scope);
    return reusable ? kept : row;
  });
  sidebarRowCache.clear();
  for (const row of shared) sidebarRowCache.set(row.id, row);
  return shared;
}
// 页面版本：挂载入口给出构建时写入的版本；夹具页面没有，不做版本核对。
let pageBuild: string | null = null;

export function setPageBuild(build: string): void {
  pageBuild = build;
}

const PAGE_BUILD_RETRY_MS = [2_000, 5_000, 15_000];

function pageBuildOutdated(state: WebShellState): boolean {
  return Boolean(pageBuild && state.build && state.build !== pageBuild);
}

// 服务端升级后旧页面的展示合同可能已过期：每次连上都对照页面自带的版本，变了就提示刷新。
// 读取失败（重启中的短暂 503 等）按退避有限重试；live 为假时本次连接已结束，不再重试。
function checkPageBuild(outdated: () => void, live: () => boolean, attempt = 0) {
  if (!pageBuild) return;
  void fetchChatJson<unknown>("/api/shell/state").then(webShellState).then((state) => {
    if (pageBuildOutdated(state)) outdated();
  }).catch(() => {
    const delay = PAGE_BUILD_RETRY_MS[attempt];
    if (delay !== undefined) window.setTimeout(() => { if (live()) checkPageBuild(outdated, live, attempt + 1); }, delay);
  });
}

export function useDesktopChatController() {
  const [surface, setSurface] = useState<"chat" | "runtime">(
    () => new URLSearchParams(window.location.search).get("surface") === "runtime" ? "runtime" : "chat",
  );
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [newChatGroup, setNewChatGroup] = useState<GroupRef | null>(null);
  // Session 宽键在首条消息接纳时固定；之后每次发送重复声明只是幂等核对。
  const sessionScopesRef = useRef(new Map<string, Record<string, string>>());
  const newChatScopeRef = useRef<Record<string, string> | null>(null);
  const pluginCatalog = usePluginUiCatalogVersion();
  const [activeSessionId, setActiveSessionId] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState("");
  // 通知入口统一使用网页导航，会话参数在 chatReady 后消费一次。
  const [requestedSessionId, setRequestedSessionId] = useState(
    () => {
      const requested = new URLSearchParams(window.location.search).get("session");
      if (requested !== null) return requested;
      try { return sessionStorage.getItem("akashic.chat.active-session") ?? ""; }
      catch (error) { if (!(error instanceof DOMException)) throw error; return ""; }
    },
  );
  const [streamStore] = useState(() => new StreamProjectionStore<ChatMessage>());
  const [timelineRefresh, setTimelineRefresh] = useState(0);
  const [timelineMessages, setTimelineState] = useState<TimelineMessage[]>([]);
  const timelineRef = useRef<TimelineMessage[]>([]);
  const setTimelineMessages = useCallback((next: TimelineMessage[]) => {
    timelineRef.current = next;
    setTimelineState(next);
  }, []);
  const followAfterRef = useRef<number | null>(null);
  const [followAfter, setFollowAfterState] = useState<number | null>(null);
  const setFollowAfter = useCallback((next: number | null) => {
    followAfterRef.current = next;
    setFollowAfterState(next);
  }, []);
  const [replyActivities, setReplyActivities] = useState<ReplyActivity[]>([]);
  const replyActivitiesRef = useRef<ReplyActivity[]>([]);
  const [replyAvailable, setReplyAvailableState] = useState<boolean | null>(null);
  const replyAvailableRef = useRef<boolean | null>(null);
  const setReplyAvailable = useCallback((next: boolean | null) => {
    replyAvailableRef.current = next;
    setReplyAvailableState(next);
  }, []);
  const [historyThroughSeq, setHistoryThroughSeq] = useState<number | null>(null);
  const [messages, setMessagesState] = useState<ChatMessage[]>([]);
  const [historyBeforeSeq, setHistoryBeforeSeq] = useState<number | null>(null);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [activeSessionDeleted, setActiveSessionDeleted] = useState(false);
  const [deletedNotice, setDeletedNotice] = useState<{ key: string; title: string } | null>(null);
  const deletedNoticeTimerRef = useRef(0);
  const [historyLoadingOlder, setHistoryLoadingOlder] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  // 历史没读到的会话：对话区写明原因并给重试，不退回新会话的空白页（那会让人以为这段对话是空的，接着发也看不到上文）。
  const [historyFailure, setHistoryFailure] = useState<{ sessionId: string; reason: string } | null>(null);
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
  const [errorFromSend, setErrorFromSend] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [pageOutdated, setPageOutdated] = useState(false);
  const [shellState, setShellState] = useState<WebShellState | null>(null);
  const [replyTarget, setReplyTarget] = useState<TimelineReply | null>(null);
  // 引用目标与草稿一样按会话保留：切去别的会话查点东西再回来，引用条还在。
  const replyTargetsRef = useRef(new Map<string, TimelineReply>());
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const socketRef = useRef<WebSocket | null>(null);
  const connectionTaskRef = useRef<AbortController | null>(null);
  const connectRef = useRef<(() => WebSocket) | null>(null);
  const messageElementsRef = useRef(new Map<string, HTMLDivElement>());
  const activeSessionRef = useRef("");
  const statusRef = useRef<ChatStatus>("idle");
  const statusLiveRef = useRef<ChatStatus>("idle");
  const sessionsRequestRef = useRef<AbortController | null>(null);
  const messagesRequestRef = useRef<AbortController | null>(null);
  const olderMessagesRequestRef = useRef<AbortController | null>(null);
  const sendRequestRef = useRef<AbortController | null>(null);
  const stopRequestRef = useRef<AbortController | null>(null);
  const ackWaitersRef = useRef(new Map<string, (error?: Error) => void>());
  const steerResultRef = useRef<(requestId: string, result: SteerResult) => void>(() => undefined);
  const probeRef = useRef<(() => void) | null>(null);
  const tailCacheRef = useRef(new Map<string, SessionTail>());
  const tailRequestsRef = useRef(new Map<string, { controller: AbortController; promise: Promise<SessionTail> }>());
  const chatReady = shellState?.chatReady === true;
  const navigationPins = useNavigationPins(chatReady);
  const reloadPins = navigationPins.reload;
  const reloadPinsRef = useRef(reloadPins);
  useEffect(() => { reloadPinsRef.current = reloadPins; }, [reloadPins]);

  useEffect(() => reportShellSession(activeSessionId), [activeSessionId]);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
    // 在恢复已有会话前，不用初始化空值擦掉标签页导航位置。
    if (requestedSessionId) return;
    try {
      if (activeSessionId) sessionStorage.setItem("akashic.chat.active-session", activeSessionId);
      else sessionStorage.removeItem("akashic.chat.active-session");
    } catch (error) { if (!(error instanceof DOMException)) throw error; }

  }, [activeSessionId, requestedSessionId]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // 当前错误是否只是"断线导致没送达"：连接恢复后自动撤下，草稿仍在输入框里。
  const deliveryErrorRef = useRef(false);
  const reportError = useCallback((error: unknown, nextStatus?: ChatStatus): void => {
    if (isAbortError(error)) return;
    console.error("[chat] request failed", error);
    deliveryErrorRef.current = error instanceof DeliveryFailed;
    setErrorFromSend(error instanceof DeliveryFailed || error instanceof SendFailed);
    setError(errorMessage(error));
    if (nextStatus) { statusLiveRef.current = nextStatus; setStatus(nextStatus); }
  }, []);

  // LRU 写回；capacity 足够覆盖最近几个会话，超出后淘汰最久未命中项。
  const cacheSessionTail = useCallback((sessionId: string, tail: SessionTail) => {
    const cache = tailCacheRef.current;
    cache.delete(sessionId);
    cache.set(sessionId, tail);
    while (cache.size > SESSION_TAIL_CACHE_LIMIT) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
  }, []);

  // 预取与正式激活共享尾页读取；缓存被分页/实时帧更新后，旧快照不能覆盖它。
  const readSessionTail = useCallback((sessionId: string) => {
    const pending = tailRequestsRef.current.get(sessionId);
    if (pending && !pending.controller.signal.aborted) return pending;
    const controller = new AbortController();
    const previous = tailCacheRef.current.get(sessionId);
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`;
    const promise = fetchChatJson<unknown>(`${endpoint}?page_size=50`, { signal: controller.signal })
      .then((payload) => {
        const page = chatHistoryPage(payload, endpoint);
        if (page.items.some((row) => row.session_id !== sessionId)) throw new Error("历史页属于其他会话");
        const tail = { items: page.items, throughSeq: page.throughSeq,
          beforeSeq: page.beforeSeq, hasMore: page.hasMore, deleted: page.deleted, fetchedAt: Date.now() };
        if (!controller.signal.aborted && tailCacheRef.current.get(sessionId) === previous) {
          cacheSessionTail(sessionId, tail);
        }
        return tailCacheRef.current.get(sessionId) ?? tail;
      })
      .finally(() => {
        if (tailRequestsRef.current.get(sessionId)?.controller === controller) tailRequestsRef.current.delete(sessionId);
      });
    const request = { controller, promise };
    tailRequestsRef.current.set(sessionId, request);
    return request;
  }, [cacheSessionTail]);

  // 预取失败由正式激活路径报告；已经结束的失败不会阻止下一次读取。
  const prefetchSessionTail = useCallback((sessionId: string) => {
    if (!sessionId || tailCacheRef.current.has(sessionId)) return;
    void readSessionTail(sessionId).promise.catch(() => undefined);
  }, [readSessionTail]);

  const loadSessions = useCallback(async () => {
    sessionsRequestRef.current?.abort();
    const controller = new AbortController();
    sessionsRequestRef.current = controller;
    try {
      const items = new Map<string, SessionRow>();
      const cursors = new Set<string>();
      let query = "page_size=200";
      while (true) {
        const page = sessionPage(await fetchChatJson<unknown>(`/api/chat/sessions?${query}`, { signal: controller.signal }));
        if (controller.signal.aborted) return;
        page.items.forEach((session) => { if (!items.has(session.key)) items.set(session.key, session); });
        if (!page.nextCursor) break;
        query = `page_size=200&after_time=${encodeURIComponent(page.nextCursor.updated_at)}&after_key=${encodeURIComponent(page.nextCursor.session_id)}`;
        if (cursors.has(query)) throw new Error("会话目录游标未前进");
        cursors.add(query);
      }
      items.forEach((session) => {
        if (session.scope && Object.keys(session.scope).length) sessionScopesRef.current.set(session.key, session.scope);
      });
      setSessions([...items.values()]);
    } finally {
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null;
    }
  }, []);

  const loadMessages = useCallback(async (sessionId: string) => {
    const request = readSessionTail(sessionId);
    const { controller } = request;
    if (messagesRequestRef.current !== controller) messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    messagesRequestRef.current = controller;
    setHistoryLoading(true);
    setHistoryFailure(null);
    const current = () => messagesRequestRef.current === controller
      && !controller.signal.aborted && activeSessionRef.current === sessionId;
    try {
      const page = await request.promise;
      if (!current()) return;
      streamStore.clear();
      const saved = new Set(page.items.map((item) => item.id));
      setMessages((current) => sendRequestRef.current ? current.filter((item) => !saved.has(item.id)) : []);
      setTimelineMessages(page.items);
      setActiveSessionDeleted(page.deleted);
      setTimelineRefresh((revision) => revision + 1);
      if (statusLiveRef.current !== "uploading") setStatusLive(replyChatStatus(replyActivitiesRef.current, messagesRef.current.length,
        page.items, replyAvailableRef.current));
      setFollowAfter(page.throughSeq);
      setHistoryThroughSeq(page.throughSeq);
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
      followSession(socketRef.current ?? connectRef.current?.() ?? null, sessionId, page.throughSeq);
    } catch (error) {
      if (!current()) return;
      // 会话已不存在（404）时重试没有意义，照旧回到首页版式开新对话。
      if (isAbortError(error) || (error instanceof ChatRequestError && error.status === 404)) throw error;
      console.error("[chat] history request failed", error);
      const reason = errorMessage(error);
      setHistoryFailure({ sessionId, reason });
      throw new HistoryUnavailable(reason);
    } finally {
      if (messagesRequestRef.current === controller) {
        messagesRequestRef.current = null;
        setHistoryLoading(false);
      }
    }
  }, [readSessionTail, setFollowAfter, setMessages, setStatusLive, setTimelineMessages, streamStore]);

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
      const merged = mergeTimelineMessages(timelineRef.current, page.items);
      setTimelineMessages(merged);
      const cached = tailCacheRef.current.get(sessionId);
      if (cached) cacheSessionTail(sessionId, { ...cached, items: merged, beforeSeq: page.beforeSeq, hasMore: page.hasMore });
      setHistoryBeforeSeq(page.beforeSeq);
      setHistoryHasMore(page.hasMore);
    } catch (error) {
      if (olderMessagesRequestRef.current === controller
        && !controller.signal.aborted && activeSessionRef.current === sessionId) throw error;
    } finally {
      if (olderMessagesRequestRef.current === controller) {
        olderMessagesRequestRef.current = null;
        setHistoryLoadingOlder(false);
      }
    }
  }, [cacheSessionTail, historyBeforeSeq, historyThroughSeq, historyLoadingOlder, setTimelineMessages]);

  useEffect(() => {
    streamStore.reconcileBaseline(messages);
  }, [messages, streamStore]);

  useEffect(() => () => streamStore.clear(), [streamStore]);

  const loadSessionsSafely = useCallback(() => loadSessions().catch((error: unknown) => reportError(error)), [loadSessions, reportError]);
  // 实时帧驱动的目录刷新很密（启动基线帧、流式追加、活动状态都触发）：合并成尾随一次执行，
  // 避免每个帧中止重发一次全量目录读。显式数据操作（改名/删除）仍立即刷新，不走这里。
  const sessionsRefreshTimerRef = useRef<number | null>(null);
  const scheduleSessionsRefresh = useCallback((): Promise<void> => {
    if (sessionsRefreshTimerRef.current !== null) return Promise.resolve();
    return new Promise((resolve) => {
      sessionsRefreshTimerRef.current = window.setTimeout(() => {
        sessionsRefreshTimerRef.current = null;
        resolve(loadSessionsSafely());
      }, 300);
    });
  }, [loadSessionsSafely]);
  const sessionHeads = useMemo<SessionHead[]>(() => {
    const rows = new Map([...navigationPins.sessions, ...sessions].map((session) => [session.key, session]));
    return [...rows.values()].map((session) => ({ key: session.key, headSeq: session.head_seq }));
  }, [sessions, navigationPins.sessions]);
  const { statuses: sessionStatuses, receive: receiveSessionActivity, disconnect: disconnectSessionActivity } = useSessionActivity(
    sessionHeads, activeSessionId, historyLoading ? null : followAfter,
    surface === "chat", scheduleSessionsRefresh,
  );
  // 读取失败已由读取失败视图展示，其它错误照常进全局错误行。
  const reportHistoryError = useCallback((error: unknown) => {
    if (!(error instanceof HistoryUnavailable)) reportError(error);
  }, [reportError]);
  const loadMessagesSafely = useCallback((sessionId: string) => loadMessages(sessionId).catch(reportHistoryError), [loadMessages, reportHistoryError]);

  // 输入框工具位的插件可以阻止发送（例如所选模型不可用）；宿主只显示原因。
  const { blockReason } = useComposerContributions();
  const sendProblem = blockReason;
  const canSend = chatReady && !sendProblem && !activeSessionDeleted;

  const closeConnection = useCallback(() => {
    // React 可立即重新挂载；先撤销旧连接，不能等异步 finalizer 才清引用。
    socketRef.current = null;
    resetPluginUiEvents();
    connectionTaskRef.current?.abort();
    connectionTaskRef.current = null;
    disconnectSessionActivity();
  }, [disconnectSessionActivity]);

  /** 复用当前连接，或启动一个可整体中断的重连任务。 */
  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return socketRef.current;
    closeConnection();
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${window.location.host}/ws?watch_sessions=true`;
    const first = new WebSocket(url);
    socketRef.current = first;
    const controller = new AbortController();
    connectionTaskRef.current = controller;
    // 退避驱动绑定这一代连接的固定信号：不复读 connectionTaskRef,abort 与 ref 置空之间没有可乘的间隙。
    const reconnect = createReconnectScheduler(RECONNECT_MAX_DELAY_MS, controller.signal);
    const run = async (): Promise<void> => {
      let socket = first;
      try {
        while (true) {
          const current = socket;
          let openedAt: number | null = null;
          let heartbeat: number | undefined;
          let probeTimer: number | undefined;
          let probe: (() => void) | null = null;
          let event: CloseEvent;
          try {
            // 等待关闭事件：正常 close、心跳超时合成事件或被中断，三者只有一个兑现。
            event = await new Promise<CloseEvent>((resolve, reject) => {
              const onAbort = () => reject(CONNECTION_ABORTED);
              controller.signal.addEventListener("abort", onAbort, { once: true });
              const finish = (closeEvent: CloseEvent) => {
                controller.signal.removeEventListener("abort", onAbort);
                resolve(closeEvent);
              };
              // 任何入站帧都证明连接活着；ping 超时不等浏览器的关闭握手，直接按断线进入重连。
              probe = () => {
                if (current.readyState !== WebSocket.OPEN || probeTimer !== undefined) return;
                current.send(JSON.stringify({ type: "ping", request_id: createUuid() }));
                probeTimer = window.setTimeout(() => {
                  console.warn("[chat-ui] ws heartbeat timeout");
                  finish(new CloseEvent("close", { code: 4000, reason: "heartbeat timeout" }));
                }, HEARTBEAT_TIMEOUT_MS);
              };
              current.onmessage = (event) => {
                if (socketRef.current !== current) return;
                window.clearTimeout(probeTimer);
                probeTimer = undefined;
                console.debug("[chat-ui] ws message", typeof event.data);
                try {
                  const value: unknown = JSON.parse(String(event.data));
                  const activity = readSessionActivityFrame(value);
                  if (activity) {
                    receiveSessionActivity(activity);
                    return;
                  }
                  if (receivePluginUiEvent(value)) return;
                  const frame = readMessageLogFrame(value);
                  if (frame) {
                    if (frame.session_id !== activeSessionRef.current) return;
                    if (frame.type === "session.following") {
                      setTimelineRefresh((revision) => revision + 1);
                    } else if (frame.type === "messages.appended") {
                      if (frame.after_seq !== followAfterRef.current) throw new Error("实时消息游标不连续，请重新连接");
                      const merged = mergeTimelineMessages(timelineRef.current, frame.items);
                      setTimelineMessages(merged);
                      const cached = tailCacheRef.current.get(frame.session_id);
                      if (cached) cacheSessionTail(frame.session_id, { ...cached, items: merged, throughSeq: frame.next_after_seq, fetchedAt: Date.now() });
                      setFollowAfter(frame.next_after_seq);
                      const saved = new Set(frame.items.map((item) => item.id));
                      for (const id of saved) ackWaitersRef.current.get(id)?.();
                      setMessages((currentMessages) => currentMessages.filter((item) => !saved.has(item.id)));
                      if (statusLiveRef.current !== "uploading") setStatusLive(replyChatStatus(replyActivitiesRef.current, messagesRef.current.length + Number(sendRequestRef.current !== null), timelineRef.current, replyAvailableRef.current));
                      scheduleSessionsRefresh();
                    } else if (frame.type === "reply.status") {
                      replyActivitiesRef.current = frame.items;
                      setReplyActivities(frame.items);
                      setReplyAvailable(frame.available);
                      if (statusLiveRef.current !== "uploading") setStatusLive(replyChatStatus(frame.items, messagesRef.current.length + Number(sendRequestRef.current !== null), timelineRef.current, frame.available));
                    }
                    return;
                  }
                  const other = parseChatFrame(value);
                  if (other.type === "sessions.changed") {
                    scheduleSessionsRefresh();
                    void reloadPinsRef.current();
                  } else if (other.type === "message.accepted") {
                    ackWaitersRef.current.get(other.request_id)?.();
                  } else if (other.type === "message.steer.result") {
                    steerResultRef.current(other.request_id, other);
                  } else if (other.type === "error") {
                    const waiter = ackWaitersRef.current.get(other.request_id);
                    if (waiter) {
                      waiter(new SendRejected(other.message));
                      return;
                    }
                    setMessages((currentMessages) => currentMessages.filter((item) => item.id !== other.request_id));
                    reportError(new Error(other.message), "error");
                  }
                } catch (error) {
                  reportError(error, "error");
                }
              };
              current.onopen = () => {
                if (socketRef.current !== current) return;
                openedAt = performance.now();
                heartbeat = window.setInterval(() => probe?.(), HEARTBEAT_INTERVAL_MS);
                probeRef.current = probe;
                setConnectionError("");
                if (deliveryErrorRef.current) {
                  deliveryErrorRef.current = false;
                  setError("");
                }
                console.info("[chat-ui] ws connected", current.url);
                checkPageBuild(() => setPageOutdated(true), () => socketRef.current === current);
                const sessionId = activeSessionRef.current;
                if (sessionId) {
                  if (followAfterRef.current === null) void loadMessagesSafely(sessionId);
                  else followSession(current, sessionId, followAfterRef.current);
                }
              };
              current.onerror = () => current.close();
              current.onclose = (closeEvent) => {
                // 短连仍属于同一次中断，不能凭握手成功刷新退避和预算。
                if (openedAt !== null && performance.now() - openedAt >= RECONNECT_MAX_DELAY_MS) {
                  reconnect.reset();
                }
                finish(closeEvent);
              };
            });
          } finally {
            // 对应原 Effect.ensuring：关闭事件返回与连接任务被中断，两种结束都执行且只执行一次。
            current.onmessage = current.onopen = current.onerror = current.onclose = null;
            window.clearInterval(heartbeat);
            window.clearTimeout(probeTimer);
            if (probeRef.current === probe) probeRef.current = null;
            if (socketRef.current === current) socketRef.current = null;
            current.close(1000, "connection released");
          }
          console.warn("[chat-ui] ws close", { code: event.code, reason: event.reason });
          replyActivitiesRef.current = [];
          disconnectSessionActivity();
          setReplyActivities([]);
          setReplyAvailable(null);
          // 插件推送的会话状态属于这一代连接：断线即失效，重连后的基线帧再装回。
          resetPluginUiEvents();
          setStatusLive(replyChatStatus([], messagesRef.current.length, [], false));
          if (event.code !== 1000 && event.code !== 1013) setConnectionError(RECONNECTING_NOTICE);
          await reconnect.next();
          // 退避结束后 abort 可能刚好落在微任务间隙：建 socket 前再核一次,避免给已撤销的一代接出活连接。
          if (controller.signal.aborted) return;
          socket = new WebSocket(url);
          socketRef.current = socket;
        }
      } catch (error) {
        // 中断记号：连接已被 closeConnection 撤销，对应原 Fiber.interrupt——不走 catchAll，不提示错误。
        if (error === CONNECTION_ABORTED) return;
        // 对应原 Effect.catchAll：退避预算耗尽等失败统一落到错误行。
        setConnectionError("暂时无法连接，请重试");
      }
    };
    // 兜底:finally/catch 内部一旦抛错(如组件已卸载后的 setState 边界),不外泄成 unhandled rejection。
    void run().catch((error) => console.error("[chat-ui] connection loop escaped", error));
    return first;
  }, [cacheSessionTail, closeConnection, disconnectSessionActivity, loadMessagesSafely, scheduleSessionsRefresh,
    receiveSessionActivity, reportError, setFollowAfter, setMessages, setReplyAvailable, setStatusLive, setTimelineMessages]);

  // 手机从后台回来或网络切换后，旧连接常常已经半开；立刻探测一次，不等下一轮心跳。
  // 已断开、正在退避等待时直接重连并重置退避：网络回来了不必再等最长 30 秒。
  useEffect(() => {
    const check = () => {
      if (document.visibilityState !== "visible") return;
      if (probeRef.current) probeRef.current();
      else if (connectionTaskRef.current && !socketRef.current) connect();
    };
    window.addEventListener("online", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      window.removeEventListener("online", check);
      document.removeEventListener("visibilitychange", check);
    };
  }, [connect]);

  useEffect(() => {
    // 只等待首次启动就绪；此后的断线与恢复由聊天连接负责。
    // 就绪发现延迟 = 轮询间隔：固定 1.2s 时平均白等 600ms（冷启动尾段正是等它，见 #1556）。
    // 在首个 1.2s 窗口内加插 150/450/1050ms 三次早探（之后节奏与原调度逐点相同），
    // 任何就绪时刻都不比原调度更差，期望发现延迟 600ms → ~200ms；代价只是启动期三个小包。
    const READY_POLL_BACKOFF_MS = [150, 300, 600, 150];
    const controller = new AbortController();
    const run = async (): Promise<void> => {
      let attempt = 0;
      while (true) {
        try {
          const next = await fetchChatJson<unknown>("/api/shell/state", { signal: controller.signal }).then(webShellState);
          setShellState(next);
          if (pageBuildOutdated(next)) setPageOutdated(true);
          if (next?.chatReady === true) return;
        } catch (error) {
          // 中断直接退出（对应原 Fiber.interrupt 不进 catchAll）；其余失败照原样上报后继续下一轮。
          if (controller.signal.aborted || error === CONNECTION_ABORTED) return;
          reportError(error);
        }
        try {
          const delay = attempt < READY_POLL_BACKOFF_MS.length ? READY_POLL_BACKOFF_MS[attempt] : 1200;
          attempt += 1;
          await sleepInterruptible(delay, controller.signal);
        } catch {
          // 中断（含睡眠中）：轮询结束，不再上报。
          return;
        }
      }
    };
    void run();
    return () => { controller.abort(); };
  }, [reportError]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  // 只有真正换代（已有非空目录变成新修订）才重建传输：旧代推送状态随之失效、
  // 基线帧重取。首次目录到达（"" → 首个修订）时连接上的基线帧仍然有效——插件推送按
  // (plugin, revision, session) 键入 pushed，订阅端自取最新值——这时重建只是重复一轮
  // 握手、基线订阅与目录取证，慢链路下每次启动白付一整波重复请求与 RTT。
  // 卸载断连由独立的 cleanup 负责，目录到达不再经过 effect 反跑误杀健康连接。
  const catalogRevisionRef = useRef<string | null>(null);
  useEffect(() => {
    const revision = pluginCatalog.catalogRevision;
    const previous = catalogRevisionRef.current;
    catalogRevisionRef.current = revision;
    if (previous !== null && previous !== "" && previous !== revision) closeConnection();
    if (!socketRef.current && !connectionTaskRef.current) connect();
  }, [closeConnection, connect, pluginCatalog.catalogRevision]);

  useEffect(() => () => closeConnection(), [closeConnection]);

  // 就绪后继续等待在途请求；只有旧请求失败才补发一次，成功结果直接复用。
  const startupRequestsRef = useRef<Partial<Record<"sessions", Promise<void>>>>({});
  useEffect(() => {
    let cancelled = false;
    const run = async (key: "sessions", task: () => Promise<void>): Promise<void> => {
      const previous = startupRequestsRef.current[key];
      const request = previous ?? task();
      startupRequestsRef.current[key] = request;
      try {
        await request;
      } catch (error) {
        if (cancelled) return;
        delete startupRequestsRef.current[key];
        if (!chatReady || isAbortError(error)) return;
        if (previous) await run(key, task);
        else reportError(error);
      }
    };
    void run("sessions", loadSessions);
    return () => { cancelled = true; };
  }, [chatReady, loadSessions, reportError]);

  // ?session= 直达会话时同步预热尾页缓存；激活仍等 chatReady，命中后立即可见。
  useEffect(() => {
    if (requestedSessionId) prefetchSessionTail(requestedSessionId);
  }, [prefetchSessionTail, requestedSessionId]);

  // 健康探测暂时失败不取消用户已经发送的请求或正在读取的历史。
  useEffect(() => () => {
      sessionsRequestRef.current?.abort();
      messagesRequestRef.current?.abort();
      olderMessagesRequestRef.current?.abort();
      sendRequestRef.current?.abort();
      stopRequestRef.current?.abort();
      tailRequestsRef.current.forEach((request) => request.controller.abort());
      window.clearTimeout(deletedNoticeTimerRef.current);
      window.clearTimeout(sessionsRefreshTimerRef.current ?? undefined);
  }, []);

  useEffect(() => {
    if (!chatReady) return;
    const controller = new AbortController();
    void loadWebPluginCatalog(controller.signal).catch((error: unknown) => {
      if (!isAbortError(error)) reportError(error);
    });
    return () => controller.abort();
  }, [chatReady, reportError]);

  const groupSections = useNavigationGroupSections(chatReady, pluginCatalog.version, reportError);

  const ensureSession = useCallback(async () => {
    if (activeSessionRef.current) {
      const sessionId = activeSessionRef.current;
      if (followAfterRef.current === null) await loadMessages(sessionId);
      return sessionId;
    }
    const sessionId = `akashic:${createUuid().replaceAll("-", "")}`;
    if (newChatScopeRef.current) sessionScopesRef.current.set(sessionId, newChatScopeRef.current);
    newChatScopeRef.current = null;
    activeSessionRef.current = sessionId;
    setFollowAfter(-1);
    followSession(socketRef.current, sessionId, -1);
    // 发送前只准备路由；上传失败时编辑器仍留在原草稿位置。
    return sessionId;
  }, [loadMessages, setFollowAfter]);

  // 等服务端对 requestId 的确认；超时返回 false 交给调用方换连接重发，服务端拒绝时抛 SendRejected。
  const waitForAccepted = useCallback((requestId: string, signal: AbortSignal) => new Promise<boolean>((resolve, reject) => {
    const finish = (result: boolean | Error) => {
      window.clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      ackWaitersRef.current.delete(requestId);
      if (result instanceof Error) reject(result);
      else resolve(result);
    };
    const abort = () => finish(new DOMException("请求已取消", "AbortError"));
    const timer = window.setTimeout(() => finish(false), SEND_ACK_TIMEOUT_MS);
    signal.addEventListener("abort", abort, { once: true });
    ackWaitersRef.current.set(requestId, (error) => finish(error ?? true));
  }), []);

  // 1. 发出后等确认；2. 超时或连接打不开就丢弃当前连接，用同一 requestId 在新连接上重发；3. 用尽次数后交还给输入框。
  const deliverMessage = useCallback(async (requestId: string, payload: Record<string, unknown>, signal: AbortSignal, onRetry: () => void) => {
    for (let attempt = 1; ; attempt += 1) {
      const socket = connect();
      try {
        await sendWhenOpen(socket, payload, signal);
        if (await waitForAccepted(requestId, signal)) return;
      } catch (error) {
        if (isAbortError(error) || error instanceof SendRejected) throw error;
        console.warn("[chat-ui] send attempt failed", { attempt, error });
      }
      if (attempt >= SEND_ATTEMPTS) throw new DeliveryFailed("网络不稳定，消息没有送达，已放回输入框。");
      console.warn("[chat-ui] send not acknowledged, reconnecting", { attempt });
      onRetry();
      if (socketRef.current === socket) closeConnection();
    }
  }, [closeConnection, connect, waitForAccepted]);

  const sendMessage = useCallback(async (text: string, files: ComposerFile[]) => {
    const cleanText = text.trim();
    if (!cleanText && files.length === 0) return;
    if (!canSend) throw new Error(sendProblem || "聊天服务暂不可用，请稍后重试。");
    if (sendRequestRef.current) throw new Error("消息正在发送，请稍等。");
    setError("");
    setStatusLive(files.length ? "uploading" : "submitted");
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    const controller = new AbortController();
    sendRequestRef.current = controller;
    const clientMessageId = createUuidV7();
    const reply = replyTarget;
    // 本地预览先进入消息区；上传完成后换成服务端附件身份。
    setMessages((current) => [...current, {
      // 只发附件时不写正文：附件条本身就是这条消息，不再在气泡里重复一遍文件名。
      id: clientMessageId, role: "user", content: cleanText,
      attachments: files.map((file, index) => ({
        id: `${clientMessageId}:${index}`, type: "file", filename: file.filename,
        mediaType: file.mediaType || "application/octet-stream", url: file.url || "", sizeBytes: file.sizeBytes,
      })),
      blocks: [], createdAt: new Date().toISOString(), canonical: false, delivery: "sending",
    }]);
    try {
      const sessionId = await ensureSession();
      controller.signal.throwIfAborted();
      console.info("[chat-ui] sendMessage", {
        sessionId,
        textLength: cleanText.length,
        files: files.length,
      });
      const media = await uploadFiles(files, controller.signal);
      controller.signal.throwIfAborted();
      const attachments = media.map((item) => uploadedFileToAttachment(item));
      setMessages((current) => current.map((item) => item.id === clientMessageId ? { ...item, attachments } : item));
      setStatusLive("submitted");
      const payload: Record<string, unknown> = {
        type: "message.send",
        request_id: clientMessageId,
        session_id: sessionId,
        text: cleanText,
        media: media.map((item) => item.artifact_id),
      };
      if (reply) payload.reply_to_message_id = reply.id;
      const scope = sessionScopesRef.current.get(sessionId);
      if (scope) payload.session_dimensions = scope;
      const { parts } = composerContributions();
      if (parts.length) payload.parts = parts;
      const markDelivery = (delivery: ChatMessage["delivery"]) => setMessages((current) => current.map((item) => item.id === clientMessageId ? { ...item, delivery } : item));
      await deliverMessage(clientMessageId, payload, controller.signal, () => markDelivery("retrying"));
      controller.signal.throwIfAborted();
      markDelivery(undefined);
      setActiveSessionId(sessionId);
      console.debug("[chat-ui] send frame delivered", { sessionId });
      notifyComposerSent();
      for (const [key, value] of replyTargetsRef.current) if (value === reply) replyTargetsRef.current.delete(key);
      setReplyTarget(null);
      return sessionId;
    } catch (error) {
      setMessages((current) => current.filter((message) => message.id !== clientMessageId));
      if (isAbortError(error)) {
        if (sendRequestRef.current === controller) setStatusLive("idle");
        throw error;
      }
      reportError(error instanceof DeliveryFailed ? error
        : new SendFailed(`没有发出：${errorMessage(error).replace(/[。.]$/u, "")}。内容还在输入框里，可以直接再发。`), "error");
      throw error;
    } finally {
      if (sendRequestRef.current === controller) sendRequestRef.current = null;
    }
  }, [canSend, sendProblem, deliverMessage, ensureSession, replyTarget, reportError, setMessages, setStatusLive]);

  // 重跑失败的这一轮：只引用原输入，服务端追加恢复记录，不新增用户消息。
  const retryTurn = useCallback(async (inputId: string) => {
    const sessionId = activeSessionRef.current;
    if (!sessionId || !canSend || sendRequestRef.current) return;
    setError("");
    const controller = new AbortController();
    sendRequestRef.current = controller;
    setStatusLive("submitted");
    const requestId = createUuidV7();
    try {
      await deliverMessage(requestId, {
        type: "message.send", request_id: requestId, session_id: sessionId, text: "", media: [], retry_of: inputId,
      }, controller.signal, () => undefined);
    } catch (error) {
      if (sendRequestRef.current === controller) setStatusLive("idle");
      if (!isAbortError(error)) reportError(error, "error");
    } finally {
      if (sendRequestRef.current === controller) sendRequestRef.current = null;
    }
  }, [canSend, deliverMessage, reportError, setStatusLive]);

  const sendSteerFrame = useCallback((payload: Record<string, unknown>) => sendWhenOpen(connect(), payload), [connect]);
  const steers = useSteers({ sessionId: activeSessionId, timeline: timelineMessages, send: sendSteerFrame });
  useEffect(() => { steerResultRef.current = steers.onResult; }, [steers.onResult]);

  const stopTurn = useCallback(() => {
    if (sendRequestRef.current) {
      sendRequestRef.current.abort();
      if (socketRef.current?.readyState === WebSocket.CONNECTING) {
        closeConnection();
        socketRef.current = null;
        statusRef.current = "idle";
      }
      setStatusLive("idle");
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
      .catch((error: unknown) => {
        reportError(error, "error");
        setStopPending(false);
      })
      .finally(() => {
        if (stopRequestRef.current === controller) stopRequestRef.current = null;
      });
  }, [activeSessionId, closeConnection, connect, reportError, setStatusLive]);

  // "正在停止"持续到本轮真正结束：/stop 发出只代表请求送达，后端要到下一个检查点才停。
  // status 由回复活动推导，生成中不会出现非生成的中间值；离开生成集合即本轮结束。
  useEffect(() => {
    if (stopPending && !isGeneratingChatStatus(status)) setStopPending(false);
  }, [status, stopPending]);
  // 自点击起 10 秒仍未结束时复位，让用户可以再停一次。
  useEffect(() => {
    if (!stopPending) return;
    const timer = window.setTimeout(() => setStopPending(false), 10_000);
    return () => window.clearTimeout(timer);
  }, [stopPending]);

  const startNewChat = useCallback(() => {
    setSurface("chat");
    clearChatRoute();
    activeSessionRef.current = "";
    closeConnection();
    messagesRequestRef.current?.abort();
    olderMessagesRequestRef.current?.abort();
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    messagesRequestRef.current = null;
    olderMessagesRequestRef.current = null;
    setHistoryLoading(false);
    setActiveSessionId("");
    setPendingSessionId("");
    setMessages([]);
    setTimelineMessages([]);
    setActiveSessionDeleted(false);
    setFollowAfter(null);
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
    newChatScopeRef.current = null;
    setNewChatGroup(null);
    // 空白页仍订阅侧栏摘要；旧会话已清空，重连不会继续跟随旧正文。
    connect();
  }, [closeConnection, connect, setFollowAfter, setMessages, setReplyAvailable, setTimelineMessages]);

  const startGroupChat = useCallback((ref: GroupRef) => {
    startNewChat();
    newChatScopeRef.current = { [ref.dimension]: ref.id };
    setNewChatGroup(ref);
  }, [startNewChat]);

  const activateSession = useCallback((sessionId: string) => {
    if (surface === "chat" && activeSessionRef.current === sessionId) return;
    const row = sessions.find((session) => session.key === sessionId)
      ?? navigationPins.sessions.find((session) => session.key === sessionId);
    if (row) sessionScopesRef.current.set(sessionId, row.scope ?? {});
    // 置顶解析可能早于完整目录；打开已有会话不能沿用先前分组草稿的 scope。
    newChatScopeRef.current = null;
    setNewChatGroup(null);
    setSurface("chat");
    clearChatRoute();
    // 空白页只订阅目录，没有旧正文跟随；首次进入会话可以复用已有连接。
    if (activeSessionRef.current) closeConnection();
    activeSessionRef.current = sessionId;
    sendRequestRef.current?.abort();
    stopRequestRef.current?.abort();
    messagesRequestRef.current?.abort();
    messagesRequestRef.current = null;
    setHistoryLoading(false);
    olderMessagesRequestRef.current?.abort();
    olderMessagesRequestRef.current = null;
    setHistoryLoadingOlder(false);
    const cached = tailCacheRef.current.get(sessionId);
    const cachedFresh = cached !== undefined && Date.now() - cached.fetchedAt < SESSION_TAIL_FRESH_MS;
    setFollowAfter(cached?.throughSeq ?? null);
    replyActivitiesRef.current = [];
    setReplyActivities([]);
    setReplyAvailable(null);
    setStatusLive("idle");
    setStopPending(false);
    setActiveSessionId(sessionId);
    setPendingSessionId(sessionId);
    setTimelineMessages(cached?.items ?? []);
    setActiveSessionDeleted(cached?.deleted ?? false);
    setMessages([]);
    setHistoryThroughSeq(cached?.throughSeq ?? null);
    setHistoryBeforeSeq(cached?.beforeSeq ?? null);
    setHistoryHasMore(cached?.hasMore ?? false);
    setReplyTarget(replyTargetsRef.current.get(sessionId) ?? null);
    void (cachedFresh ? Promise.resolve() : loadMessages(sessionId))
      .catch(reportHistoryError)
      .finally(() => {
        if (activeSessionRef.current === sessionId) setPendingSessionId("");
      });
    // 新鲜快照跳过了分页拉取，仍需恢复 WebSocket 增量跟随。
    if (cachedFresh) followSession(socketRef.current ?? connectRef.current?.() ?? null, sessionId, cached.throughSeq);
  }, [closeConnection, loadMessages, reportError, reportHistoryError, setFollowAfter, setMessages, setTimelineMessages, setReplyAvailable, setStatusLive, surface, sessions, navigationPins.sessions]);

  useEffect(() => {
    if (!chatReady || !requestedSessionId) return;
    setRequestedSessionId("");
    activateSession(requestedSessionId);
  }, [activateSession, chatReady, requestedSessionId]);

  /** 软删只移除目录呈现；撤销窗口内可经 undelete 恢复，超窗后 UI 不再提供恢复入口。 */
  const deleteSession = useCallback(async (sessionId: string, title: string) => {
    try {
      await setChatSessionDeleted(sessionId, true);
    } catch (error) {
      reportError(error);
      throw error;
    }
    setSessions((current) => current.filter((session) => session.key !== sessionId));
    window.clearTimeout(deletedNoticeTimerRef.current);
    setDeletedNotice({ key: sessionId, title });
    deletedNoticeTimerRef.current = window.setTimeout(() => setDeletedNotice(null), DELETED_NOTICE_MS);
    void loadSessionsSafely();
    void navigationPins.reload();
    // 删除当前会话等于离开它：回到新会话草稿；消息仍在服务端只读保留。
    if (activeSessionRef.current === sessionId) startNewChat();
  }, [loadSessionsSafely, navigationPins, reportError, startNewChat]);

  const restoreSession = useCallback(async (sessionId: string) => {
    try {
      await setChatSessionDeleted(sessionId, false);
    } catch (error) {
      reportError(error);
      return;
    }
    setDeletedNotice((notice) => {
      if (notice?.key === sessionId) window.clearTimeout(deletedNoticeTimerRef.current);
      return notice?.key === sessionId ? null : notice;
    });
    void loadSessionsSafely();
    void navigationPins.reload();
    if (activeSessionRef.current === sessionId) setActiveSessionDeleted(false);
  }, [loadSessionsSafely, navigationPins, reportError]);

  const dismissDeletedNotice = useCallback(() => {
    window.clearTimeout(deletedNoticeTimerRef.current);
    setDeletedNotice(null);
  }, []);

  /** 标题覆盖：成功后本地行即时更新，再对齐目录与置顶解析；拒绝时向调用者抛出以便行内保留编辑态。 */
  const renameSession = useCallback(async (sessionId: string, title: string) => {
    try {
      const stored = await renameChatSession(sessionId, title);
      setSessions((current) => current.map((session) =>
        session.key === sessionId ? { ...session, title: stored } : session));
    } catch (error) {
      reportError(error);
      throw error;
    }
    void loadSessionsSafely();
    void navigationPins.reload();
  }, [loadSessionsSafely, navigationPins, reportError]);


  const handleReplyMessage = useCallback((reply: TimelineReply) => {
    if (activeSessionRef.current) replyTargetsRef.current.set(activeSessionRef.current, reply);
    setReplyTarget(reply);
  }, []);
  const cancelReply = useCallback(() => {
    replyTargetsRef.current.delete(activeSessionRef.current);
    setReplyTarget(null);
  }, []);
  const handleCopiedMessage = useCallback((messageId: string) => {
    setCopiedMessageId(messageId);
    window.setTimeout(() => setCopiedMessageId(""), 1200);
  }, []);
  // 侧栏行 memo 以会话对象身份为凭：目录重拉（流式追加、活动状态都会触发）后字段不变的
  // 会话复用上一份对象，整表身份稳定，未变化的行不重渲染。按值比较，复用总是安全；
  // 每次重建整表替换，不积累已删除会话。
  const sidebarSessions = useMemo(() => {
    const known = new Set(sessions.map((session) => session.key));
    // 补齐已解析置顶的标题/身份，不改变现有最近目录的自然顺序。
    const source = [...sessions, ...navigationPins.sessions.filter((session) => !known.has(session.key))];
    return shareSidebarRows(source.map((session) => ({
      id: session.key,
      title: sessionLabel(session),
      preview: session.message_count === undefined ? "" : `${session.message_count} 条消息`,
      updatedAt: session.updated_at,
      createdAt: session.created_at,
      active: activeSessionId === session.key,
      status: sessionStatuses.get(session.key),
      scope: session.scope ?? {},
      scoped: Object.keys(session.scope ?? {}).length > 0,
    })));
  }, [activeSessionId, sessions, navigationPins.sessions, sessionStatuses]);
  // 新对话在首条消息进入目录前沿用发起时选定的分组。
  const activeRow = activeSessionId ? sessions.find((session) => session.key === activeSessionId)
    ?? navigationPins.sessions.find((session) => session.key === activeSessionId) : undefined;
  const activeScope = useMemo(() => activeRow ? activeRow.scope
    : newChatGroup ? { [newChatGroup.dimension]: newChatGroup.id } : undefined, [activeRow, newChatGroup]);
  const navigationGroups = useMemo((): NavigationGroupsState => {
    // 当前会话归入第一个认得它取值的分区，与侧栏的归类一致。
    const active = groupSections.sections.flatMap(({ source, groups }) => {
      const id = activeScope?.[source.provider.dimension];
      const group = id ? groups.find((item) => item.id === id) : undefined;
      return group ? [{ dimension: source.provider.dimension, id: group.id, label: group.label }] : [];
    })[0] ?? null;
    return { ...groupSections, active, startChat: startGroupChat };
  }, [activeScope, groupSections, startGroupChat]);
  const retryHistory = useCallback(() => {
    if (activeSessionRef.current) void loadMessagesSafely(activeSessionRef.current);
  }, [loadMessagesSafely]);
  const retry = useCallback(() => {
    setError("");
    closeConnection();
    replyActivitiesRef.current = [];
    setReplyActivities([]);
    setReplyAvailable(null);
    const sessionId = activeSessionRef.current;
    if (sessionId && timelineRef.current.length) void loadMessagesSafely(sessionId);
    else connect();
    void loadSessionsSafely();
  }, [closeConnection, connect, loadMessagesSafely, loadSessionsSafely, setReplyAvailable]);

  return {
    surface, sidebarSessions, activeSessionId, pendingSessionId, chatReady, messages, timelineMessages, timelineRefresh, replyActivities, replyAvailable, status,
    streamStore, messageElementsRef, copiedMessageId, shellState, stopPending,
    canSend, sendProblem, draftKey: activeSessionId || `new:${newChatGroup ? groupKey(newChatGroup) : ""}`,
    historyHasMore, historyLoading, historyLoadingOlder, loadOlderMessages, sessionRequested: requestedSessionId !== "",
    historyFailure: activeSessionId !== "" && historyFailure?.sessionId === activeSessionId ? historyFailure.reason : null, retryHistory,
    activeSessionDeleted, deletedNotice, deleteSession, restoreSession, dismissDeletedNotice, renameSession,
    // 自动重连是会自己恢复的临时状态，单独给视图一个安静的状态行；重连失败等才走错误行。
    reconnecting: !error && connectionError === RECONNECTING_NOTICE,
    connectionLost: connectionError !== "", pageOutdated,
    replyTarget, error: error || (connectionError === RECONNECTING_NOTICE ? "" : connectionError),
    // 发送失败时"重试"（重连并重新加载）不会重新发送，反而误导；发送钮就是重试入口，横幅不再给按钮。
    errorRetry: !error || !errorFromSend,
    activateSession, prefetchSessionTail, startNewChat, handleReplyMessage, handleCopiedMessage,
    reportError, cancelReply, sendMessage, retryTurn, stopTurn, retry,

    steers: steers.items, steerMessage: steers.steer, withdrawSteer: steers.withdraw, reclaimSteers: steers.reclaim,
    navigationGroups, navigationPins,
  };
}

export type DesktopChatController = ReturnType<typeof useDesktopChatController>;
