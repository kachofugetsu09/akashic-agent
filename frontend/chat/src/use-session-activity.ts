import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SessionStatus } from "./session-status-mark";
import { ChatRequestError, fetchChatJson } from "./web-chat-data";

const SEEN_KEY = "akashic.chat.seen-head.v1";

/** 目录与 WebSocket 共享已提交消息的水位。 */
export interface SessionHead {
  key: string;
  headSeq: number | undefined;
}

interface SessionActivityFrame {
  snapshot: boolean;
  available: boolean;
  active: string[];
  heads: Record<string, number>;
  removed: string[];
  /** 服务端共享的已读水位；缺席时退回本浏览器记录。 */
  read_available?: boolean;
  read?: Record<string, number>;
}

const seqMap = (value: unknown): value is Record<string, number> => typeof value === "object" && value !== null
  && !Array.isArray(value) && Object.entries(value).every(([key, seq]) => key && Number.isSafeInteger(seq) && Number(seq) >= -1);

/** 只在 WebSocket 边界检查摘要；其他消息继续交给原协议解析器。 */
export function readSessionActivityFrame(value: unknown): SessionActivityFrame | null {
  if (typeof value !== "object" || value === null || !("type" in value) || value.type !== "sessions.activity") return null;
  const row = value as Record<string, unknown>;
  const ids = (items: unknown): items is string[] => Array.isArray(items)
    && items.every((id) => typeof id === "string" && id.length > 0);
  if (row.version !== 1 || typeof row.snapshot !== "boolean" || typeof row.available !== "boolean"
    || !ids(row.active) || !ids(row.removed) || (!row.available && row.active.length > 0)
    || !seqMap(row.heads) || (row.read_available !== undefined && typeof row.read_available !== "boolean")
    || (row.read !== undefined && !seqMap(row.read))) {
    throw new Error("会话活动摘要无效");
  }
  return row as unknown as SessionActivityFrame;
}

function readSeen(): Record<string, number> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(SEEN_KEY) ?? "{}");
    if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, seq]) => Number.isSafeInteger(seq) && Number(seq) >= -1));
  } catch { return {}; }
}

/** 已读只跟进可见页面已经加载的消息，目录刷新不代替阅读。 */
function useSeenHeads(heads: readonly SessionHead[], activeSessionId: string, shownHead: number | null, visible: boolean) {
  const [seen, setSeen] = useState<Record<string, number>>(readSeen);
  useEffect(() => {
    setSeen((current) => {
      const next = { ...current };
      for (const { key, headSeq } of heads) {
        if (headSeq !== undefined && next[key] === undefined) next[key] = headSeq;
      }
      if (visible && activeSessionId && shownHead !== null) {
        next[activeSessionId] = Math.max(next[activeSessionId] ?? -1, shownHead);
      }
      if (Object.keys(next).every((key) => next[key] === current[key])) return current;
      try { localStorage.setItem(SEEN_KEY, JSON.stringify(next)); }
      catch { /* 禁用存储时只在本次浏览内生效。 */ }
      return next;
    });
  }, [heads, activeSessionId, shownHead, visible]);
  return seen;
}

/** 保留实际读过的待提交水位；临时失败退避重试，重连快照会重新核对。 */
function useMarkRead(reads: Record<string, number> | null, activeSessionId: string, shownHead: number | null, reading: boolean) {
  const pending = useRef(new Map<string, number>());
  useEffect(() => {
    if (reads === null) return;
    // 1. 只记录可见页面已加载的消息；切换会话不丢弃此前读过的水位。
    if (reading && activeSessionId && shownHead !== null) {
      pending.current.set(activeSessionId, Math.max(pending.current.get(activeSessionId) ?? -1, shownHead));
    }
    for (const [id, seq] of pending.current) if (seq <= (reads[id] ?? -1)) pending.current.delete(id);
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    // 2. 单调写入允许重试；取消旧请求后，迟到结果不能清掉较新的待提交水位。
    const send = async () => {
      for (const [id, seq] of pending.current) {
        try {
          await fetchChatJson("/api/chat/sessions/read", {
            method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
            body: JSON.stringify({ session_id: id, seq }),
          });
          if (controller.signal.aborted) return;
          if (pending.current.get(id) === seq) pending.current.delete(id);
        } catch (error) {
          if (controller.signal.aborted) return;
          if (error instanceof TypeError || (error instanceof ChatRequestError && (error.status >= 500 || [408, 429].includes(error.status)))) {
            timer = setTimeout(() => void send(), delay);
            delay = Math.min(delay * 2, 30_000);
            return;
          }
          if (pending.current.get(id) === seq) pending.current.delete(id);
          console.error("已读水位上报失败", id, error);
        }
      }
    };
    void send();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [reads, activeSessionId, shownHead, reading]);
}

/** 一个连接的摘要快照与增量；未读水位独立于回复开始、结束是否被观察到。 */
export function useSessionActivity(
  heads: readonly SessionHead[], activeSessionId: string, shownHead: number | null,
  readingChat: boolean, reloadSessions: () => Promise<void>,
) {
  const [data, setData] = useState({ heads: {} as Record<string, number>, running: new Set<string>(), available: false,
    reads: {} as Record<string, number>, readAvailable: null as boolean | null });
  const current = useRef(data);
  const latest = useRef({ heads, activeSessionId, reloadSessions });
  useEffect(() => { latest.current = { heads, activeSessionId, reloadSessions }; }, [heads, activeSessionId, reloadSessions]);
  const [visible, setVisible] = useState(() => document.visibilityState === "visible");
  useEffect(() => {
    const refresh = () => setVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", refresh);
    return () => document.removeEventListener("visibilitychange", refresh);
  }, []);

  // 1. 摘要变化才更新 React；后台回复结束时补目录标题、时间和排序。
  const receive = useCallback((frame: SessionActivityFrame) => {
    const previous = current.current;
    const nextHeads = frame.snapshot ? { ...frame.heads } : { ...previous.heads, ...frame.heads };
    frame.removed.forEach((key) => { delete nextHeads[key]; });
    const running = new Set(frame.active);
    const sameHeads = Object.keys(previous.heads).length === Object.keys(nextHeads).length
      && Object.keys(nextHeads).every((key) => nextHeads[key] === previous.heads[key]);
    const sameRunning = running.size === previous.running.size && [...running].every((key) => previous.running.has(key));
    const readAvailable = frame.read_available ?? false;
    const nextReads = frame.snapshot ? { ...frame.read } : { ...previous.reads, ...frame.read };
    frame.removed.forEach((key) => { delete nextReads[key]; });
    const sameReads = !frame.snapshot && readAvailable === previous.readAvailable
      && Object.keys(previous.reads).length === Object.keys(nextReads).length
      && Object.keys(nextReads).every((key) => nextReads[key] === previous.reads[key]);
    if (!sameHeads || !sameRunning || !sameReads || frame.available !== previous.available) {
      current.current = { heads: sameHeads ? previous.heads : nextHeads,
        running: sameRunning ? previous.running : running, available: frame.available,
        reads: sameReads ? previous.reads : nextReads, readAvailable };
      setData(current.current);
    }
    const known = new Set(latest.current.heads.map((head) => head.key));
    const settled = [...previous.running].some((key) => !running.has(key));
    const refreshHeads = Object.keys(frame.heads).some((key) => !known.has(key)
      || (key !== latest.current.activeSessionId && !running.has(key)));
    if (frame.snapshot || frame.removed.length || settled || refreshHeads) {
      void latest.current.reloadSessions();
    }
  }, []);
  const disconnect = useCallback(() => {
    if (!current.current.available && current.current.running.size === 0) return;
    current.current = { ...current.current, running: new Set(), available: false };
    setData(current.current);
  }, []);

  // 2. 目录提供首次基线；事件水位独立更新，不需要每次重新拉取全部分页。
  const liveHeads = useMemo(() => heads.map((head) => ({ ...head,
    headSeq: data.heads[head.key] === undefined ? head.headSeq
      : Math.max(data.heads[head.key], head.headSeq ?? -1) })), [heads, data.heads]);
  const reading = visible && readingChat;
  const seen = useSeenHeads(liveHeads, activeSessionId, shownHead, reading);
  useMarkRead(data.readAvailable === true ? data.reads : null, activeSessionId, shownHead, reading);
  const statuses = useMemo(() => {
    const result = new Map<string, SessionStatus>();
    // 服务端水位跨设备共享；尚未收到首帧时不亮未读，避免与本地记录来回切换。
    const readOf = (key: string): number | undefined => data.readAvailable === true ? data.reads[key] ?? -1
      : data.readAvailable === false ? seen[key] : undefined;
    for (const { key, headSeq } of liveHeads) {
      const read = readOf(key);
      if (data.available && data.running.has(key)) result.set(key, "running");
      // 正在阅读的会话不亮未读：水位帧可能先于消息帧到达，避免在当前行上闪一下。
      else if (!(reading && key === activeSessionId) && headSeq !== undefined
        && read !== undefined && headSeq > read) result.set(key, "unread");
    }
    return result;
  }, [liveHeads, data.available, data.running, data.reads, data.readAvailable, seen, reading, activeSessionId]);
  return { statuses, receive, disconnect };
}
