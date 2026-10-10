import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SessionStatus } from "./session-status-mark";
import { fetchChatJson } from "./web-chat-data";

const ENDPOINT = "/api/chat/sessions/activity";
const SEEN_KEY = "akashic.chat.seen-head.v1";
const POLL_MS = 4000;

/** 会话目录里能判断"有无新消息"的最小事实。 */
export interface SessionHead {
  key: string;
  headSeq: number | undefined;
}

function readActive(payload: unknown): ReadonlySet<string> {
  if (typeof payload !== "object" || payload === null) throw new Error("会话活动响应无效");
  const body = payload as Record<string, unknown>;
  if (body.version !== 1 || !Array.isArray(body.active) || body.active.some((id) => typeof id !== "string")) {
    throw new Error("会话活动响应无效");
  }
  return new Set(body.active as string[]);
}

function readSeen(): Record<string, number> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(SEEN_KEY) ?? "{}");
    if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
    return Object.fromEntries(Object.entries(value).filter(([, seq]) => Number.isSafeInteger(seq)));
  } catch {
    return {};
  }
}

// 1. 轮询哪些会话在回复：只在页面可见时读，失败只当作"没有在回复"，不打扰用户。
function useRunningSessions(onSettled: () => void): ReadonlySet<string> {
  const [running, setRunning] = useState<ReadonlySet<string>>(new Set());
  const previous = useRef<ReadonlySet<string>>(new Set());
  const settled = useRef(onSettled);
  settled.current = onSettled;
  useEffect(() => {
    let active = true;
    let request: AbortController | null = null;
    const refresh = async () => {
      if (document.visibilityState === "hidden") return;
      request?.abort();
      const controller = new AbortController();
      request = controller;
      let next: ReadonlySet<string> = new Set();
      try {
        next = readActive(await fetchChatJson<unknown>(ENDPOINT, { signal: controller.signal }));
      } catch {
        if (controller.signal.aborted) return;
      }
      if (!active) return;
      // 2. 有会话刚结束回复：目录里的 head_seq 已前进，重新读取目录才能得知未读。
      if ([...previous.current].some((id) => !next.has(id))) settled.current();
      previous.current = next;
      setRunning(next);
    };
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, POLL_MS);
    const onVisible = () => { void refresh(); };
    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      active = false;
      request?.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);
  return running;
}

// 3. 本浏览器记住每个会话"上次看到的 head_seq"；当前打开的会话持续跟进，首次出现的会话以当前为基线。
function useSeenHeads(heads: readonly SessionHead[], activeSessionId: string): Readonly<Record<string, number>> {
  const [seen, setSeen] = useState<Record<string, number>>(readSeen);
  useEffect(() => {
    setSeen((current) => {
      const next = { ...current };
      let changed = false;
      for (const { key, headSeq } of heads) {
        if (headSeq === undefined) continue;
        if (key === activeSessionId || next[key] === undefined) {
          if (next[key] !== headSeq) { next[key] = headSeq; changed = true; }
        }
      }
      if (!changed) return current;
      try { localStorage.setItem(SEEN_KEY, JSON.stringify(next)); } catch { /* 禁用存储时只在本次浏览内生效。 */ }
      return next;
    });
  }, [heads, activeSessionId]);
  return seen;
}

/** 侧栏会话状态：正在回复优先于未读；当前打开的会话永远不算未读。 */
export function useSessionActivity(
  heads: readonly SessionHead[], activeSessionId: string, reloadSessions: () => void,
): ReadonlyMap<string, SessionStatus> {
  const onSettled = useCallback(() => reloadSessions(), [reloadSessions]);
  const running = useRunningSessions(onSettled);
  const seen = useSeenHeads(heads, activeSessionId);
  return useMemo(() => {
    const statuses = new Map<string, SessionStatus>();
    for (const { key, headSeq } of heads) {
      if (running.has(key)) statuses.set(key, "running");
      else if (key !== activeSessionId && headSeq !== undefined && seen[key] !== undefined && headSeq > seen[key]) {
        statuses.set(key, "unread");
      }
    }
    return statuses;
  }, [heads, running, seen, activeSessionId]);
}
