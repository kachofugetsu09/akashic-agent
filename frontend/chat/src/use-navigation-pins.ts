import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage, fetchChatJson, sessionPage, type SessionRow } from "./web-chat-data";

/** kind 为 "session" 时 id 是会话；否则 kind 是会话维度名、id 是该维度的值（如项目）。 */
export interface NavigationPin {
  kind: string;
  id: string;
}

const PIN_KIND = /^[a-z][a-z0-9_]{0,31}$/;

interface NavigationPinsSnapshot {
  pins: NavigationPin[];
  sessions: SessionRow[];
}

const ENDPOINT = "/api/chat/navigation/pins";
// 展开的分组按 groupKey（维度:取值）记住。
const EXPANDED_KEY = "akashic.chat.expanded-groups";

function snapshot(payload: unknown): NavigationPinsSnapshot {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("置顶列表无效");
  const row = payload as Record<string, unknown>;
  if (!Array.isArray(row.pins) || row.pins.some((pin: unknown) => {
    if (!pin || typeof pin !== "object" || Array.isArray(pin)) return true;
    const item = pin as Record<string, unknown>;
    return typeof item.kind !== "string" || !PIN_KIND.test(item.kind) || typeof item.id !== "string" || !item.id;
  })) throw new Error("置顶列表无效");
  const pins = row.pins as NavigationPin[];
  if (new Set(pins.map((pin) => `${pin.kind}:${pin.id}`)).size !== pins.length) throw new Error("置顶列表包含重复项目");
  return { pins, sessions: sessionPage({ items: row.sessions, next_cursor: null }).items };
}

function readExpandedGroups(): ReadonlySet<string> {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(EXPANDED_KEY) ?? "[]");
    return new Set(Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : []);
  } catch {
    return new Set();
  }
}

/** 置顶事实来自 workspace；展开仅是本浏览器的呈现偏好，与对象所在分区无关。 */
export function useNavigationPins(enabled: boolean) {
  const [data, setData] = useState<NavigationPinsSnapshot>({ pins: [], sessions: [] });
  const [ready, setReady] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const pendingRef = useRef(false);
  const versionRef = useRef(0);
  const deferredReloadRef = useRef<(() => Promise<void>) | null>(null);
  const [expandedGroups, setExpandedGroups] = useState(readExpandedGroups);

  const reload = useCallback(async function refresh(): Promise<void> {
    if (!enabled) return;
    if (pendingRef.current) {
      deferredReloadRef.current = refresh;
      return;
    }
    const version = ++versionRef.current;
    try {
      const next = snapshot(await fetchChatJson<unknown>(ENDPOINT));
      if (version !== versionRef.current) return;
      setData(next);
      setReady(true);
      setError("");
    } catch (reason) {
      if (version === versionRef.current) setError(errorMessage(reason));
    }
  }, [enabled]);

  useEffect(() => {
    void reload();
    const refresh = () => { void reload(); };
    window.addEventListener("focus", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      versionRef.current += 1;
      deferredReloadRef.current = null;
    };
  }, [reload]);

  useEffect(() => {
    try { localStorage.setItem(EXPANDED_KEY, JSON.stringify([...expandedGroups])); }
    catch { /* 禁用存储时仍允许本次浏览的展开操作。 */ }
  }, [expandedGroups]);

  const toggleGroup = useCallback((key: string) => {
    setExpandedGroups((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // 切换到某分组的对话时自动展开它，其余分组保持用户上次的折叠状态。
  const expandGroup = useCallback((key: string) => {
    setExpandedGroups((current) => current.has(key) ? current : new Set(current).add(key));
  }, []);

  const setPinned = useCallback(async (pin: NavigationPin, pinned: boolean) => {
    if (!enabled || !ready || pendingRef.current) return;
    pendingRef.current = true;
    setPending(true);
    setError("");
    const version = ++versionRef.current;
    try {
      const next = snapshot(await fetchChatJson<unknown>(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...pin, pinned }),
      }));
      if (version === versionRef.current) setData(next);
    } catch (reason) {
      if (version === versionRef.current) setError(`${errorMessage(reason)}；可刷新置顶列表确认结果`);
    } finally {
      pendingRef.current = false;
      setPending(false);
      const deferred = deferredReloadRef.current;
      deferredReloadRef.current = null;
      if (deferred) void deferred();
    }
  }, [enabled, ready]);

  return { ...data, ready: ready && enabled, pending, error, reload, setPinned, expandedGroups, toggleGroup, expandGroup };
}

export type NavigationPinsState = ReturnType<typeof useNavigationPins>;
