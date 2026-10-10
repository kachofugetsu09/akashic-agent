import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage, fetchChatJson, sessionPage, type SessionRow } from "./web-chat-data";

export interface NavigationPin {
  kind: "project" | "session";
  id: string;
}

interface NavigationPinsSnapshot {
  pins: NavigationPin[];
  sessions: SessionRow[];
}

const ENDPOINT = "/api/chat/navigation/pins";
const EXPANDED_KEY = "akashic.chat.expanded-projects";

function snapshot(payload: unknown): NavigationPinsSnapshot {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("置顶列表无效");
  const row = payload as Record<string, unknown>;
  if (!Array.isArray(row.pins) || row.pins.some((pin: unknown) => {
    if (!pin || typeof pin !== "object" || Array.isArray(pin)) return true;
    const item = pin as Record<string, unknown>;
    return (item.kind !== "project" && item.kind !== "session") || typeof item.id !== "string" || !item.id;
  })) throw new Error("置顶列表无效");
  const pins = row.pins as NavigationPin[];
  if (new Set(pins.map((pin) => `${pin.kind}:${pin.id}`)).size !== pins.length) throw new Error("置顶列表包含重复项目");
  return { pins, sessions: sessionPage({ items: row.sessions, next_cursor: null }).items };
}

function readExpandedProjects(): ReadonlySet<string> {
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
  const [expandedProjects, setExpandedProjects] = useState(readExpandedProjects);

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
    try { localStorage.setItem(EXPANDED_KEY, JSON.stringify([...expandedProjects])); }
    catch { /* 禁用存储时仍允许本次浏览的展开操作。 */ }
  }, [expandedProjects]);

  const toggleProject = useCallback((projectId: string) => {
    setExpandedProjects((current) => {
      const next = new Set(current);
      if (next.has(projectId)) next.delete(projectId);
      else next.add(projectId);
      return next;
    });
  }, []);

  // 切换到某项目的对话时自动展开它，其余项目保持用户上次的折叠状态。
  const expandProject = useCallback((projectId: string) => {
    setExpandedProjects((current) => current.has(projectId) ? current : new Set(current).add(projectId));
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

  return { ...data, ready: ready && enabled, pending, error, reload, setPinned, expandedProjects, toggleProject, expandProject };
}

export type NavigationPinsState = ReturnType<typeof useNavigationPins>;
