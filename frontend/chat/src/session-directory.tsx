import { Folder } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { copyText } from "./copy-text";
import { queryHostPlugin, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { directoryName, directoryState, directoryStatus, type DirectoryState } from "./directory-data";

interface CurrentDirectory extends DirectoryState {
  agents: { status: string; sources: string[]; error?: string };
}

function currentDirectory(row: Record<string, unknown>): CurrentDirectory {
  const state = directoryState(row);
  const rules = row.agents;
  if (typeof rules !== "object" || rules === null || !("status" in rules) || typeof rules.status !== "string"
    || !("sources" in rules) || !Array.isArray(rules.sources) || rules.sources.some((path: unknown) => typeof path !== "string")
    || ("error" in rules && typeof rules.error !== "string")) throw new Error("仓库规则状态响应无效");
  return { ...state, agents: { status: rules.status, sources: rules.sources as string[],
    error: "error" in rules ? rules.error as string : undefined } };
}

/** 只读 Session 状态；工具回执、重新聚焦及可见轮询都重新检查执行主机。 */
export function SessionDirectory({ sessionId, refreshKey }: { sessionId: string; refreshKey: string }) {
  const catalog = usePluginUiCatalogVersion();
  const installed = catalog.installed("standard_tools");
  const [current, setCurrent] = useState<CurrentDirectory | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!installed) return;
    let active = true;
    let request: AbortController | null = null;
    const refresh = async () => {
      if (document.visibilityState === "hidden") return;
      request?.abort();
      const controller = new AbortController();
      request = controller;
      try {
        const next = currentDirectory(await queryHostPlugin("standard_tools", "directory.current", {}, controller.signal, sessionId));
        if (active && !controller.signal.aborted) { setCurrent(next); setError(""); }
      } catch (reason) {
        if (active && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : "目录状态未能刷新");
      }
    };
    void refresh();
    const onVisible = () => { void refresh(); };
    const timer = window.setInterval(onVisible, 15000);
    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      active = false;
      request?.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [catalog.version, installed, refreshKey, retry, sessionId]);
  if (!installed) return null;
  return <DirectoryMarker current={current} error={error} onRefresh={() => setRetry((value) => value + 1)} />;
}

// 顶栏右侧的安静标记：平时只是"文件夹 + 目录名"，异常时才变色；点开看完整路径。
function DirectoryMarker({ current, error, onRefresh }: {
  current: CurrentDirectory | null;
  error: string;
  onRefresh: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const abnormal = current && !["available", "unset"].includes(current.status);
  // 1. 点击面板外或按 Esc 收起，Esc 把焦点还给触发按钮。
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  // 2. 目录未设置属于常态，不占位置；已设置或出现异常才显示标记。
  if (!current?.path) return null;
  const label = abnormal ? directoryStatus(current.status) : directoryName(current.path);
  const copy = () => {
    void copyText(current.path ?? "").then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return <div className="directory-marker" ref={rootRef}>
    <button ref={triggerRef} type="button" className="directory-marker__trigger" data-alert={abnormal || undefined}
      aria-expanded={open} aria-label={`工作目录：${current.path}${abnormal ? `，${label}` : ""}`}
      onClick={() => setOpen((value) => !value)}>
      <Folder size={14} aria-hidden="true" />
      <span>{label}</span>
    </button>
    {open ? <div className="directory-marker__panel" role="group" aria-label="工作目录">
      <code className="directory-path">{current.path}</code>
      <p>{abnormal ? `${directoryStatus(current.status)}${current.error ? `：${current.error}` : ""}` : "此对话独立使用这个目录，不影响项目默认目录。"}</p>
      {current.agents.status === "ready" && current.agents.sources.length ? <p>
        仓库规则：{current.agents.sources.map((path) => directoryName(path)).join("、")}
      </p> : current.agents.status !== "ready" ? <p role="status">仓库规则不可用：{current.agents.error}</p> : null}
      {error ? <p className="directory-error" role="status">{error}</p> : null}
      <div className="directory-marker__actions">
        <button type="button" onClick={copy}>{copied ? "已复制" : "复制路径"}</button>
        <button type="button" onClick={onRefresh}>刷新</button>
      </div>
    </div> : null}
  </div>;
}
