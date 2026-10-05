import { useEffect, useState } from "react";
import { queryHostPlugin, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { directoryState, directoryStatus, type DirectoryState } from "./directory-data";

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
  return <div className="session-directory">
    <details>
      <summary><span className="session-directory__arrow" aria-hidden="true">▾</span><span>工作目录</span><span className="directory-path" title={current?.path ?? undefined}>{current?.path ?? (current ? "未设置" : "正在读取…")}</span>
        <small>{error ? "状态未能刷新" : current && current.status !== "unset" ? directoryStatus(current.status) : ""}</small></summary>
      <div className="session-directory__details">
        {current?.path ? <p>当前对话独立使用此目录。Agent 可通过工具切换，项目默认目录保持固定。</p>
          : <p>当前对话未指定目录，Shell 和文件保持既有默认目录。之后绑定项目不会改变这个对话。</p>}
        {current?.agents.status === "ready" && current.agents.sources.length ? <>
          <strong>AGENTS 来源</strong>
          <ul>{current.agents.sources.map((path) => <li className="directory-path" key={path}>{path}</li>)}</ul>
        </> : current?.path && current?.agents.status !== "ready" ? <p role="status">仓库规则不可用：{current?.agents.error}</p> : null}
        {error ? <p className="directory-error" role="status">{error}</p> : null}
        <button type="button" onClick={() => setRetry((value) => value + 1)}>刷新状态</button>
      </div>
    </details>
  </div>;
}
