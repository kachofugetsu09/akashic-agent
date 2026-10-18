import { Folder } from "lucide-react";
import { useEffect, useState } from "react";
import { HeaderMarker, HeaderMarkerActions, copyText, render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";

interface CurrentDirectory {
  path: string | null;
  status: string;
  error?: string;
  agents: { status: string; sources: string[]; error?: string };
}

const STATUS_LABELS: Record<string, string> = { available: "可用", unset: "未设置", not_found: "目录已缺失",
  not_directory: "不是目录", permission_denied: "没有访问权限", offline: "执行主机离线", io_error: "读取失败" };
const statusLabel = (status: string) => STATUS_LABELS[status] ?? status;

// 顶栏只露出目录名；完整路径留给点开的面板。
function directoryName(path: string): string {
  const trimmed = path.length > 1 ? path.replace(/\/+$/, "") : path;
  return trimmed.slice(trimmed.lastIndexOf("/") + 1) || trimmed;
}

function currentDirectory(row: Record<string, unknown>): CurrentDirectory {
  const rules = row.agents;
  if ((row.path !== null && typeof row.path !== "string") || typeof row.status !== "string"
    || (row.error !== undefined && typeof row.error !== "string")) throw new Error("目录状态响应无效");
  if (typeof rules !== "object" || rules === null || !("status" in rules) || typeof rules.status !== "string"
    || !("sources" in rules) || !Array.isArray(rules.sources) || rules.sources.some((path: unknown) => typeof path !== "string")
    || ("error" in rules && typeof rules.error !== "string")) throw new Error("仓库规则状态响应无效");
  return { path: row.path, status: row.status, error: row.error as string | undefined,
    agents: { status: rules.status, sources: rules.sources as string[], error: "error" in rules ? rules.error as string : undefined } };
}

// 只读会话状态；工具回执（宿主失效通知）、重新聚焦及可见轮询都重新检查执行主机。
function SessionDirectory({ context }: { context: PluginUiContext }) {
  const [current, setCurrent] = useState<CurrentDirectory | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      if (document.visibilityState === "hidden") return;
      try {
        const next = currentDirectory(await context.query("directory.current", {}, { cache: "none" }));
        if (active) { setCurrent(next); setError(""); }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "目录状态未能刷新");
      }
    };
    void refresh();
    const onVisible = () => { void refresh(); };
    const timer = window.setInterval(onVisible, 15000);
    const unsubscribe = context.onInvalidate(onVisible);
    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      active = false;
      unsubscribe();
      window.clearInterval(timer);
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [context, retry]);
  // 目录未设置属于常态，不占位置；已设置或出现异常才显示标记。
  if (!current?.path) return null;
  return <DirectoryPanel current={current} path={current.path} error={error} onRefresh={() => setRetry((value) => value + 1)} />;
}

function DirectoryPanel({ current, path, error, onRefresh }: {
  current: CurrentDirectory;
  path: string;
  error: string;
  onRefresh: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const abnormal = !["available", "unset"].includes(current.status);
  const label = abnormal ? statusLabel(current.status) : directoryName(path);
  const copy = () => {
    void copyText(path).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return <HeaderMarker icon={<Folder size={14} aria-hidden="true" />} label={label} alert={abnormal}
    ariaLabel={`工作目录：${path}${abnormal ? `，${label}` : ""}`} panelLabel="工作目录">
    <code>{path}</code>
    <p>{abnormal ? `${statusLabel(current.status)}${current.error ? `：${current.error}` : ""}` : "此对话独立使用这个目录，不影响项目默认目录。"}</p>
    {current.agents.status === "ready" && current.agents.sources.length ? <p>
      仓库规则：{current.agents.sources.map((source) => directoryName(source)).join("、")}
    </p> : current.agents.status !== "ready" ? <p role="status">仓库规则不可用：{current.agents.error}</p> : null}
    {error ? <p data-tone="error" role="status">{error}</p> : null}
    <HeaderMarkerActions>
      <button type="button" onClick={copy}>{copied ? "已复制" : "复制路径"}</button>
      <button type="button" onClick={onRefresh}>刷新</button>
    </HeaderMarkerActions>
  </HeaderMarker>;
}

const definition: PluginUiDefinition = {
  slots: { "header.end": { mount: (host, context) => render(host, <SessionDirectory context={context} />) } },
};
export default definition;
