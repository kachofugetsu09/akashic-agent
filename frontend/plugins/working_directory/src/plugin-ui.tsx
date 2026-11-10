import { useEffect, useMemo, useState } from "react";
import { copyText, render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { ICONS, useValue, valueStore, type ValueStore } from "../../../../packages/akashic-plugin-kit/src/meta-store";
import "./plugin-ui.css";

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

interface DirectoryView { current: CurrentDirectory; path: string; error: string; refresh(): void }

// 只读会话状态；工具回执（宿主失效通知）、重新聚焦及可见轮询都重新检查执行主机。
// 不渲染界面：目录登记为说明行「在哪」路径的一段，详细内容在宿主的会话信息面板里展开。
function SessionDirectory({ context }: { context: PluginUiContext<"header.meta"> }) {
  const [current, setCurrent] = useState<CurrentDirectory | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    // 只采纳最后一次发出的读取；较早的响应晚到时不能把标记退回旧目录。
    let latest = 0;
    const refresh = async () => {
      if (document.visibilityState === "hidden") return;
      const request = ++latest;
      try {
        const next = currentDirectory(await context.query("directory.current", {}, { cache: "none" }));
        if (request === latest) { setCurrent(next); setError(""); }
      } catch (reason) {
        if (request === latest) setError(reason instanceof Error ? reason.message : "目录状态未能刷新");
      }
    };
    // 先订阅宿主能力再建计时器：宿主缺少能力时在这里失败，不留下无人清理的轮询。
    const onVisible = () => { void refresh(); };
    const stopShown = context.onShown(onVisible);
    const unsubscribe = context.onInvalidate(onVisible);
    const timer = window.setInterval(onVisible, 15000);
    void refresh();
    return () => {
      latest = -1;
      unsubscribe();
      stopShown();
      window.clearInterval(timer);
    };
  }, [context, retry]);
  const store = useMemo<ValueStore<DirectoryView | null>>(() => valueStore(null), []);
  const path = current?.path ?? null;
  useEffect(() => {
    // 目录未设置属于常态，不占位置；已设置或出现异常才登记。
    if (!current || !path) { store.set(null); context.meta.publish([]); return; }
    store.set({ current, path, error, refresh: () => setRetry((value) => value + 1) });
    const abnormal = !["available", "unset"].includes(current.status);
    context.meta.publish([{
      id: "directory", scope: "project", path: true, weight: 10, iconSvg: ICONS.folder,
      tone: abnormal ? "alert" : "info", label: "工作目录",
      text: abnormal ? statusLabel(current.status) : directoryName(path), value: directoryName(path),
      detail: (host) => render(host, <DirectoryDetail store={store} />),
    }]);
  }, [context, current, error, path, store]);
  return null;
}

function DirectoryDetail({ store }: { store: ValueStore<DirectoryView | null> }) {
  const view = useValue(store);
  const [copied, setCopied] = useState(false);
  if (!view) return null;
  const { current, path, error } = view;
  const abnormal = !["available", "unset"].includes(current.status);
  const copy = () => {
    void copyText(path).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return <div className="directory-detail">
    <code>{path}</code>
    <p data-tone={abnormal ? "error" : undefined}>{abnormal ? `${statusLabel(current.status)}${current.error ? `：${current.error}` : ""}` : "此对话独立使用这个目录，不影响项目默认目录。"}</p>
    {current.agents.status === "ready" && current.agents.sources.length ? <p>
      仓库规则：{current.agents.sources.map((source) => directoryName(source)).join("、")}
    </p> : current.agents.status !== "ready" ? <p role="status">仓库规则不可用：{current.agents.error}</p> : null}
    {error ? <p data-tone="error" role="status">{error}</p> : null}
    <div className="directory-detail__actions">
      <button type="button" onClick={copy}>{copied ? "已复制" : "复制路径"}</button>
      <button type="button" onClick={view.refresh}>刷新</button>
    </div>
  </div>;
}

const definition: PluginUiDefinition = {
  slots: {
    "header.meta": { mount: (host, context) => render(host, <SessionDirectory context={context} />) },
  },
};
export default definition;
