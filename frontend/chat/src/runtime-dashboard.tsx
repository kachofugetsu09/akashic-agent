import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  BookOpenText,
  CheckCircle2,
  ChevronRight,
  Copy,
  FileText,
  RefreshCw,
  Server,
  Timer,
} from "lucide-react";
import { MessageResponse } from "@/components/ai-elements/message-response";
import "./runtime-dashboard.css";

type RuntimeView = "documents" | "mcp" | "jobs";

interface RuntimeDocument {
  id: string;
  title: string;
  relative_path: string;
  group: string;
  description: string;
  available: boolean;
}

interface RuntimeMcp {
  owner_id: string;
  name: string;
  tool_count: number;
}

interface RuntimeJob {
  id: string;
  name: string | null;
  trigger: string;
  tier: string;
  fire_at: string;
  timezone: string;
  enabled: boolean;
  run_count: number;
}

interface RuntimeCapabilities {
  snapshot_id: string;
  plugins: unknown[];
  skills: unknown[];
  mcp_servers: RuntimeMcp[];
}

interface RuntimeDetail {
  title: string;
  subtitle: string;
  markdown: string;
  copyText: string;
}

interface RuntimeOverview {
  documents: RuntimeDocument[];
  jobs: RuntimeJob[];
  capabilities: RuntimeCapabilities;
}

const viewMeta = {
  documents: { label: "文档", icon: BookOpenText },
  mcp: { label: "MCP 与 Skills", icon: Server },
  jobs: { label: "定时任务", icon: Timer },
} satisfies Record<RuntimeView, { label: string; icon: typeof BookOpenText }>;

/** Render the desktop read-only projection backed by the mobile runtime owners. */
export function RuntimeDashboard() {
  const [view, setView] = useState<RuntimeView>("documents");
  const [overview, setOverview] = useState<RuntimeOverview | null>(null);
  const [selectedKey, setSelectedKey] = useState("");
  const [detail, setDetail] = useState<RuntimeDetail | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [syncedAt, setSyncedAt] = useState<Date | null>(null);

  const loadOverview = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [documents, jobs, capabilities] = await Promise.all([
        fetchRuntimeItems<RuntimeDocument>("/api/chat/runtime/documents"),
        fetchRuntimeItems<RuntimeJob>("/api/chat/runtime/jobs"),
        fetchRuntimeJson<RuntimeCapabilities>("/api/chat/runtime/capabilities"),
      ]);
      setOverview({ documents, jobs, capabilities });
      setSyncedAt(new Date());
    } catch (loadError) {
      setError(runtimeErrorMessage(loadError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  const items = useMemo(() => runtimeItems(view, overview), [overview, view]);

  useEffect(() => {
    const first = items[0];
    setSelectedKey(first?.key ?? "");
    setDetail(null);
    setDetailOpen(false);
  }, [items, view]);

  useEffect(() => {
    if (!selectedKey) return;
    const controller = new AbortController();
    setDetailLoading(true);
    setError("");
    void fetchRuntimeDetail(view, selectedKey, controller.signal)
      .then(setDetail)
      .catch((loadError: unknown) => {
        if (loadError instanceof DOMException && loadError.name === "AbortError") return;
        setError(runtimeErrorMessage(loadError));
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [selectedKey, view]);

  const selectView = (nextView: RuntimeView) => {
    setView(nextView);
    setDetailOpen(false);
  };

  return (
    <section className={`runtime-dashboard runtime-dashboard--${view}`}>
      <header className="runtime-dashboard__topbar">
        <div>
          <h1>知识与运行</h1>
          <p>当前电脑的只读投影 · 与移动端使用相同的运行目录</p>
        </div>
        <div className="runtime-dashboard__sync">
          {syncedAt ? <span><CheckCircle2 size={17} />已同步 · {formatSyncTime(syncedAt)}</span> : null}
          <button type="button" onClick={() => void loadOverview()} disabled={loading}>
            <RefreshCw size={17} className={loading ? "is-spinning" : ""} />
            {loading ? "正在刷新" : "刷新"}
          </button>
        </div>
      </header>

      <section className="runtime-dashboard__metrics" aria-label="运行概览">
        <RuntimeMetric icon={<BookOpenText size={20} />} value={overview?.documents.length ?? 0} label="核心文档" tone="documents" />
        <RuntimeMetric icon={<Server size={20} />} value={overview?.capabilities.mcp_servers.reduce((sum, item) => sum + item.tool_count, 0) ?? 0} label={`MCP 工具 · ${overview?.capabilities.skills.length ?? 0} Skills`} tone="mcp" />
        <RuntimeMetric icon={<Timer size={20} />} value={overview?.jobs.filter((job) => job.enabled).length ?? 0} label="已启用定时任务" tone="jobs" />
      </section>

      <nav className="runtime-dashboard__tabs" role="tablist" aria-label="知识与运行目录">
        {(Object.keys(viewMeta) as RuntimeView[]).map((key) => {
          const Icon = viewMeta[key].icon;
          return (
            <button key={key} type="button" role="tab" aria-selected={view === key} onClick={() => selectView(key)}>
              <Icon size={20} />
              <span>{viewMeta[key].label}</span>
            </button>
          );
        })}
      </nav>

      <div className="runtime-dashboard__notice-slot">
        {error ? <div className="runtime-dashboard__error" role="alert"><AlertCircle size={18} />{error}</div> : null}
      </div>

      <section className={`runtime-directory ${detailOpen ? "detail-open" : ""}`}>
        <div className="runtime-directory__list">
          <header>
            <h2>{viewMeta[view].label}</h2>
            <p>{directoryDescription(view, overview)}</p>
          </header>
          {loading ? <p className="runtime-directory__empty">正在读取最新运行目录…</p> : null}
          {!loading && items.length === 0 ? <p className="runtime-directory__empty">当前目录没有可展示的项目。</p> : null}
          {items.map((item) => (
            <button
              className="runtime-directory__item"
              type="button"
              key={item.key}
              aria-selected={selectedKey === item.key}
              disabled={item.disabled}
              onClick={() => {
                setSelectedKey(item.key);
                setDetailOpen(true);
              }}
            >
              <span className="runtime-directory__item-icon">{item.icon}</span>
              <span className="runtime-directory__item-copy"><strong>{item.title}</strong><small>{item.description}</small></span>
              {item.status ? <span className={`runtime-directory__status ${item.statusTone ?? ""}`}>{item.status}</span> : null}
              <ChevronRight size={18} aria-hidden="true" />
            </button>
          ))}
        </div>

        <article className="runtime-detail">
          <header className="runtime-detail__header">
            <button className="runtime-detail__back" type="button" onClick={() => setDetailOpen(false)}><ArrowLeft size={18} />返回{viewMeta[view].label}</button>
            <div><h2>{detail?.title ?? "选择一个项目"}</h2><p>{detail?.subtitle ?? "在左侧目录中选择要查看的内容。"}</p></div>
            {detail ? <button className="runtime-detail__copy" type="button" onClick={() => void navigator.clipboard.writeText(detail.copyText)}><Copy size={17} />复制标识</button> : null}
          </header>
          <div className="runtime-detail__scroll">
            {detailLoading ? <p className="runtime-detail__loading"><RefreshCw size={22} className="is-spinning" />正在读取最新内容…</p> : null}
            {!detailLoading && detail ? <MessageResponse className="runtime-detail__markdown">{detail.markdown}</MessageResponse> : null}
            {!detailLoading && !detail ? <p className="runtime-detail__loading">详情将在这里显示。</p> : null}
          </div>
        </article>
      </section>
    </section>
  );
}

function RuntimeMetric({ icon, value, label, tone }: { icon: React.ReactNode; value: number; label: string; tone: RuntimeView }) {
  return <div className={`runtime-metric ${tone}`}><span>{icon}</span><div><small>{label}</small><strong>{value}</strong></div></div>;
}

interface DirectoryItem {
  key: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  disabled?: boolean;
  status?: string;
  statusTone?: string;
}

function runtimeItems(view: RuntimeView, overview: RuntimeOverview | null): DirectoryItem[] {
  if (!overview) return [];
  if (view === "documents") {
    return overview.documents.map((document) => ({
      key: document.id,
      title: document.title,
      description: `${document.description} · ${document.relative_path}`,
      icon: <FileText size={19} />,
      disabled: !document.available,
      status: document.available ? undefined : "不可用",
    }));
  }
  if (view === "mcp") {
    return overview.capabilities.mcp_servers.map((server) => ({
      key: `${server.owner_id}\u0000${server.name}`,
      title: server.name,
      description: `${server.tool_count} 个工具 · ${server.owner_id}`,
      icon: <Server size={20} />,
      status: "Ready",
      statusTone: "enabled",
    }));
  }
  return overview.jobs.map((job) => ({
    key: job.id,
    title: job.name || "未命名定时任务",
    description: `${formatFireAt(job.fire_at)} · ${job.trigger}/${job.tier}`,
    icon: <Timer size={20} />,
    status: job.enabled ? "启用" : "停用",
    statusTone: job.enabled ? "enabled" : "",
  }));
}

async function fetchRuntimeDetail(view: RuntimeView, key: string, signal: AbortSignal): Promise<RuntimeDetail> {
  if (view === "documents") {
    const payload = await fetchRuntimeJson<Record<string, unknown>>(`/api/chat/runtime/documents/${encodeURIComponent(key)}`, signal);
    return {
      title: requireString(payload.title, "文档标题"),
      subtitle: `只读 · ${requireString(payload.relative_path, "文档路径")}`,
      markdown: requireString(payload.markdown, "文档内容"),
      copyText: requireString(payload.relative_path, "文档路径"),
    };
  }
  if (view === "mcp") {
    const [ownerId, name] = key.split("\u0000");
    const query = new URLSearchParams({ owner_id: ownerId, name });
    const payload = await fetchRuntimeJson<Record<string, unknown>>(`/api/chat/runtime/mcp?${query}`, signal);
    return {
      title: requireString(payload.name, "MCP 名称"),
      subtitle: `MCP Server · ${requireString(payload.owner_id, "MCP owner")}`,
      markdown: requireString(payload.markdown, "MCP 内容"),
      copyText: `${ownerId}/${name}`,
    };
  }
  const payload = await fetchRuntimeJson<Record<string, unknown>>(`/api/chat/runtime/jobs/${encodeURIComponent(key)}`, signal);
  return {
    title: typeof payload.name === "string" && payload.name ? payload.name : "未命名定时任务",
    subtitle: `Schedule · ${requireString(payload.timezone, "任务时区")}`,
    markdown: requireString(payload.markdown, "任务内容"),
    copyText: requireString(payload.id, "任务标识"),
  };
}

async function fetchRuntimeItems<T>(url: string): Promise<T[]> {
  const payload = await fetchRuntimeJson<{ items: T[] }>(url);
  if (!Array.isArray(payload.items)) throw new Error(`${url} 返回了无效列表`);
  return payload.items;
}

async function fetchRuntimeJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const payload = await response.json() as { detail?: unknown };
    throw new Error(typeof payload.detail === "string" ? payload.detail : `${url} 请求失败 (${response.status})`);
  }
  return await response.json() as T;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label}无效`);
  return value;
}

function directoryDescription(view: RuntimeView, overview: RuntimeOverview | null): string {
  if (view === "documents") return "与移动端相同的固定运行目录";
  if (view === "mcp") return `${overview?.capabilities.plugins.length ?? 0} 插件 · ${overview?.capabilities.skills.length ?? 0} Skills`;
  return "来自 Scheduler owner 的实时任务投影";
}

function formatFireAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatSyncTime(value: Date): string {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(value);
}

function runtimeErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "运行目录读取失败";
}
