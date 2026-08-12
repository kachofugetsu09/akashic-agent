export type RuntimeView = "documents" | "mcp" | "jobs";

export interface RuntimeDocument {
  id: string;
  title: string;
  relative_path: string;
  group: string;
  description: string;
  available: boolean;
}

export interface RuntimeMcp {
  owner_id: string;
  name: string;
  tool_count: number;
}

export interface RuntimeJob {
  id: string;
  name: string | null;
  trigger: string;
  tier: string;
  fire_at: string;
  timezone: string;
  enabled: boolean;
  run_count: number;
}

export interface RuntimeCapabilities {
  snapshot_id: string;
  plugins: unknown[];
  skills: unknown[];
  mcp_servers: RuntimeMcp[];
}

export interface RuntimeDetail {
  title: string;
  subtitle: string;
  markdown: string;
  copyText: string;
}

export interface RuntimeOverview {
  documents: RuntimeDocument[];
  jobs: RuntimeJob[];
  capabilities: RuntimeCapabilities;
}

export interface RuntimeDirectoryItem {
  key: string;
  title: string;
  description: string;
  icon: RuntimeView;
  disabled?: boolean;
  status?: string;
  statusTone?: string;
}

/** Load the three read-only runtime owners as one overview transaction. */
export async function loadRuntimeOverview(): Promise<RuntimeOverview> {
  const [documents, jobs, capabilities] = await Promise.all([
    fetchRuntimeItems<RuntimeDocument>("/api/chat/runtime/documents"),
    fetchRuntimeItems<RuntimeJob>("/api/chat/runtime/jobs"),
    fetchRuntimeJson<RuntimeCapabilities>("/api/chat/runtime/capabilities"),
  ]);
  return { documents, jobs, capabilities };
}

/** Load the selected runtime detail from its explicit owner. */
export async function loadRuntimeDetail(view: RuntimeView, key: string, signal: AbortSignal): Promise<RuntimeDetail> {
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

export function runtimeItems(view: RuntimeView, overview: RuntimeOverview | null): RuntimeDirectoryItem[] {
  if (!overview) return [];
  if (view === "documents") {
    return overview.documents.map((document) => ({
      key: document.id,
      title: document.title,
      description: `${document.description} · ${document.relative_path}`,
      icon: "documents",
      disabled: !document.available,
      status: document.available ? undefined : "不可用",
    }));
  }
  if (view === "mcp") {
    return overview.capabilities.mcp_servers.map((server) => ({
      key: `${server.owner_id}\u0000${server.name}`,
      title: server.name,
      description: `${server.tool_count} 个工具 · ${server.owner_id}`,
      icon: "mcp",
      status: "可用",
      statusTone: "enabled",
    }));
  }
  return overview.jobs.map((job) => ({
    key: job.id,
    title: job.name || "未命名定时任务",
    description: `${formatFireAt(job.fire_at)} · ${job.trigger}/${job.tier}`,
    icon: "jobs",
    status: job.enabled ? "启用" : "停用",
    statusTone: job.enabled ? "enabled" : "",
  }));
}

export function runtimeDirectoryDescription(view: RuntimeView, overview: RuntimeOverview | null): string {
  if (view === "documents") return "与移动端相同的固定运行目录";
  if (view === "mcp") return `${overview?.capabilities.plugins.length ?? 0} 插件 · ${overview?.capabilities.skills.length ?? 0} Skills`;
  return "来自 Scheduler owner 的实时任务投影";
}

export function formatRuntimeSyncTime(value: Date): string {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(value);
}

export function runtimeErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "运行目录读取失败";
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

function formatFireAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}
