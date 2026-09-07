import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type {
  FetchPageOptions as FetchPageOpts,
  FetchPageResult,
  WorkbenchDispatch as PluginDispatch,
  WorkbenchPanelEntry,
} from "@akashic/workbench-ui-v2";
import "./style.css";

let dashboardRequest: WebHostContextV1["http"]["request"] | null = null;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (!dashboardRequest) throw new Error("Akasha 工作台面板未激活");
  const response = await dashboardRequest(path, init);
  const body = await response.json() as T & { detail?: unknown; message?: unknown };
  if (!response.ok) throw new Error(String(body.detail ?? body.message ?? `HTTP ${response.status}`));
  return body;
}

function escapeHtml(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character] ?? character);
}

function encodePath(value: string): string {
  return value.split("/").map(encodeURIComponent).join("/");
}

interface MessageEvidence {
  message_id: string;
  session_id: string;
  seq: number;
  recorded_at: string;
  author: string;
  source: string;
  lane: "dense" | "completion";
  presented: boolean;
  text: string;
  text_truncated: boolean;
}

interface RecallHit {
  lane: "dense" | "completion";
  score: number;
  sources: string[];
  basin_ids: string[];
  messages: MessageEvidence[];
}

interface RecallRow {
  query_id: string;
  session_key: string;
  seq: number;
  ts: string;
  query_text: string;
  source: Record<string, unknown>;
  graph_version: number;
  hit_count: number;
  presented_count: number;
  dense_count: number;
  completion_count: number;
  active_basin_count: number;
  pushes: number;
  residual_l1: number;
  hits: RecallHit[];
}

interface RecallDetail extends RecallRow {
  limit: number;
}

interface InspectorOverview {
  available: boolean;
  total: number;
}

function shortTime(value: unknown): string {
  if (!value) return "—";
  const parsed = new Date(String(value));
  if (Number.isNaN(parsed.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(parsed);
}

function fixed(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function sourceLabel(source: Record<string, unknown>): string {
  if (source.kind === "context") return `上下文 · ${String(source.source ?? "")}`;
  if (source.kind === "tool") return `工具 · ${String(source.call_ref ?? "")}`;
  return `程序 · ${String(source.key ?? "")}`;
}

function renderMessages(messages: MessageEvidence[], empty: string): string {
  if (!messages.length) return `<p class="akasha-empty">${escapeHtml(empty)}</p>`;
  return `
    <ol class="akasha-evidence-list">
      ${messages.map((message, index) => `
        <li class="akasha-evidence">
          <span class="akasha-evidence-rank" aria-hidden="true">${index + 1}</span>
          <div class="akasha-evidence-main">
            <p>${escapeHtml(message.text || "（空消息）")}${message.text_truncated ? " …" : ""}</p>
          </div>
          <div class="akasha-evidence-meta">
            <span class="akasha-chip">${escapeHtml(message.author)} · ${escapeHtml(message.source)}</span>
            <span class="akasha-chip">#${escapeHtml(message.seq)}</span>
            <time class="akasha-chip akasha-chip--time">${escapeHtml(shortTime(message.recorded_at))}</time>
            <span class="akasha-chip">${message.presented ? "已呈现" : "命中"}</span>
            <code>${escapeHtml(message.message_id)}</code>
          </div>
        </li>
      `).join("")}
    </ol>
  `;
}

function evidenceLane(title: string, description: string, hit: RecallHit | undefined): string {
  const messages = hit?.messages ?? [];
  return `
    <details class="akasha-section akasha-lane akasha-lane--${escapeHtml(hit?.lane ?? "empty")}">
      <summary>
        <span class="akasha-lane-copy"><strong>${escapeHtml(title)}</strong><small>${escapeHtml(description)}</small></span>
        <span class="akasha-lane-count">${messages.length}</span>
      </summary>
      ${renderMessages(messages, "这一条通道没有命中消息。")}
    </details>
  `;
}

function metric(label: string, value: unknown, detail: string): string {
  return `<div class="akasha-metric"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd><p>${escapeHtml(detail)}</p></div>`;
}

function renderFilters(container: HTMLElement, dispatch: PluginDispatch): WebUiDisposer | void {
  const value = dispatch.filters["q"] ?? "";
  const existing = container.querySelector<HTMLInputElement>("[data-akasha-search]");
  if (existing) {
    if (document.activeElement !== existing && existing.value !== value) existing.value = value;
    return;
  }
  container.innerHTML = `<div class="akasha-filter"><label><span>搜索检索记录</span><input type="search" value="${escapeHtml(value)}" placeholder="Query、消息或 Session" data-akasha-search /></label><md-text-button data-akasha-clear ${value ? "" : "disabled"}>清空</md-text-button></div>`;
  const input = container.querySelector<HTMLInputElement>("[data-akasha-search]")!;
  const clear = container.querySelector<HTMLElement>("[data-akasha-clear]")!;
  let timer = 0;
  const onInput = (): void => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      const query = input.value.trim();
      if (query) dispatch.setFilter("q", query); else dispatch.clearFilter("q");
    }, 200);
  };
  const onClear = (): void => { input.value = ""; dispatch.clearFilter("q"); };
  input.addEventListener("input", onInput);
  clear.addEventListener("click", onClear);
  return () => { window.clearTimeout(timer); input.removeEventListener("input", onInput); clear.removeEventListener("click", onClear); };
}

function renderDetail(item: RecallDetail, closePane?: () => void): string {
  const dense = item.hits.filter((hit) => hit.lane === "dense");
  const completion = item.hits.filter((hit) => hit.lane === "completion");
  return `
    <article class="akasha-inspector">
      <header class="akasha-query"><div><h2>${escapeHtml(item.query_text)}</h2><p class="akasha-query-meta">${escapeHtml(shortTime(item.ts))} · seq ${escapeHtml(item.seq)} · ${escapeHtml(item.session_key || "程序查询")}</p><p class="akasha-query-meta">${escapeHtml(sourceLabel(item.source))}</p></div>${closePane ? '<md-icon-button class="akasha-close" data-akasha-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}</header>
      <section class="akasha-overview" aria-labelledby="akasha-overview-title"><div class="akasha-overview-heading"><div><h3 id="akasha-overview-title">${item.presented_count} 条消息实际呈现</h3></div><p>图版本 ${item.graph_version} · 查询上限 ${item.limit}</p></div><dl class="akasha-metrics">
        ${metric("命中回忆", item.hit_count, "Recall 记录中选中的回忆条目")}
        ${metric("活跃情景簇", item.active_basin_count, "Recall 记录的真实 completion 指标")}
        ${metric("扩散次数", item.pushes, "查询完成时记录的 pushes")}
        ${metric("残余质量", fixed(item.residual_l1), "查询完成时记录的 residual_l1")}
      </dl></section>
      <section class="akasha-evidence-group" aria-labelledby="akasha-evidence-title"><div class="akasha-section-heading"><h3 id="akasha-evidence-title">原始 Message 证据</h3><small>正文来自 MessageReader；列表页只显示 240 字预览</small></div><div class="akasha-lanes">
        ${dense.map((hit) => evidenceLane("Dense 通道", hit.sources.join(" · "), hit)).join("")}
        ${completion.map((hit) => evidenceLane("Completion 通道", hit.sources.join(" · "), hit)).join("")}
      </div></section>
    </article>
  `;
}

const panel = {
  id: "akasha-inspector",
  label: "Akasha 检索",
  viewLabel: "Akasha 检索",
  pageSize: 25,
  rowKey: "query_id",
  countTitle(total: number): string { return `${total} 轮检索`; },
  columns: [
    { key: "session_key", label: "会话", width: 120, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    { key: "seq", label: "Seq", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "query_text", label: "查询", flex: true, fmt: "text-preview", cellClass: "content-preview" },
    { key: "dense_count", label: "Dense", width: 70, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "completion_count", label: "Completion", width: 96, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "presented_count", label: "已呈现", width: 78, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "active_basin_count", label: "情景簇", width: 78, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "pushes", label: "Pushes", width: 78, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
  ],
  renderFilters,
  async getCount({ signal }: { signal: AbortSignal }): Promise<number | null> {
    const result = await api<InspectorOverview>("/api/dashboard/akasha-inspector/overview", { signal });
    return result.available ? result.total : null;
  },
  async fetchPage({ page, pageSize, filters, signal }: FetchPageOpts): Promise<FetchPageResult> {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (filters?.["session_key"]) params.set("session_key", filters["session_key"]);
    if (filters?.["q"]) params.set("q", filters["q"]);
    const result = await api<{ items: Record<string, unknown>[]; total: number }>(`/api/dashboard/akasha-inspector/turns?${params.toString()}`, { signal });
    return { items: result.items, total: result.total };
  },
  async fetchDetail(item: Record<string, unknown>, { signal }: { signal: AbortSignal }): Promise<Record<string, unknown>> {
    return api(`/api/dashboard/akasha-inspector/turns/${encodePath(String(item["query_id"] ?? ""))}`, { signal });
  },
  renderDetail(item: Record<string, unknown> | null, container: HTMLElement, dispatch: PluginDispatch): void {
    if (!item) {
      container.innerHTML = '<div class="akasha-detail-empty"><div class="akasha-detail-empty__title">Akasha Inspector</div><div class="akasha-detail-empty__text">选择一轮检索，查看实际 Recall 与原始 Message。</div></div>';
      return;
    }
    container.innerHTML = renderDetail(item as unknown as RecallDetail, dispatch.closePane);
    container.querySelector("[data-akasha-close]")?.addEventListener("click", () => dispatch.closePane());
  },
} satisfies WorkbenchPanelEntry;

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  dashboardRequest = ctx.http.request;
  const release = ctx.ui.inject("workbench.panels.v2", (mount) => mount.register(panel));
  return () => { release(); dashboardRequest = null; };
}
