import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type {
  FetchPageOptions as FetchPageOpts,
  FetchPageResult,
  WorkbenchPanelEntry,
} from "@akashic/workbench-ui-v2";
import "./style.css";

let dashboardRequest: WebHostContextV1["http"]["request"] | null = null;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  if (!dashboardRequest) throw new Error("Wake 工作台面板未激活");
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

interface WakeAttempt {
  attempt_id: string;
  timer_id: string;
  scheduled_for: string;
  fired_at: string;
  mail_watermark: number | null;
  outcome:
    | "checking"
    | "no_due"
    | "content_insufficient"
    | "admission_rejected"
    | "shared"
    | "model_skip"
    | "deferred"
    | "cancelled_after_fire"
    | "delivery_unknown"
    | "failed";
  owner: "alert" | "content" | "drift" | null;
  detail: string | null;
  completed_at: string | null;
}

function timeText(value: unknown): string {
  if (!value) return "进行中";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function ownerText(owner: WakeAttempt["owner"]): string {
  if (!owner) return "无待办";
  return { alert: "Alert", content: "Content", drift: "Drift" }[owner];
}

function watermarkText(value: unknown): string {
  return value === null || value === undefined ? "未读取" : String(value);
}

function outcomeText(outcome: WakeAttempt["outcome"]): string {
  return {
    checking: "检查中",
    no_due: "没有到期信件",
    content_insufficient: "Content 不足",
    admission_rejected: "Admission 未通过",
    shared: "已发送",
    model_skip: "模型跳过",
    deferred: "已延期",
    cancelled_after_fire: "触发后关闭",
    delivery_unknown: "送达未知",
    failed: "检查失败",
  }[outcome];
}

function renderWakeDetail(attempt: WakeAttempt, closePane?: () => void): string {
  return `
    <article class="wake-run">
      <header class="wake-run-header">
        <div>
          <p>${escapeHtml(ownerText(attempt.owner))} · ${escapeHtml(timeText(attempt.fired_at))}</p>
          <h2>${escapeHtml(outcomeText(attempt.outcome))}</h2>
        </div>
        ${closePane ? '<md-icon-button data-wake-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}
      </header>

      <dl class="wake-summary">
        <div><dt>计划时间</dt><dd>${escapeHtml(timeText(attempt.scheduled_for))}</dd></div>
        <div><dt>信箱水位</dt><dd>${escapeHtml(watermarkText(attempt.mail_watermark))}</dd></div>
        <div><dt>检查完成</dt><dd>${escapeHtml(timeText(attempt.completed_at))}</dd></div>
      </dl>

      <section class="wake-section">
        <h3>这次检查</h3>
        <p>${escapeHtml(attempt.detail || "Timer 已触发，正在检查 EventMail。")}</p>
        <p><code>${escapeHtml(attempt.timer_id)}</code></p>
      </section>
    </article>
  `;
}

const panel = {
  id: "wake-attempts",
  label: "Wake 检查",
  viewLabel: "Wake 检查",
  pageSize: 25,
  rowKey: "attempt_id",
  countTitle(total: number): string { return `${total} 次定时检查`; },
  columns: [
    { key: "fired_at", label: "触发时间", width: 130, renderCell: timeText },
    { key: "owner", label: "输入", width: 90, renderCell: ownerText },
    { key: "mail_watermark", label: "信箱水位", width: 90, renderCell: watermarkText },
    { key: "outcome", label: "结果", width: 120, renderCell: outcomeText },
    { key: "detail", label: "说明", flex: true, fmt: "text-preview" },
  ],
  async getCount({ signal }: { signal: AbortSignal }): Promise<number | null> {
    const result = await api<{ total: number }>("/api/dashboard/wake/attempts?page=1&page_size=1", { signal });
    return result.total;
  },
  async fetchPage({ page, pageSize, signal }: FetchPageOpts): Promise<FetchPageResult> {
    const result = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/wake/attempts?page=${page}&page_size=${pageSize}`,
      { signal },
    );
    return result;
  },
  async fetchDetail(item: Record<string, unknown>, { signal }: { signal: AbortSignal }): Promise<Record<string, unknown>> {
    return api(`/api/dashboard/wake/attempts/${encodeURIComponent(String(item["attempt_id"] ?? ""))}`, { signal });
  },
  renderDetail(item, container, dispatch): void {
    if (!item) {
      container.innerHTML = '<p class="wake-empty">选择一次定时检查，查看它当时看到的 EventMail 水位和结果。</p>';
      return;
    }
    container.innerHTML = renderWakeDetail(
      item as unknown as WakeAttempt,
      dispatch.closePane,
    );
    container.querySelector("[data-wake-close]")?.addEventListener("click", () => dispatch.closePane());
  },
} satisfies WorkbenchPanelEntry;

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  dashboardRequest = ctx.http.request;
  const release = ctx.ui.inject("workbench.panels.v2", (mount) => mount.register(panel));
  return () => {
    release();
    dashboardRequest = null;
  };
}
