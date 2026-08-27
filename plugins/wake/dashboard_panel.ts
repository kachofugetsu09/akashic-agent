/// <reference path="../../types/akashic-dashboard.d.ts" />

interface WakeRun {
  run_id: string;
  owner: "alert" | "content" | "drift";
  started_at: string;
  candidates_seen: number;
  candidates_selected: number;
  decision: "share" | "skip" | "defer" | null;
  decision_detail: string | null;
  completed_at: string | null;
  screening?: Array<{
    candidate_id?: string;
    initial_interest?: string;
    question?: string;
    payload?: Record<string, unknown>;
  }>;
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

function ownerText(owner: WakeRun["owner"]): string {
  return { alert: "Alert", content: "Content", drift: "Drift" }[owner];
}

function decisionText(decision: WakeRun["decision"]): string {
  return decision === "share"
    ? "决定发送"
    : decision === "skip"
      ? "已跳过"
      : decision === "defer"
        ? "稍后重试"
        : "判断中";
}

function renderScreening(items: NonNullable<WakeRun["screening"]>): string {
  if (!items.length) return '<p class="wake-empty">这一轮没有形成有效初筛结果。</p>';
  return `<ol class="wake-candidates">${items.map((item) => `
    <li>
      <div class="wake-candidate-head">
        <strong>${escapeHtml(item.initial_interest || "Alert 输入")}</strong>
        ${item.candidate_id ? `<code>${escapeHtml(item.candidate_id)}</code>` : ""}
      </div>
      ${item.question ? `<p>${escapeHtml(item.question)}</p>` : ""}
      ${item.payload ? `<pre>${escapeHtml(JSON.stringify(item.payload, null, 2))}</pre>` : ""}
    </li>
  `).join("")}</ol>`;
}

function renderWakeDetail(run: WakeRun, closePane?: () => void): string {
  return `
    <article class="wake-run">
      <header class="wake-run-header">
        <div>
          <p>${escapeHtml(ownerText(run.owner))} · ${escapeHtml(timeText(run.started_at))}</p>
          <h2>${escapeHtml(decisionText(run.decision))}</h2>
        </div>
        ${closePane ? '<md-icon-button data-wake-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}
      </header>

      <dl class="wake-summary">
        <div><dt>看到</dt><dd>${run.candidates_seen}</dd></div>
        <div><dt>进入调查</dt><dd>${run.candidates_selected}</dd></div>
        <div><dt>完成时间</dt><dd>${escapeHtml(timeText(run.completed_at))}</dd></div>
      </dl>

      <section class="wake-section">
        <h3>初筛</h3>
        <p>这里只回答“可能感兴趣吗”；事实核实和偏好召回发生在下一轮。</p>
        ${renderScreening(run.screening ?? [])}
      </section>

      <section class="wake-section wake-decision">
        <h3>最终决定</h3>
        <p class="wake-decision-state wake-decision-state--${escapeHtml(run.decision || "pending")}">${escapeHtml(decisionText(run.decision))}</p>
        <p>${escapeHtml(run.decision_detail || "模型仍在调查。")}</p>
      </section>
    </article>
  `;
}

window.AkashicDashboard.registerPlugin({
  id: "wake_decisions",
  label: "Wake 判断",
  viewLabel: "Wake 判断",
  pageSize: 25,
  rowKey: "run_id",
  countTitle(total: number): string { return `${total} 轮判断`; },
  columns: [
    { key: "started_at", label: "时间", width: 130, renderCell: timeText },
    { key: "owner", label: "输入", width: 90, renderCell: ownerText },
    {
      key: "candidates_seen",
      label: "看到 / 调查",
      width: 110,
      renderCell(_value, item) {
        return `${Number(item["candidates_seen"] ?? 0)} / ${Number(item["candidates_selected"] ?? 0)}`;
      },
    },
    { key: "decision", label: "决定", width: 100, renderCell: decisionText },
    { key: "decision_detail", label: "理由或正文", flex: true, fmt: "text-preview" },
  ],
  async getCount(): Promise<number | null> {
    const result = await api<{ total: number }>("/api/dashboard/wake/runs?page=1&page_size=1");
    return result.total;
  },
  async fetchPage({ page, pageSize }: FetchPageOpts): Promise<FetchPageResult> {
    const result = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/wake/runs?page=${page}&page_size=${pageSize}`,
    );
    return result;
  },
  async fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    return api(`/api/dashboard/wake/runs/${encodeURIComponent(String(item["run_id"] ?? ""))}`);
  },
  renderDetail(item, container, dispatch): void {
    if (!item) {
      container.innerHTML = '<p class="wake-empty">选择一轮判断，查看初筛、调查范围和最终决定。</p>';
      return;
    }
    container.innerHTML = renderWakeDetail(
      item as unknown as WakeRun,
      dispatch?.closePane,
    );
    container.querySelector("[data-wake-close]")?.addEventListener("click", () => dispatch?.closePane?.());
  },
});
