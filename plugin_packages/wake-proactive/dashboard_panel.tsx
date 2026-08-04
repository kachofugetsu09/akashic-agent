/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, JsonView, Markdown, api } from "@akashic/dashboard-ui";

interface Page {
  items: Record<string, unknown>[];
  total: number;
}

function shortTime(value: unknown): string {
  if (!value) return "—";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function count(value: unknown): number {
  if (Array.isArray(value)) return value.length;
  return value && typeof value === "object" ? Object.keys(value).length : 0;
}

function actionLabel(value: string): string {
  if (value === "reply") return "决定回复";
  if (value === "skip") return "决定跳过";
  return "等待判断";
}

function JsonSection({ title, value, open = false }: { title: string; value: unknown; open?: boolean }): ReactElement {
  return (
    <details className="wake-audit" open={open}>
      <summary>
        <span>{title}</span>
        <Chip tone="muted">{count(value)}</Chip>
      </summary>
      <JsonView value={value} />
    </details>
  );
}

function Detail({ item }: { item: Record<string, unknown> | null }): ReactElement {
  if (!item) {
    return <div className="wake-empty">选择一次唤醒，查看完整判断链。</div>;
  }
  const observations = Array.isArray(item.observations)
    ? item.observations as Record<string, unknown>[]
    : [];
  const action = String(item.terminal_action || "pending");
  return (
    <main className="wake-detail" aria-labelledby="wake-detail-title">
      <header className="wake-summary">
        <div>
          <span>主动唤醒判断</span>
          <h2 id="wake-detail-title">{actionLabel(action)}</h2>
          <small>{shortTime(item.now_utc)} · {String(item.session_key || "未关联会话")}</small>
        </div>
        <Chip tone={action === "skip" ? "muted" : "accent"} dot>
          {actionLabel(action)}
        </Chip>
      </header>
      <section className="wake-message" aria-label="最终行为">
        <span>{action === "reply" ? "最终发送内容" : "本轮结果"}</span>
        <Markdown>{String(item.final_message || "本轮决定不主动打扰。")}</Markdown>
      </section>
      <section className="wake-reasoning" aria-labelledby="wake-reasoning-title">
        <div className="wake-section-heading">
          <span>判断路径</span>
          <h3 id="wake-reasoning-title">从触发信号到最终动作</h3>
        </div>
      {observations.map((observation, index) => (
        <div className="wake-phase" key={`${String(observation.kind)}-${index}`}>
          <div className="wake-phase-title"><b>{index + 1}</b><span>观测 · {String(observation.kind || "unknown")}</span></div>
          <JsonSection title="触发原因" value={observation.trigger} open />
          <JsonSection title="进入判断的候选" value={observation.candidates} />
          <JsonSection title="送给 LLM 的输入" value={observation.llm_input} />
        </div>
      ))}
      </section>
      <section className="wake-technical" aria-labelledby="wake-technical-title">
        <div className="wake-section-heading">
          <span>审计信息</span>
          <h3 id="wake-technical-title">计划、调查与引用</h3>
        </div>
        <JsonSection title="初筛计划 Scratchpad" value={item.scratchpad} />
        <JsonSection title="正文与记忆调查结果" value={item.investigations} />
        <JsonSection title="最终引用 ID" value={item.cited_ids} />
        <JsonSection title="展示序号映射" value={item.display_event_map} />
        <JsonSection title="来源引用" value={item.source_refs} />
      </section>
    </main>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "wake-proactive",
  label: "主动唤醒",
  viewLabel: "主动唤醒",
  rowKey: "wake_id",
  pageSize: 50,
  defaultSortBy: "now_utc",
  defaultSortOrder: "desc",
  columns: [
    {
      key: "session_key",
      label: "会话",
      width: 170,
      fmt: "mono-session",
      cellClass: "mono cell-session",
      rawTitle: true,
    },
    {
      key: "now_utc",
      label: "时间",
      width: 112,
      fmt: "wake-time",
      cellClass: "mono cell-time",
      rawTitle: true,
    },
    {
      key: "terminal_action",
      label: "结果",
      width: 88,
      cellClass: "cell-status",
      renderCell(value) {
        const action = String(value || "pending");
        const tone = action === "reply"
          ? "proactive-result-reply"
          : action === "skip"
            ? "proactive-result-skip"
            : "proactive-result-unknown";
        return `<span class="status-pill ${tone}">${escapeHtml(actionLabel(action))}</span>`;
      },
    },
    {
      key: "final_message",
      label: "最终内容",
      flex: true,
      fmt: "text-preview",
      cellClass: "content-preview",
      rawTitle: true,
    },
  ],
  async getCount() {
    const data = await api<{ total: number }>("/api/dashboard/wake-proactive/runs?page=1&page_size=1");
    return data.total || 0;
  },
  async fetchPage({ page, pageSize }) {
    return api<Page>(`/api/dashboard/wake-proactive/runs?page=${page}&page_size=${pageSize}`);
  },
  async fetchDetail(item) {
    return api<Record<string, unknown>>(
      `/api/dashboard/wake-proactive/runs/${encodeURIComponent(String(item.wake_id))}`,
    );
  },
  Detail,
  formatters: { "wake-time": shortTime },
});
