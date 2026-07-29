/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, JsonView, Markdown, Panel, Stack, api } from "@akashic/dashboard-ui";

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
    <Stack className="wake-detail">
      <Panel className="wake-summary">
        <div>
          <span>Wake run</span>
          <strong>{shortTime(item.now_utc)}</strong>
        </div>
        <Chip tone={action === "skip" ? "muted" : "accent"} dot>
          {action}
        </Chip>
      </Panel>
      <Panel className="wake-message">
        <span>最终行为</span>
        <Markdown>{String(item.final_message || "本轮决定不主动打扰。")}</Markdown>
      </Panel>
      {observations.map((observation, index) => (
        <Stack className="wake-phase" key={`${String(observation.kind)}-${index}`}>
          <div className="wake-phase-title">
            触发观测 · {String(observation.kind || "unknown")}
          </div>
          <JsonSection title="触发原因" value={observation.trigger} open />
          <JsonSection title="进入判断的候选" value={observation.candidates} />
          <JsonSection title="送给 LLM 的输入" value={observation.llm_input} />
        </Stack>
      ))}
      <JsonSection title="初筛计划 Scratchpad" value={item.scratchpad} open />
      <JsonSection title="正文与记忆调查结果" value={item.investigations} />
      <JsonSection title="最终引用 ID" value={item.cited_ids} />
      <JsonSection title="展示序号映射" value={item.display_event_map} />
      <JsonSection title="来源引用" value={item.source_refs} />
    </Stack>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "wake-proactive",
  label: "Wake Proactive",
  viewLabel: "wake proactive",
  rowKey: "wake_id",
  pageSize: 50,
  defaultSortBy: "now_utc",
  defaultSortOrder: "desc",
  columns: [
    {
      key: "session_key",
      label: "Session",
      width: 170,
      fmt: "mono-session",
      cellClass: "mono cell-session",
      rawTitle: true,
    },
    {
      key: "now_utc",
      label: "Wake time",
      width: 112,
      fmt: "wake-time",
      cellClass: "mono cell-time",
      rawTitle: true,
    },
    {
      key: "terminal_action",
      label: "Action",
      width: 88,
      cellClass: "cell-status",
      renderCell(value) {
        const action = String(value || "pending");
        const tone = action === "reply"
          ? "proactive-result-reply"
          : action === "skip"
            ? "proactive-result-skip"
            : "proactive-result-unknown";
        return `<span class="status-pill ${tone}">${escapeHtml(action)}</span>`;
      },
    },
    {
      key: "final_message",
      label: "Message",
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
