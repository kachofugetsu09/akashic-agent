/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, api } from "@akashic/dashboard-ui";

interface Page {
  items: Record<string, unknown>[];
  total: number;
}

function shortTime(value: unknown): string {
  const date = new Date(String(value || ""));
  return Number.isNaN(date.getTime())
    ? String(value || "-")
    : `${date.getMonth() + 1}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function Detail({ item }: { item: Record<string, unknown> | null }): ReactElement {
  if (!item) return <div className="detail-empty"><div className="detail-empty-title">Default Proactive</div><div className="detail-empty-text">选择一条 tick 查看旧主动推送链路。</div></div>;
  return <div className="detail-wrap">
    <div className="detail-title">{String(item.tick_id || "")}</div>
    <div className="detail-grid">
      <div className="detail-row"><div className="detail-row-label">result</div><div className="detail-row-val"><Chip>{String(item.terminal_action || "-")}</Chip></div></div>
      <div className="detail-row"><div className="detail-row-label">session</div><div className="detail-row-val"><code>{String(item.session_key || "-")}</code></div></div>
    </div>
    <div className="detail-block"><div className="detail-label">Final message</div><div className="detail-content ak-plugin-pre-wrap">{String(item.final_message || "-")}</div></div>
    <div className="detail-block"><div className="detail-label">Trace</div><pre className="detail-content">{JSON.stringify(item, null, 2)}</pre></div>
  </div>;
}

window.AkashicDashboard.registerPlugin({
  id: "default-proactive",
  label: "Default Tick",
  viewLabel: "default proactive",
  rowKey: "tick_id",
  pageSize: 50,
  defaultSortBy: "started_at",
  defaultSortOrder: "desc",
  columns: [
    { key: "session_key", label: "Session", width: 150, cellClass: "mono" },
    { key: "started_at", label: "Started", width: 104, fmt: "short-time", cellClass: "mono" },
    { key: "terminal_action", label: "Result", width: 110 },
    { key: "final_message", label: "Message", flex: true, cellClass: "content-preview" },
  ],
  async getCount() {
    const data = await api<{ counts: { tick_logs: number } }>("/api/dashboard/proactive/overview");
    return data.counts.tick_logs || 0;
  },
  async fetchPage({ page, pageSize, sortBy, sortOrder }) {
    const query = new URLSearchParams({ page: String(page), page_size: String(pageSize), sort_by: sortBy, sort_order: sortOrder });
    return api<Page>(`/api/dashboard/proactive/tick_logs?${query}`);
  },
  async fetchDetail(item) {
    return api<Record<string, unknown>>(`/api/dashboard/proactive/tick_logs/${encodeURIComponent(String(item.tick_id))}`);
  },
  Detail,
  formatters: { "short-time": shortTime },
});
