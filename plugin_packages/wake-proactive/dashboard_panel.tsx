/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, api } from "@akashic/dashboard-ui";

function Detail({ item }: { item: Record<string, unknown> | null }): ReactElement {
  if (!item) return <div className="detail-empty"><div className="detail-empty-title">Wake Proactive</div><div className="detail-empty-text">选择一次唤醒查看筛选、调查与最终行为。</div></div>;
  return <div className="detail-wrap">
    <div className="detail-title">{String(item.wake_id || "")}</div>
    <div className="detail-grid">
      <div className="detail-row"><div className="detail-row-label">action</div><div className="detail-row-val"><Chip>{String(item.terminal_action || "-")}</Chip></div></div>
      <div className="detail-row"><div className="detail-row-label">session</div><div className="detail-row-val"><code>{String(item.session_key || "-")}</code></div></div>
    </div>
    <div className="detail-block"><div className="detail-label">Final message</div><div className="detail-content ak-plugin-pre-wrap">{String(item.final_message || "-")}</div></div>
    <div className="detail-block"><div className="detail-label">Scratchpad</div><pre className="detail-content">{JSON.stringify(item.scratchpad || {}, null, 2)}</pre></div>
    <div className="detail-block"><div className="detail-label">Investigations</div><pre className="detail-content">{JSON.stringify(item.investigations || {}, null, 2)}</pre></div>
  </div>;
}

window.AkashicDashboard.registerPlugin({
  id: "wake-proactive",
  label: "Wake Proactive",
  viewLabel: "wake proactive",
  rowKey: "wake_id",
  pageSize: 50,
  columns: [
    { key: "session_key", label: "Session", width: 150, cellClass: "mono" },
    { key: "now_utc", label: "Wake time", width: 180, cellClass: "mono" },
    { key: "terminal_action", label: "Action", width: 110 },
    { key: "final_message", label: "Message", flex: true, cellClass: "content-preview" },
  ],
  async getCount() {
    const data = await api<{ total: number }>("/api/dashboard/wake-proactive/runs?page=1&page_size=1");
    return data.total || 0;
  },
  async fetchPage({ page, pageSize }) {
    return api<{ items: Record<string, unknown>[]; total: number }>(`/api/dashboard/wake-proactive/runs?page=${page}&page_size=${pageSize}`);
  },
  async fetchDetail(item) {
    return api<Record<string, unknown>>(`/api/dashboard/wake-proactive/runs/${encodeURIComponent(String(item.wake_id))}`);
  },
  Detail,
});
