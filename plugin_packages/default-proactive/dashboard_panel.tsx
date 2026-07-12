/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, JsonView, Markdown, Panel, Stack, api } from "@akashic/dashboard-ui";

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
  if (!item) return <div className="default-empty">选择一条 Tick 查看旧主动推送链路。</div>;
  return <Stack className="default-detail">
    <Panel className="default-head"><div><span>TICK</span><strong>{shortTime(item.started_at)}</strong></div><Chip>{String(item.terminal_action || "-")}</Chip></Panel>
    <Panel className="default-message"><span>最终消息</span><Markdown>{String(item.final_message || "本轮没有发送消息。")}</Markdown></Panel>
    <div className="default-trace"><span>执行记录</span><JsonView value={item} /></div>
  </Stack>;
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
    { key: "session_key", label: "Session", width: 150 },
    { key: "started_at", label: "Started", width: 104, fmt: "short-time" },
    { key: "terminal_action", label: "Result", width: 110 },
    { key: "final_message", label: "Message", flex: true },
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
