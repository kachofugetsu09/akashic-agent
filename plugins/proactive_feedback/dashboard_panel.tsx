/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, api } from "@akashic/dashboard-ui";

interface Overview {
  total: number;
}

interface FetchPage {
  items: Record<string, unknown>[];
  total: number;
}

function _score(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function _shortTs(value: unknown): string {
  const text = String(value || "");
  if (!text) return "-";
  const d = new Date(text);
  if (Number.isNaN(d.getTime())) return text;
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function _lag(value: unknown): string {
  if (typeof value !== "number") return "-";
  if (value < 60) return `${value}s`;
  if (value < 3600) return `${Math.round(value / 60)}m`;
  return `${(value / 3600).toFixed(1)}h`;
}

function _tone(type: string): "success" | "warning" | "danger" | "accent" | "muted" {
  if (type === "explicit_quote") return "accent";
  if (type === "topic_follow") return "success";
  if (type === "unscored") return "warning";
  if (type === "no_topic_follow") return "muted";
  return "muted";
}

function _escape(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function _cellText(value: unknown): string {
  const text = String(value || "").trim();
  return _escape(text || "-");
}

function _typeCell(value: unknown): string {
  const type = String(value || "");
  const tone = type === "explicit_quote" ? "accent" : type === "topic_follow" ? "success" : type === "unscored" ? "warning" : "muted";
  return `<span class="${window.AkashicDashboard.ui.cx.badge(tone)}">${_escape(type || "-")}</span>`;
}

function FeedbackDetail(props: { item: Record<string, unknown> | null }): ReactElement {
  const item = props.item;
  if (!item) {
    return <div className="detail-empty"><div className="detail-empty-title">反馈详情</div><div className="detail-empty-text">点开一条记录后，这里会显示用户回复、命中的 proactive 和助手后续回复。</div></div>;
  }
  const type = String(item.feedback_type || "");
  return (
    <div className="detail-wrap">
      <div className="detail-toolbar">
        <div>
          <div className="detail-title">反馈链路</div>
          <div className="detail-subtext">{String(item.session_key || "")} · {String(item.user_message_id || "")}</div>
        </div>
      </div>
      <div className="detail-grid">
        <DetailRow label="type" value={<Chip tone={_tone(type)}>{type}</Chip>} />
        <DetailRow label="confidence" value={<code>{String(item.confidence || "-")}</code>} />
        <DetailRow label="matched_by" value={<code>{String(item.matched_by || "-")}</code>} />
        <DetailRow label="pua" value={<code>{_score(item.pua_score)}</code>} />
        <DetailRow label="lag" value={<code>{_lag(item.lag_seconds)}</code>} />
        <DetailRow label="proactive_id" value={<code>{String(item.proactive_message_id || "-")}</code>} />
      </div>
      <TextBlock title="User Reply" text={String(item.user_reply_preview || item.user_preview || "")} />
      {item.quoted_preview ? <TextBlock title="Quoted Proactive" text={String(item.quoted_preview)} /> : null}
      <TextBlock title="Matched Proactive" text={String(item.proactive_preview || "")} />
      <TextBlock title="Assistant Reply" text={String(item.assistant_preview || "")} />
    </div>
  );
}

function DetailRow(props: { label: string; value: ReactElement }): ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{props.label}</div><div className="detail-row-val">{props.value}</div></div>;
}

function TextBlock(props: { title: string; text: string }): ReactElement {
  return (
    <div className="detail-block">
      <div className="detail-label">{props.title}</div>
      <div className="detail-content whitespace-pre-wrap">{props.text || "-"}</div>
    </div>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "proactive_feedback",
  label: "Feedback",
  viewLabel: "feedback",
  pageSize: 50,
  rowKey: "id",

  countTitle(total: number): string {
    return `共 ${total} 条反馈`;
  },

  columns: [
    { key: "created_at", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true },
    { key: "feedback_type", label: "Type", width: 126, renderCell: _typeCell },
    { key: "confidence", label: "Conf", width: 72, cellClass: "mono cell-metric" },
    { key: "pua_score", label: "PUA", width: 66, fmt: "score", cellClass: "mono cell-metric", align: "right" },
    { key: "lag_seconds", label: "Lag", width: 68, fmt: "lag", cellClass: "mono cell-metric", align: "right" },
    { key: "user_reply_preview", label: "User Reply", flex: true, renderCell: _cellText, cellClass: "content-preview", rawTitle: true },
    { key: "proactive_preview", label: "Matched Proactive", flex: true, renderCell: _cellText, cellClass: "content-preview", rawTitle: true },
  ],

  async getCount(): Promise<number | null> {
    try {
      const overview = await api<Overview>("/api/dashboard/proactive-feedback/overview");
      return overview.total || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    const data = await api<FetchPage>(`/api/dashboard/proactive-feedback/events?${params.toString()}`);
    return { items: data.items || [], total: data.total || 0 };
  },

  async fetchDetail(item: Record<string, unknown>) {
    return api<Record<string, unknown>>(`/api/dashboard/proactive-feedback/events/${item.id}`);
  },

  Detail: FeedbackDetail,

  formatters: {
    score: (value: unknown) => _score(value),
    lag: (value: unknown) => _lag(value),
    "mono-time": (value: unknown) => _shortTs(value),
  },
});
