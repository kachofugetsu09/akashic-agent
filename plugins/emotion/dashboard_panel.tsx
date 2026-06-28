/// <reference path="../../types/akashic-dashboard.d.ts" />
import { type ReactElement } from "react";
import { Chip, api } from "@akashic/dashboard-ui";

interface Overview {
  state: Record<string, unknown> | null;
  effect_count: number;
}

interface FetchPage {
  items: Record<string, unknown>[];
  total: number;
}

function _score(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function _delta(value: unknown): string {
  if (typeof value !== "number") return "-";
  return value > 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
}

function _shortTs(value: unknown): string {
  const text = String(value || "");
  if (!text) return "-";
  const d = new Date(text);
  if (Number.isNaN(d.getTime())) return text;
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function _escape(value: unknown): string {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function _toneCell(value: unknown): string {
  const text = String(value || "-");
  const tone = text === "raise_send_bar" ? "warning" : text === "lower_send_bar" ? "success" : "muted";
  return `<span class="${window.AkashicDashboard.ui.cx.badge(tone)}">${_escape(text)}</span>`;
}

function EmotionDetail(props: { item: Record<string, unknown> | null }): ReactElement {
  const item = props.item;
  if (!item) {
    return <div className="detail-empty"><div className="detail-empty-title">VAD Effect</div><div className="detail-empty-text">点开一条 effect 后，这里会显示这次主动 tick 的情绪影响。</div></div>;
  }
  return (
    <div className="detail-wrap">
      <div className="detail-toolbar">
        <div>
          <div className="detail-title">VAD 主动影响</div>
          <div className="detail-subtext">{String(item.tick_id || "")}</div>
        </div>
      </div>
      <div className="detail-grid">
        <DetailRow label="expected" value={<Chip tone={String(item.expected_effect) === "raise_send_bar" ? "warning" : "success"}>{String(item.expected_effect || "-")}</Chip>} />
        <DetailRow label="tone" value={<code>{String(item.tone_label || "-")}</code>} />
        <DetailRow label="V" value={<code>{_score(item.valence)}</code>} />
        <DetailRow label="A" value={<code>{_score(item.arousal)}</code>} />
        <DetailRow label="D" value={<code>{_score(item.dominance)}</code>} />
        <DetailRow label="threshold" value={<code>{_score(item.base_threshold)} → {_score(item.final_threshold)}</code>} />
      </div>
      <TextBlock title="Prompt Section" text={String(item.prompt_section || "")} />
      <TextBlock title="Metadata" text={JSON.stringify(item.metadata || {}, null, 2)} />
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
  id: "emotion",
  label: "Emotion",
  viewLabel: "emotion",
  pageSize: 50,
  rowKey: "id",

  countTitle(total: number): string {
    return `共 ${total} 条 effect`;
  },

  columns: [
    { key: "created_at", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true },
    { key: "expected_effect", label: "Effect", width: 124, renderCell: _toneCell },
    { key: "tone_label", label: "Tone", width: 136, cellClass: "mono" },
    { key: "valence", label: "V", width: 58, fmt: "score", cellClass: "mono cell-metric", align: "right" },
    { key: "arousal", label: "A", width: 58, fmt: "score", cellClass: "mono cell-metric", align: "right" },
    { key: "dominance", label: "D", width: 58, fmt: "score", cellClass: "mono cell-metric", align: "right" },
    { key: "threshold_delta", label: "Δ", width: 58, fmt: "delta", cellClass: "mono cell-metric", align: "right" },
    { key: "tick_id", label: "Tick", flex: true, cellClass: "mono content-preview", rawTitle: true },
  ],

  async getCount(): Promise<number | null> {
    try {
      const overview = await api<Overview>("/api/dashboard/emotion/overview");
      return overview.effect_count || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    const data = await api<FetchPage>(`/api/dashboard/emotion/effects?${params.toString()}`);
    return { items: data.items || [], total: data.total || 0 };
  },

  async fetchDetail(item: Record<string, unknown>) {
    return api<Record<string, unknown>>(`/api/dashboard/emotion/effects/${item.id}`);
  },

  Detail: EmotionDetail,

  formatters: {
    score: (value: unknown) => _score(value),
    delta: (value: unknown) => _delta(value),
    "mono-time": (value: unknown) => _shortTs(value),
  },
});
