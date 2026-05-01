/// <reference path="../../types/akashic-dashboard.d.ts" />

interface RecallItem {
  id: string;
  summary: string;
  injected?: boolean;
  tags?: string[];
}

interface ContextPrepare {
  items: RecallItem[];
  injected_items: RecallItem[];
}

interface RecallCall {
  items: RecallItem[];
}

interface RecallTurnRow {
  turn_id: string;
  session_key: string;
  timestamp: string;
  user_text: string;
  context_prepare_count: number;
  recall_memory_count: number;
}

interface RecallTurnDetail extends RecallTurnRow {
  context_prepare?: ContextPrepare;
  recall_memory_calls?: RecallCall[];
}

function _renderRecallItems(items: RecallItem[], source: string): string {
  if (!items.length) {
    return '<div class="muted-text">没有召回条目。</div>';
  }
  return `
    <div class="recall-item-list">
      ${items.map((item) => `
        <div class="recall-item recall-item-${escapeHtml(source)}">
          <div class="recall-item-head">
            <code>${escapeHtml(item.id || "-")}</code>
            ${item.injected === true ? '<span class="recall-tag recall-tag-injected">已注入</span>' : ""}
            ${(item.tags ?? []).map((tag) => `<span class="recall-tag">${escapeHtml(tag)}</span>`).join("")}
          </div>
          <div class="recall-summary">${escapeHtml(item.summary || "")}</div>
        </div>
      `).join("")}
    </div>
  `;
}

window.AkashicDashboard.registerPlugin({
  id: "recall_inspector",
  label: "Recall Inspector",
  viewLabel: "recall inspector",
  pageSize: 25,
  rowKey: "turn_id",

  countTitle(total: number): string {
    return `${total} 轮召回`;
  },

  columns: [
    { key: "session_key", label: "Session", width: 108, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    { key: "timestamp",   label: "Time",    width: 96,  fmt: "mono-time",    cellClass: "mono cell-time",    rawTitle: true },
    { key: "user_text",   label: "User",    flex: true, fmt: "text-preview", cellClass: "content-preview" },
    { key: "context_prepare_count", label: "Prepare", width: 72, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "recall_memory_count",   label: "Recall",  width: 72, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const r = await api("/api/dashboard/recall-inspector/overview");
      return r["available"] ? ((r["total"] as number) || 0) : null;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    const data = await api(`/api/dashboard/recall-inspector/turns?${params.toString()}`);
    return {
      items: (data["items"] as Record<string, unknown>[]) || [],
      total: (data["total"] as number) || 0,
    };
  },

  async fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    const turnId = String(item["turn_id"] ?? "");
    return api(`/api/dashboard/recall-inspector/turns/${encodePath(turnId)}`);
  },

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void {
    if (!item) {
      container.innerHTML = `
        <div class="detail-empty">
          <div class="detail-empty-title">Recall Inspector</div>
          <div class="detail-empty-text">点开一轮记录后，这里会显示 context prepare 和 recall_memory 召回的记忆。</div>
        </div>
      `;
      return;
    }

    const turn = item as unknown as RecallTurnDetail;
    const contextPrepare = turn.context_prepare ?? { items: [], injected_items: [] };
    const recallCalls = turn.recall_memory_calls ?? [];

    container.innerHTML = `
      <div class="detail-wrap">
        <div class="detail-toolbar">
          <div>
            <div class="detail-title">召回记录</div>
            <div class="detail-subtext">${escapeHtml(turn.session_key || "")} · ${escapeHtml(turn.turn_id || "")}</div>
          </div>
        </div>
        <div class="detail-block">
          <div class="detail-label">User Message</div>
          <div class="detail-content">${renderMarkdown(turn.user_text || "")}</div>
        </div>
        <div class="detail-block">
          <div class="detail-label">预检索总召回</div>
          ${_renderRecallItems(contextPrepare.items, "context")}
        </div>
        <div class="detail-block">
          <div class="detail-label">最终注入</div>
          ${_renderRecallItems(contextPrepare.injected_items, "inject")}
        </div>
        <div class="detail-block">
          <div class="detail-label">Recall 返回</div>
          ${recallCalls.length
            ? recallCalls.map((call) => _renderRecallItems(call.items || [], "recall")).join("")
            : '<div class="muted-text">本轮没有显式调用 recall_memory。</div>'
          }
        </div>
      </div>
    `;
  },
});
