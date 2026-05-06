/// <reference path="../../types/akashic-dashboard.d.ts" />

interface KVCacheSummary {
  tracked_turn_count: number;
  prompt_tokens: number;
  hit_tokens: number;
  miss_tokens: number;
  hit_rate: number | null;
  last_tracked_at: string | null;
}

interface KVCacheTurn {
  id: number;
  ts: string;
  session_key: string;
  user_preview: string;
  prompt_tokens: number;
  hit_tokens: number;
  miss_tokens: number;
  hit_rate: number | null;
}

interface KVCacheTurnDetail extends KVCacheTurn {
  summary?: KVCacheSummary;
}

function _formatNumber(value: unknown): string {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function _formatRate(value: unknown): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function _cacheRateClass(rate: number | null): string {
  if (rate == null) {
    return "cache-rate-0";
  }
  const bucket = Math.max(0, Math.min(100, Math.round(rate * 10) * 10));
  return `cache-rate-${bucket}`;
}

function _renderSummary(summary: KVCacheSummary): string {
  return `
    <div class="detail-grid">
      ${_detailMetric("tracked_turn_count", _formatNumber(summary.tracked_turn_count))}
      ${_detailMetric("prompt_tokens", _formatNumber(summary.prompt_tokens))}
      ${_detailMetric("hit_tokens", _formatNumber(summary.hit_tokens))}
      ${_detailMetric("miss_tokens", _formatNumber(summary.miss_tokens))}
    </div>
    <div class="cache-meter" aria-label="KVCache hit rate">
      <div class="cache-meter-fill ${escapeHtml(_cacheRateClass(summary.hit_rate))}"></div>
    </div>
  `;
}

function _detailMetric(label: string, value: string): string {
  return `
    <div class="detail-row">
      <div class="detail-row-label">${escapeHtml(label)}</div>
      <div class="detail-row-val"><code>${escapeHtml(value)}</code></div>
    </div>
  `;
}

function _renderTurns(turns: KVCacheTurn[]): string {
  if (!turns.length) {
    return '<div class="cache-turn-empty">暂无 KVCache 记录。</div>';
  }
  return `
    <div class="cache-turns">
      <div class="cache-turns-head">
        <span>Recent turns</span>
        <span>Hit / prompt</span>
      </div>
      ${turns.map((turn) => `
        <div class="cache-turn-row">
          <div class="cache-turn-main">
            <div class="cache-turn-rate">${escapeHtml(_formatRate(turn.hit_rate))}</div>
            <div class="cache-turn-text">${escapeHtml(turn.user_preview || "（无内容）")}</div>
          </div>
          <div class="cache-turn-meta">
            <span>${escapeHtml(window.AkashicDashboard._formatters["mono-time"](turn.ts, turn as unknown as Record<string, unknown>))}</span>
            <span>${escapeHtml(_formatNumber(turn.hit_tokens))} / ${escapeHtml(_formatNumber(turn.prompt_tokens))}</span>
            <span>${escapeHtml(turn.session_key)}</span>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

window.AkashicDashboard.registerPlugin({
  id: "status_commands",
  label: "KV Cache",
  viewLabel: "kv cache",
  pageSize: 25,
  rowKey: "id",

  countTitle(total: number): string {
    return `${total} 轮 KVCache`;
  },

  columns: [
    { key: "session_key", label: "Session", width: 108, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    { key: "ts", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true },
    { key: "hit_rate", label: "Hit", width: 72, fmt: "cache-rate", cellClass: "mono cell-metric", align: "right" },
    { key: "hit_tokens", label: "Hit Tokens", width: 92, fmt: "number", cellClass: "mono cell-metric", align: "right" },
    { key: "prompt_tokens", label: "Prompt", width: 92, fmt: "number", cellClass: "mono cell-metric", align: "right" },
    { key: "user_preview", label: "User", flex: true, fmt: "text-preview", cellClass: "content-preview" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const summary = await api<KVCacheSummary>("/api/dashboard/status-commands/kvcache/overview");
      return summary.tracked_turn_count || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    const data = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/status-commands/kvcache/turns?${params.toString()}`,
    );
    return {
      items: data.items || [],
      total: data.total || 0,
    };
  },

  async fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    const turnId = Number(item["id"] ?? 0);
    return api(`/api/dashboard/status-commands/kvcache/turns/${turnId}`);
  },

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void {
    if (!item) {
      container.innerHTML = `
        <div class="detail-empty">
          <div class="detail-empty-title">KV Cache</div>
          <div class="detail-empty-text">点开一轮记录后，这里会显示命中率、token 分布和原始用户消息预览。</div>
        </div>
      `;
      return;
    }

    const turn = item as unknown as KVCacheTurnDetail;
    const summary: KVCacheSummary = turn.summary ?? {
      tracked_turn_count: 1,
      prompt_tokens: turn.prompt_tokens,
      hit_tokens: turn.hit_tokens,
      miss_tokens: turn.miss_tokens,
      hit_rate: turn.hit_rate,
      last_tracked_at: turn.ts,
    };
    container.innerHTML = `
      <div class="detail-wrap">
        <div class="detail-toolbar">
          <div>
            <div class="detail-title">KV Cache</div>
            <div class="detail-subtext">${escapeHtml(turn.session_key || "")} · ${escapeHtml(String(turn.id || ""))}</div>
          </div>
        </div>
        <div class="detail-block">
          <div class="detail-label">命中概览</div>
          ${_renderSummary(summary)}
        </div>
        <div class="detail-block">
          <div class="detail-label">当前轮次</div>
          ${_renderTurns([turn])}
        </div>
      </div>
    `;
  },

  formatters: {
    "cache-rate": (value: unknown) => _formatRate(value),
    number: (value: unknown) => _formatNumber(value),
  },
});
