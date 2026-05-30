/// <reference path="../../types/akashic-dashboard.d.ts" />

// ── Types ────────────────────────────────────────────────────────────────────

interface AkashaCandidate {
  key: string;
  user_message: string;
  assistant_preview: string;
  score: number;
  source: string;
  path_type: string;
  fan: number;
  direct: number;
  state: number;
  edge: number;
  long: number;
  resource: number;
  ripple: number;
  seed_key: string;
  bridge_key: string;
  suppressed: string;
}

interface AkashaCard extends AkashaCandidate {
  user_message: string;
  assistant_preview: string;
  source_ref: string;
}

interface AkashaQueryRow {
  query_id: string;
  session_key: string;
  seq: number;
  query_text: string;
  intent: string;
  ts: string;
  seed_count: number;
  pool_count: number;
  activated_count: number;
  activation_threshold: number;
  dense_count: number;
  ripple_count: number;
  inject_chars: number;
  source_ref_count: number;
}

interface AkashaQueryDetail extends AkashaQueryRow {
  activation_items: AkashaCandidate[];
  dense_items: AkashaCard[];
  ripple_items: AkashaCard[];
  text_block_preview: string;
}

interface AkashaOverview {
  available: boolean;
  total: number;
  latest_at: string | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Prefix all helpers with ai_ to avoid collisions with other panels.
function ai_fmt2(v: number | null | undefined): string {
  if (v == null) return "-";
  return v.toFixed(2);
}

function ai_fmtScore(v: number | null | undefined): string {
  if (v == null) return "-";
  const n = Number(v);
  const cls = n >= 0.5 ? "ai-score-hi" : n >= 0.25 ? "ai-score-mid" : "ai-score-lo";
  return `<span class="ai-score ${cls}">${n.toFixed(3)}</span>`;
}

function ai_sourceTag(source: string): string {
  const cls = {
    Dense: "ai-src-dense",
    "Dense(FB)": "ai-src-densefb",
    FTS: "ai-src-fts",
    BlackHole: "ai-src-blackhole",
    Bridge: "ai-src-bridge",
  }[source] ?? "ai-src-other";
  return `<span class="ai-tag ${cls}">${escapeHtml(source)}</span>`;
}

function ai_pathTag(pt: string): string {
  const cls = {
    direct: "ai-path-direct",
    "1hop": "ai-path-1hop",
    "2hop": "ai-path-2hop",
    bridge: "ai-path-bridge",
  }[pt] ?? "";
  return `<span class="ai-tag ${cls}">${escapeHtml(pt)}</span>`;
}

function ai_suppressedTag(s: string): string {
  if (!s) return "";
  return `<span class="ai-tag ai-suppressed">${escapeHtml(s)}</span>`;
}

function ai_shortKey(key: string): string {
  // session_key:seq → last 2 segments for readability
  const parts = key.split(":");
  if (parts.length >= 2) {
    const seq = parts[parts.length - 1];
    const sk = parts.slice(0, -1).join(":");
    const short = sk.length > 20 ? "…" + sk.slice(-18) : sk;
    return `${short}:${seq}`;
  }
  return key;
}

function ai_shortTs(value: unknown): string {
  if (!value) return "-";
  const d = new Date(String(value));
  if (isNaN(d.getTime())) return String(value);
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// ── Activation table ───────────────────────────────────────────────────────

function ai_renderActivationTable(items: AkashaCandidate[]): string {
  if (!items.length) {
    return '<div class="ai-empty">无激活节点</div>';
  }
  const rows = items.map((item) => `
    <tr class="${item.suppressed ? "ai-row-suppressed" : ""}">
      <td class="ai-cell-msg" title="${escapeHtml(item.user_message)}">${escapeHtml(item.user_message || "-")}</td>
      <td class="ai-cell-preview" title="${escapeHtml(item.assistant_preview)}">${escapeHtml(item.assistant_preview || "-")}</td>
      <td class="ai-cell-num">${ai_fmtScore(item.score)}</td>
      <td>${ai_sourceTag(item.source)}</td>
      <td>${ai_pathTag(item.path_type)}${ai_suppressedTag(item.suppressed)}</td>
      <td class="ai-cell-num mono">${item.fan}</td>
      <td class="ai-cell-num mono">${ai_fmt2(item.direct)}</td>
      <td class="ai-cell-num mono">${ai_fmt2(item.state)}</td>
      <td class="ai-cell-num mono">${ai_fmt2(item.edge)}</td>
      <td class="ai-cell-num mono">${ai_fmt2(item.long)}</td>
      <td class="ai-cell-num mono">${ai_fmt2(item.resource)}</td>
    </tr>
  `).join("");
  return `
    <div class="ai-table-wrap">
      <table class="ai-table ai-table-cards">
        <thead>
          <tr>
            <th class="ai-th-msg">user</th>
            <th class="ai-th-preview">assistant</th>
            <th>score</th><th>source</th><th>path</th>
            <th>fan</th><th>direct</th><th>state</th><th>edge</th><th>long</th><th>resource</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Dense / Ripple card table ─────────────────────────────────────────────

function ai_renderCardTable(items: AkashaCard[], showPath: boolean): string {
  if (!items.length) {
    return '<div class="ai-empty">无记录</div>';
  }
  const extraHeaders = showPath
    ? "<th>path</th><th>seed</th><th>bridge</th>"
    : "";
  const rows = items.map((item) => {
    const srcRefIds: string[] = (() => {
      try { return JSON.parse(item.source_ref) as string[]; }
      catch { return []; }
    })();
    const srcRefLink = srcRefIds.length
      ? `<span class="ai-source-ref" title="${escapeHtml(item.source_ref)}">${srcRefIds.length} msg</span>`
      : "-";
    const extraCells = showPath
      ? `<td>${ai_pathTag(item.path_type)}${ai_suppressedTag(item.suppressed)}</td>
         <td class="mono ai-cell-key" title="${escapeHtml(item.seed_key)}">${escapeHtml(ai_shortKey(item.seed_key))}</td>
         <td class="mono ai-cell-key" title="${escapeHtml(item.bridge_key)}">${escapeHtml(ai_shortKey(item.bridge_key))}</td>`
      : "";
    return `
      <tr>
        <td class="ai-cell-msg">${escapeHtml(item.user_message)}</td>
        <td class="ai-cell-preview">${escapeHtml(item.assistant_preview)}</td>
        <td class="ai-cell-num">${ai_fmtScore(item.score)}</td>
        <td>${ai_sourceTag(item.source)}</td>
        ${extraCells}
        <td>${srcRefLink}</td>
      </tr>
    `;
  }).join("");
  return `
    <div class="ai-table-wrap">
      <table class="ai-table ai-table-cards">
        <thead>
          <tr>
            <th class="ai-th-msg">user</th>
            <th class="ai-th-preview">assistant</th>
            <th>score</th>
            <th>source</th>
            ${extraHeaders}
            <th>ref</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

// ── Detail renderer ───────────────────────────────────────────────────────

function ai_renderEmpty(): string {
  return `
    <div class="detail-empty">
      <div class="detail-empty-title">Akasha Inspector</div>
      <div class="detail-empty-text">点开一轮检索记录，这里会显示激活图、Dense 精确命中、Ripple 联想命中和注入预览。</div>
    </div>
  `;
}

function ai_renderDetail(item: AkashaQueryDetail): string {
  const threshold = (item.activation_threshold ?? 0).toFixed(3);
  return `
    <div class="detail-wrap ai-inspector">
      <div class="detail-toolbar">
        <div>
          <div class="detail-title">Akasha 检索记录</div>
          <div class="detail-subtext mono">${escapeHtml(item.session_key)} · seq ${item.seq}</div>
        </div>
      </div>

      <!-- 1. User Query -->
      <div class="detail-block">
        <div class="detail-label">User Query</div>
        <div class="ai-query-text">${escapeHtml(item.query_text)}</div>
        <div class="ai-meta-row">
          <span class="ai-meta-kv"><span class="ai-meta-k">intent</span><span class="ai-meta-v">${escapeHtml(item.intent)}</span></span>
          <span class="ai-meta-kv"><span class="ai-meta-k">ts</span><span class="ai-meta-v mono">${escapeHtml(ai_shortTs(item.ts))}</span></span>
        </div>
      </div>

      <!-- 2. Activation 激活 -->
      <div class="detail-block">
        <div class="detail-label">Activation 激活</div>
        <div class="ai-stats-row">
          <div class="ai-stat"><span class="ai-stat-val">${item.seed_count}</span><span class="ai-stat-k">seeds</span></div>
          <div class="ai-stat-sep">→</div>
          <div class="ai-stat"><span class="ai-stat-val">${item.pool_count}</span><span class="ai-stat-k">pool</span></div>
          <div class="ai-stat-sep">→</div>
          <div class="ai-stat"><span class="ai-stat-val">${item.activated_count}</span><span class="ai-stat-k">activated</span></div>
          <div class="ai-stat-sep"> / </div>
          <div class="ai-stat"><span class="ai-stat-val ai-threshold">${threshold}</span><span class="ai-stat-k">threshold</span></div>
        </div>
        ${ai_renderActivationTable(item.activation_items || [])}
      </div>

      <!-- 3. Dense 左脑记忆 -->
      <div class="detail-block">
        <div class="detail-label">左脑记忆：精确回忆 <span class="ai-count-badge">${item.dense_count}</span></div>
        ${ai_renderCardTable(item.dense_items || [], false)}
      </div>

      <!-- 4. Ripple 右脑联想 -->
      <div class="detail-block">
        <div class="detail-label">右脑联想：潜意识第一反应 <span class="ai-count-badge">${item.ripple_count}</span></div>
        ${ai_renderCardTable(item.ripple_items || [], true)}
      </div>

      <!-- 5. Prompt 注入 -->
      <div class="detail-block">
        <div class="detail-label">Prompt 注入</div>
        <div class="ai-stats-row">
          <div class="ai-stat"><span class="ai-stat-val">${item.dense_count}</span><span class="ai-stat-k">dense</span></div>
          <div class="ai-stat-sep">+</div>
          <div class="ai-stat"><span class="ai-stat-val">${item.ripple_count}</span><span class="ai-stat-k">ripple</span></div>
          <div class="ai-stat-sep">·</div>
          <div class="ai-stat"><span class="ai-stat-val">${item.inject_chars}</span><span class="ai-stat-k">chars</span></div>
          <div class="ai-stat-sep">·</div>
          <div class="ai-stat"><span class="ai-stat-val">${item.source_ref_count}</span><span class="ai-stat-k">refs</span></div>
        </div>
        ${item.text_block_preview
          ? `<details class="ai-preview-block"><summary>注入文本预览</summary><pre class="ai-preview-pre">${escapeHtml(item.text_block_preview)}</pre></details>`
          : '<div class="ai-empty">本轮无注入（非 context intent）</div>'}
      </div>
    </div>
  `;
}

// ── Plugin registration ───────────────────────────────────────────────────

window.AkashicDashboard.registerPlugin({
  id: "akasha_inspector",
  label: "Akasha Inspector",
  viewLabel: "akasha inspector",
  pageSize: 25,
  rowKey: "query_id",

  countTitle(total: number): string {
    return `${total} 轮检索`;
  },

  columns: [
    { key: "session_key", label: "Session", width: 108, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    { key: "ts", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true,
      renderCell(value) { return escapeHtml(ai_shortTs(value)); } },
    { key: "query_text", label: "Query", flex: true, fmt: "text-preview", cellClass: "content-preview" },
    { key: "seed_count", label: "Seeds", width: 60, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "activated_count", label: "Active", width: 60, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "dense_count", label: "Dense", width: 60, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "ripple_count", label: "Ripple", width: 60, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "inject_chars", label: "Chars", width: 70, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const r = await api<AkashaOverview>("/api/dashboard/akasha-inspector/overview");
      return r.available ? (r.total ?? 0) : null;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize, filters }: FetchPageOpts): Promise<FetchPageResult> {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    if (filters?.["session_key"]) params.set("session_key", filters["session_key"]);
    if (filters?.["q"]) params.set("q", filters["q"]);
    const data = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/akasha-inspector/turns?${params.toString()}`
    );
    return { items: data.items || [], total: data.total || 0 };
  },

  async fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    const queryId = String(item["query_id"] ?? "");
    return api(`/api/dashboard/akasha-inspector/turns/${encodePath(queryId)}`);
  },

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void {
    if (!item) {
      container.innerHTML = ai_renderEmpty();
      return;
    }
    container.innerHTML = ai_renderDetail(item as unknown as AkashaQueryDetail);
  },
});
