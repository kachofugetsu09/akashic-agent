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

function ai_renderFilters(container: HTMLElement, dispatch: PluginDispatch): void {
  const q = dispatch.filters["q"] ?? "";
  const existing = container.querySelector<HTMLInputElement>("[data-ai-search]");
  if (existing) {
    if (document.activeElement !== existing && existing.value !== q) {
      existing.value = q;
    }
    return;
  }
  container.innerHTML = `
    <div class="filter-row">
      <label class="search"><span>⌕</span><input type="text" placeholder="搜索 query / session" value="${escapeHtml(q)}" data-ai-search /></label>
      <button class="ghost" type="button" data-ai-clear ${q ? "" : "disabled"}>清空</button>
    </div>
  `;
  const input = container.querySelector<HTMLInputElement>("[data-ai-search]")!;
  const clear = container.querySelector<HTMLButtonElement>("[data-ai-clear]")!;
  let debounceTimer = 0;
  input.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      const value = input.value.trim();
      if (value) dispatch.setFilter("q", value);
      else dispatch.clearFilter("q");
    }, 250);
  });
  clear.addEventListener("click", () => {
    input.value = "";
    dispatch.clearFilter("q");
  });
}

let _expanderId = 0;

function ai_textExpander(text: string | null | undefined, isAssistant: boolean = false): string {
  if (!text) return "-";
  const size = isAssistant ? "11.5px" : "13px";
  const lineH = isAssistant ? "1.4" : "1.5";
  if (text.length < 60 && !text.includes('\n')) {
    return `<div style="font-size:${size}; color:var(--color-${isAssistant ? 'muted' : 'fg'}); padding:4px 0; line-height:${lineH};">${escapeHtml(text)}</div>`;
  }
  const id = `ai-exp-${++_expanderId}`;
  return `
    <div class="ai-expander ${isAssistant ? 'ai-exp-asst' : ''}">
      <input type="checkbox" id="${id}" class="ai-exp-cb" style="display:none;" />
      <label for="${id}" class="ai-exp-header">
        <span class="ai-exp-icon">▶</span>
        <span class="ai-exp-text-closed" style="font-size:${size}">${escapeHtml(text.replace(/\n/g, ' '))}</span>
        <span class="ai-exp-text-open">Collapse</span>
      </label>
      <div class="ai-exp-full">
        <div class="ai-exp-full-inner" style="font-size:${size}">${escapeHtml(text)}</div>
      </div>
    </div>
  `;
}

function ai_assistantPreview(text: string | null | undefined): string {
  if (!text) return "";
  const prefixedText = `ASST: ${text}`;
  return `<div class="ai-asst-expander-wrap">${ai_textExpander(prefixedText, true)}</div>`;
}

function ai_sparkbar(label: string, v: number | null | undefined): string {
  if (v == null) return "";
  const percent = Math.min(100, Math.max(0, v * 100));
  return `
    <div class="ai-act-metric">
      <span class="ai-act-m-k">${label}</span>
      <div class="ai-spark-track"><div class="ai-spark-fill" style="width: ${percent}%;"></div></div>
    </div>
  `;
}

function ai_renderActivationTable(items: AkashaCandidate[]): string {
  if (!items.length) {
    return '<div class="ai-empty">无激活节点</div>';
  }
  const rows = items.map((item) => `
    <div class="ai-act-card ${item.suppressed ? "ai-row-suppressed" : ""}">
      <div class="ai-act-head">
        ${ai_fmtScore(item.score)}
        ${ai_sourceTag(item.source)}
        ${ai_pathTag(item.path_type)}${ai_suppressedTag(item.suppressed)}
      </div>
      <div class="ai-act-body">
        ${ai_textExpander(item.user_message)}
        ${ai_assistantPreview(item.assistant_preview)}
      </div>
      <div class="ai-act-metrics">
        <div class="ai-act-metric"><span class="ai-act-m-k">FAN</span><span class="ai-act-m-v mono">${item.fan}</span></div>
        ${ai_sparkbar('DIR', item.direct)}
        ${ai_sparkbar('STA', item.state)}
        ${ai_sparkbar('EDG', item.edge)}
        ${ai_sparkbar('LNG', item.long)}
        ${ai_sparkbar('RES', item.resource)}
      </div>
    </div>
  `).join("");
  return `<div class="ai-act-list">${rows}</div>`;
}

// ── Dense / Ripple card table ─────────────────────────────────────────────

function ai_renderCardTable(items: AkashaCard[], type: 'dense' | 'ripple'): string {
  if (!items.length) {
    return '<div class="ai-empty">无记录</div>';
  }
  const showPath = type === 'ripple';

  const cards = items.map((item) => {
    const srcRefIds: string[] = (() => {
      try { return JSON.parse(item.source_ref) as string[]; }
      catch { return []; }
    })();
    const refCount = srcRefIds.length;
    return `
      <div class="ai-mem-card">
        <div class="ai-mem-card-head">
          ${ai_fmtScore(item.score)}
          ${ai_sourceTag(item.source)}
          ${showPath ? ai_pathTag(item.path_type) + ai_suppressedTag(item.suppressed) : ""}
          <div style="flex:1"></div>
          ${refCount > 0 ? `<span class="ai-source-ref" title="${escapeHtml(item.source_ref)}">${refCount} refs</span>` : ""}
        </div>
        <div class="ai-mem-card-body">
          ${ai_textExpander(item.user_message)}
          ${ai_assistantPreview(item.assistant_preview)}
        </div>
        ${showPath ? `<div class="ai-mem-card-foot"><div class="ai-mem-foot-item"><span>SEED</span> <span class="mono">${escapeHtml(ai_shortKey(item.seed_key))}</span></div><div class="ai-mem-foot-item"><span>BRIDGE</span> <span class="mono">${escapeHtml(ai_shortKey(item.bridge_key))}</span></div></div>` : ""}
      </div>
    `;
  }).join("");
  return `<div class="ai-cards-list">${cards}</div>`;
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

function ai_renderDetail(item: AkashaQueryDetail, dispatch?: import("../../frontend/dashboard/src/types").PluginDispatch): string {
  const threshold = (item.activation_threshold ?? 0).toFixed(3);
  return `
    <div class="ai-inspector">
      <div class="ai-query-block">
        <div class="detail-title" style="display:flex; justify-content:space-between; align-items:flex-start;">
          <span>Akasha 检索记录 <span class="detail-subtext mono">(${escapeHtml(item.session_key)} · seq ${item.seq})</span></span>
          ${dispatch?.closePane ? `<button class="ai-close-btn" type="button" title="关闭详情面板">✕</button>` : ""}
        </div>
        <div class="ai-query-text">${escapeHtml(item.query_text)}</div>
        <div class="ai-meta-row">
          <span class="ai-meta-kv"><span class="ai-meta-k">Intent</span><span class="ai-meta-v">${escapeHtml(item.intent)}</span></span>
          <span class="ai-meta-kv"><span class="ai-meta-k">Time</span><span class="ai-meta-v">${escapeHtml(ai_shortTs(item.ts))}</span></span>
        </div>
      </div>

      <!-- KPI Grid -->
      <div class="ai-kpi-grid">
        <div class="ai-kpi-card">
          <div class="ai-kpi-val">${item.seed_count}</div>
          <div class="ai-kpi-label">SEEDS IGNITED</div>
        </div>
        <div class="ai-kpi-card">
          <div class="ai-kpi-val">${item.pool_count}</div>
          <div class="ai-kpi-label">POOL SIZE</div>
        </div>
        <div class="ai-kpi-card">
          <div class="ai-kpi-val">${item.activated_count}</div>
          <div class="ai-kpi-label">ACTIVATED</div>
        </div>
        <div class="ai-kpi-card">
          <div class="ai-kpi-val">${threshold}</div>
          <div class="ai-kpi-label">THRESHOLD</div>
        </div>
      </div>

      <!-- Activation -->
      <div class="ai-section-container ai-container-act">
        <div class="ai-section-header">
          <div class="ai-sh-text">ACTIVATION MATRIX <span class="ai-sh-sub">Neural Graph Propagation</span></div>
          <div class="ai-sh-count">${item.activation_items?.length || 0}</div>
        </div>
        ${ai_renderActivationTable(item.activation_items || [])}
      </div>

      <!-- Left Brain -->
      <div class="ai-section-container ai-container-dense">
        <div class="ai-section-header">
          <div class="ai-sh-text">LEFT BRAIN <span class="ai-sh-sub">Precise Retrieval (Dense)</span></div>
          <div class="ai-sh-count">${item.dense_count}</div>
        </div>
        ${ai_renderCardTable(item.dense_items || [], 'dense')}
      </div>

      <!-- Right Brain -->
      <div class="ai-section-container ai-container-ripple">
        <div class="ai-section-header">
          <div class="ai-sh-text">RIGHT BRAIN <span class="ai-sh-sub">Associative Leaps (Ripple)</span></div>
          <div class="ai-sh-count">${item.ripple_count}</div>
        </div>
        ${ai_renderCardTable(item.ripple_items || [], 'ripple')}
      </div>

      <!-- Prompt Synthesis -->
      <div class="detail-block">
        <div class="detail-title" style="margin-bottom:12px; font-size:14px; text-transform:uppercase; color:var(--color-subtle);">Synthesis (Prompt Injection)</div>
        <div class="ai-meta-row" style="margin-bottom:12px;">
          <span class="ai-meta-kv"><span class="ai-meta-k">Dense</span><span class="ai-meta-v">${item.dense_count}</span></span>
          <span class="ai-meta-kv"><span class="ai-meta-k">Ripple</span><span class="ai-meta-v">${item.ripple_count}</span></span>
          <span class="ai-meta-kv"><span class="ai-meta-k">Total Chars</span><span class="ai-meta-v">${item.inject_chars}</span></span>
          <span class="ai-meta-kv"><span class="ai-meta-k">Refs</span><span class="ai-meta-v">${item.source_ref_count}</span></span>
        </div>
        ${item.text_block_preview
          ? `<details class="ai-preview-block"><summary>View Injected Context</summary><pre class="ai-preview-pre">${escapeHtml(item.text_block_preview)}</pre></details>`
          : '<div class="ai-empty">No context injected in this turn.</div>'}
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
  ],

  renderFilters: ai_renderFilters,

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

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement, dispatch?: import("../../frontend/dashboard/src/types").PluginDispatch): void {
    if (!item) {
      container.innerHTML = ai_renderEmpty();
      return;
    }
    container.innerHTML = ai_renderDetail(item as unknown as AkashaQueryDetail, dispatch);
    
    if (dispatch?.closePane) {
      const closeBtn = container.querySelector(".ai-close-btn");
      if (closeBtn) {
        closeBtn.addEventListener("click", () => dispatch.closePane!());
      }
    }
  },
});
