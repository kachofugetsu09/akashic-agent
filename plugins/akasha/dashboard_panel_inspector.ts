/// <reference path="../../types/akashic-dashboard.d.ts" />

interface InspectorItem {
  query_id: string;
  session_key: string;
  ts: string;
  user_text: string;
  assistant_preview: string;
  score?: number;
  value?: number;
  seed_score?: number;
  completion_mass?: number;
  graph_only?: boolean;
  first_relation?: string | null;
  sources?: string[];
  relation_path?: string[];
}

interface InspectorRow {
  query_id: string;
  session_key: string;
  seq: number;
  ts: string;
  query_text: string;
  seed_count: number;
  activation_capture_available: boolean;
  recall_capture_available: boolean;
  activation_count: number;
  completion_count: number;
  pushes: number;
  residual_l1: number;
}

interface InspectorDetail extends InspectorRow {
  assistant_text: string;
  graph_only_count: number;
  basin_count: number;
  surprise?: number | null;
  observed_mass?: number | null;
  recurrent_mass?: number | null;
  reactivated_mass?: number | null;
  potentiated_mass?: number | null;
  inhibited_mass?: number | null;
  seeds: InspectorItem[];
  activation_items: InspectorItem[];
  left: InspectorItem[];
  right: InspectorItem[];
  tool_left: InspectorItem[];
  tool_right: InspectorItem[];
  left_count: number;
  right_count: number;
  tool_left_count: number;
  tool_right_count: number;
  inject_chars: number;
  text_block_preview: string;
}

interface InspectorOverview {
  available: boolean;
  total: number;
}

function shortTime(value: unknown): string {
  if (!value) return "—";
  const parsed = new Date(String(value));
  if (Number.isNaN(parsed.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

function fixed(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function sourceText(item: InspectorItem): string {
  const sources = item.sources ?? [];
  if (sources.length) return sources.join(" · ");
  if (item.first_relation) return item.first_relation;
  return item.graph_only ? "仅由关系补全" : "外部线索";
}

function renderItems(items: InspectorItem[], empty: string): string {
  if (!items.length) {
    return `<p class="akasha-empty">${escapeHtml(empty)}</p>`;
  }
  return `
    <ol class="akasha-evidence-list">
      ${items.map((item, index) => {
        const score = item.score ?? item.value ?? item.completion_mass ?? item.seed_score;
        const path = item.relation_path?.length
          ? `<span class="akasha-path" title="${escapeHtml(item.relation_path.join(" → "))}">${escapeHtml(item.relation_path.join(" → "))}</span>`
          : "";
        return `
          <li class="akasha-evidence">
            <span class="akasha-evidence-rank" aria-hidden="true">${index + 1}</span>
            <div class="akasha-evidence-main">
              <p>${escapeHtml(item.user_text || "（空消息）")}</p>
              ${item.assistant_preview
                ? `<p class="akasha-assistant">${escapeHtml(item.assistant_preview)}</p>`
                : ""}
            </div>
            <div class="akasha-evidence-meta">
              <time class="akasha-chip akasha-chip--time">${escapeHtml(shortTime(item.ts))}</time>
              <span class="akasha-chip">${escapeHtml(sourceText(item))}</span>
              ${score == null ? "" : `<b class="akasha-chip akasha-chip--score">${fixed(score)}</b>`}
            </div>
            ${path}
          </li>
        `;
      }).join("")}
    </ol>
  `;
}

function evidenceLane(
  title: string,
  description: string,
  lane: string,
  items: InspectorItem[],
  count: number | string,
  empty: string,
  open = false,
): string {
  return `
    <details class="akasha-section akasha-lane akasha-lane--${escapeHtml(lane)}" ${open ? "open" : ""}>
      <summary>
        <span class="akasha-lane-copy">
          <strong>${escapeHtml(title)}</strong>
          <small>${escapeHtml(description)}</small>
        </span>
        <span class="akasha-lane-count">${escapeHtml(String(count))}</span>
      </summary>
      ${renderItems(items, empty)}
    </details>
  `;
}

function metric(label: string, value: unknown, detail: string): string {
  return `
    <div class="akasha-metric">
      <dt>${escapeHtml(label)}</dt>
      <dd>${escapeHtml(String(value))}</dd>
      <p>${escapeHtml(detail)}</p>
    </div>
  `;
}

function renderFilters(container: HTMLElement, dispatch: PluginDispatch): void {
  const value = dispatch.filters["q"] ?? "";
  const existing = container.querySelector<HTMLInputElement>("[data-akasha-search]");
  if (existing) {
    if (document.activeElement !== existing && existing.value !== value) {
      existing.value = value;
    }
    return;
  }
  container.innerHTML = `
    <div class="akasha-filter">
      <label>
        <span>搜索检索记录</span>
        <input
          type="search"
          value="${escapeHtml(value)}"
          placeholder="Query、回复或 Session"
          data-akasha-search
        />
      </label>
      <md-text-button data-akasha-clear ${value ? "" : "disabled"}>清空</md-text-button>
    </div>
  `;
  const input = container.querySelector<HTMLInputElement>("[data-akasha-search]")!;
  const clear = container.querySelector<HTMLElement>("[data-akasha-clear]")!;
  let timer = 0;
  input.addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      const query = input.value.trim();
      if (query) dispatch.setFilter("q", query);
      else dispatch.clearFilter("q");
    }, 200);
  });
  clear.addEventListener("click", () => {
    input.value = "";
    dispatch.clearFilter("q");
  });
}

function renderDetail(item: InspectorDetail, closePane?: () => void): string {
  const recallCount = item.recall_capture_available
    ? item.left_count + item.right_count
    : item.left_count;
  return `
    <article class="akasha-inspector">
      <header class="akasha-query">
        <div>
          <h2>${escapeHtml(item.query_text)}</h2>
          <p class="akasha-query-meta">${escapeHtml(shortTime(item.ts))} · seq ${item.seq}<span>${escapeHtml(item.session_key)}</span></p>
        </div>
        ${closePane ? '<md-icon-button class="akasha-close" data-akasha-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}
      </header>

      <section class="akasha-overview" aria-labelledby="akasha-overview-title">
        <div class="akasha-overview-heading">
          <div>
            <h3 id="akasha-overview-title">${recallCount} 条记忆参与回答</h3>
          </div>
          <p>${item.inject_chars > 0 ? `已写入 ${item.inject_chars} 字上下文` : "没有写入 Prompt"}</p>
        </div>
        <dl class="akasha-metrics">
          ${metric("直接线索", item.seed_count, "Dense、BM25 与时序")}
          ${metric("精确回忆", item.left_count, "语义最接近的历史")}
          ${metric("模式联想", item.recall_capture_available ? item.right_count : "—", item.recall_capture_available ? `${item.basin_count} 个情景簇` : "本轮未记录")}
        </dl>
      </section>

      <details class="akasha-answer">
        <summary>
          <span><strong>助手回复</strong><small>查看这一轮的完整回答</small></span>
          <span class="akasha-answer-action">展开</span>
        </summary>
        <div class="akasha-answer-body">${escapeHtml(item.assistant_text || "（助手没有文本回复）")}</div>
      </details>

      <section class="akasha-evidence-group" aria-labelledby="akasha-evidence-title">
        <div class="akasha-section-heading">
          <h3 id="akasha-evidence-title">记忆证据</h3>
          <small>选择一组展开查看</small>
        </div>
        <div class="akasha-lanes">
        ${evidenceLane("直接线索", "最初命中的消息", "seed", item.seeds, item.seeds.length, "这一轮没有形成可持久化线索。")}
        ${item.activation_capture_available
          ? evidenceLane("图扩散候选", "由关系网络补入的候选", "activation", item.activation_items, item.activation_items.length, "图扩散没有增加候选。")
          : ""}
        ${evidenceLane("精确回忆", "语义最接近的历史消息", "precise", item.left, item.left_count, "没有精确命中。")}
        ${evidenceLane("模式联想", "跨关系补全且已与精确结果去重", "completion", item.right, item.recall_capture_available ? item.right_count : "未记录", "没有产生模式联想。")}
        ${item.tool_left_count
          ? evidenceLane("工具精确回忆", "recall_memory 的语义命中", "precise", item.tool_left, item.tool_left_count, "工具没有产生精确命中。")
          : ""}
        ${item.tool_right_count
          ? evidenceLane("工具模式联想", "recall_memory 的图关系结果", "completion", item.tool_right, item.tool_right_count, "工具没有产生模式联想。")
          : ""}
        </div>
      </section>

      <details class="akasha-learning">
        <summary><span><strong>学习变化与技术指标</strong><small>${item.activation_count} 条扩散候选 · ${item.pushes} 次扩散</small></span></summary>
        <dl>
          ${metric("惊喜度", fixed(item.surprise), "当前 cue 与已有模式的差异")}
          ${metric("观察质量", fixed(item.observed_mass), "由外部证据支持的学习质量")}
          ${metric("再激活", fixed(item.reactivated_mass), "已有关系重新获得的活性")}
          ${metric("增强 / 抑制", `${fixed(item.potentiated_mass)} / ${fixed(item.inhibited_mass)}`, "连接预算内的竞争结果")}
        </dl>
      </details>

      <details class="akasha-prompt">
        <summary><span><strong>写入 Prompt 的记忆</strong><small>${item.inject_chars} 字 · 原始上下文预览</small></span></summary>
        <pre>${escapeHtml(item.text_block_preview || "这一轮没有注入记忆。")}</pre>
      </details>
    </article>
  `;
}

window.AkashicDashboard.registerPlugin({
  id: "akasha_inspector",
  label: "Akasha 检索",
  viewLabel: "Akasha 检索",
  pageSize: 25,
  rowKey: "query_id",

  countTitle(total: number): string {
    return `${total} 轮检索`;
  },

  columns: [
    { key: "session_key", label: "会话", width: 120, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    {
      key: "ts",
      label: "时间",
      width: 110,
      cellClass: "mono cell-time",
      rawTitle: true,
      renderCell(value) { return escapeHtml(shortTime(value)); },
    },
    { key: "query_text", label: "用户问题", flex: true, fmt: "text-preview", cellClass: "content-preview" },
    { key: "seed_count", label: "线索", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "completion_count", label: "召回", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
  ],

  renderFilters,

  async getCount(): Promise<number | null> {
    try {
      const result = await api<InspectorOverview>("/api/dashboard/akasha-inspector/overview");
      return result.available ? result.total : null;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize, filters }: FetchPageOpts): Promise<FetchPageResult> {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    });
    if (filters?.["session_key"]) params.set("session_key", filters["session_key"]);
    if (filters?.["q"]) params.set("q", filters["q"]);
    const result = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/akasha-inspector/turns?${params.toString()}`,
    );
    return { items: result.items, total: result.total };
  },

  async fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    return api(
      `/api/dashboard/akasha-inspector/turns/${encodePath(String(item["query_id"] ?? ""))}`,
    );
  },

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement, dispatch?: PluginDispatch): void {
    if (!item) {
      container.innerHTML = `
        <div class="detail-empty">
          <div class="detail-empty-title">Akasha Inspector</div>
          <div class="detail-empty-text">选择一轮检索，查看它从哪些线索开始、扩散到哪里，以及最终进入 Prompt 的内容。</div>
        </div>
      `;
      return;
    }
    container.innerHTML = renderDetail(
      item as unknown as InspectorDetail,
      dispatch?.closePane,
    );
    container.querySelector("[data-akasha-close]")?.addEventListener(
      "click",
      () => dispatch?.closePane?.(),
    );
    const lanes = Array.from(container.querySelectorAll<HTMLDetailsElement>(".akasha-lane"));
    for (const lane of lanes) {
      lane.addEventListener("toggle", () => {
        if (!lane.open) return;
        for (const sibling of lanes) {
          if (sibling !== lane) sibling.open = false;
        }
      });
    }
  },
});
