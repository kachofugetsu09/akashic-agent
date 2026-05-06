/// <reference path="../../types/akashic-dashboard.d.ts" />

interface RollupCandidate {
  id: string;
  tag: "identity" | "preference";
  title: string;
  content: string;
  score: number;
  reinforcement: number;
  emotional_weight: number;
  updated_at: string;
  evidence_count: number;
  source_ids: string[];
  source_summaries: string[];
  existing_overlap: boolean;
}

interface RollupOverview {
  active_counts: Record<string, number>;
  candidate_count: number;
  pending_preview: string;
}

function _shortContent(value: unknown): string {
  const text = String(value ?? "");
  return text.length > 80 ? `${text.slice(0, 80)}...` : text;
}

function _renderMetrics(candidate: RollupCandidate): string {
  const items = [
    ["reinforcement", candidate.reinforcement],
    ["emotional_weight", candidate.emotional_weight],
    ["updated_at", candidate.updated_at || "-"],
  ];
  return `
    <div class="rollup-score-grid">
      ${items.map(([key, value]) => `
        <div class="rollup-score-item">
          <span>${escapeHtml(String(key))}</span>
          <strong>${escapeHtml(String(value))}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function _renderSourceList(candidate: RollupCandidate): string {
  return `
    <div class="rollup-source-list">
      ${candidate.source_summaries.map((summary, index) => `
        <div class="rollup-source-item">
          <code>${escapeHtml(candidate.source_ids[index] ?? "")}</code>
          <div>${escapeHtml(summary)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function _requestDashboardRefresh(): void {
  window.dispatchEvent(new CustomEvent("akashic-dashboard-refresh"));
}

async function _generateCandidates(container: HTMLElement): Promise<void> {
  const button = container.querySelector<HTMLButtonElement>("[data-rollup-generate]");
  if (button) button.disabled = true;
  try {
    await api("/api/dashboard/memory-rollup/generate", {
      method: "POST",
      body: JSON.stringify({ limit: 180 }),
    });
    _requestDashboardRefresh();
    const result = container.querySelector<HTMLElement>("[data-rollup-result]");
    if (result) result.textContent = "候选已刷新。";
  } catch (exc) {
    _showRollupError(container, exc);
  } finally {
    if (button) button.disabled = false;
  }
}

async function _commitCandidate(candidate: RollupCandidate, container: HTMLElement): Promise<void> {
  const textarea = container.querySelector<HTMLTextAreaElement>("[data-rollup-content]");
  const tagSelect = container.querySelector<HTMLSelectElement>("[data-rollup-tag]");
  const content = textarea?.value.trim() || candidate.content;
  const tag = tagSelect?.value || candidate.tag;
  try {
    await api("/api/dashboard/memory-rollup/commit", {
      method: "POST",
      body: JSON.stringify({
        items: [{ id: candidate.id, tag, content }],
      }),
    });
    _requestDashboardRefresh();
    const result = container.querySelector<HTMLElement>("[data-rollup-result]");
    if (result) result.textContent = "已写入 PENDING.md。";
  } catch (exc) {
    _showRollupError(container, exc);
  }
}

async function _ignoreCandidate(candidate: RollupCandidate, container: HTMLElement): Promise<void> {
  try {
    await api("/api/dashboard/memory-rollup/ignore", {
      method: "POST",
      body: JSON.stringify({ id: candidate.id }),
    });
    _requestDashboardRefresh();
    const result = container.querySelector<HTMLElement>("[data-rollup-result]");
    if (result) result.textContent = "已忽略，后续不会再生成这个候选。";
  } catch (exc) {
    _showRollupError(container, exc);
  }
}

async function _deleteCandidateSources(candidate: RollupCandidate, container: HTMLElement): Promise<void> {
  const confirmed = window.confirm("删除这个候选的所有来源 memory？这个操作不会写入 PENDING。");
  if (!confirmed) return;
  try {
    await api("/api/dashboard/memory-rollup/delete-sources", {
      method: "POST",
      body: JSON.stringify({ id: candidate.id }),
    });
    _requestDashboardRefresh();
    const result = container.querySelector<HTMLElement>("[data-rollup-result]");
    if (result) result.textContent = "来源 memory 已删除。";
  } catch (exc) {
    _showRollupError(container, exc);
  }
}

function _showRollupError(container: HTMLElement, exc: unknown): void {
  const target = container.querySelector<HTMLElement>("[data-rollup-result]");
  if (!target) return;
  target.textContent = exc instanceof Error ? exc.message : String(exc);
}

window.AkashicDashboard.registerPlugin({
  id: "memory_rollup",
  label: "Memory Rollup",
  viewLabel: "memory rollup",
  pageSize: 25,
  rowKey: "id",

  countTitle(total: number): string {
    return `${total} 条候选`;
  },

  columns: [
    { key: "score", label: "Hotness", width: 80, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "reinforcement", label: "Reinforce", width: 88, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "emotional_weight", label: "Weight", width: 72, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "tag", label: "Tag", width: 88, cellClass: "mono cell-type" },
    { key: "title", label: "Title", width: 112, cellClass: "content-preview" },
    { key: "content", label: "Candidate", flex: true, fmt: "rollup-content", cellClass: "content-preview" },
    { key: "evidence_count", label: "Src", width: 56, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const overview = await api<RollupOverview>("/api/dashboard/memory-rollup/overview");
      return overview.candidate_count || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const data = await api<{ items: RollupCandidate[]; total: number }>("/api/dashboard/memory-rollup/candidates");
    const start = (page - 1) * pageSize;
    return {
      items: data.items.slice(start, start + pageSize) as unknown as Record<string, unknown>[],
      total: data.total || 0,
    };
  },

  fetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
    return Promise.resolve(item);
  },

  renderDetail(item: Record<string, unknown> | null, container: HTMLElement): void {
    if (!item) {
      container.innerHTML = `
        <div class="detail-wrap">
          <div class="detail-toolbar">
            <div>
              <div class="detail-title">Memory Rollup</div>
              <div class="detail-subtext">从 active profile / preference 生成可人工确认的 PENDING 候选。</div>
            </div>
          </div>
          <div class="rollup-actions">
            <button class="primary" type="button" data-rollup-generate>生成候选</button>
          </div>
          <div class="detail-block">
            <div class="detail-label">流程</div>
            <pre class="rollup-ascii">memory2.db
  └─ profile / preference
      └─ 真实 summary 候选
          └─ 前端人工确认
              └─ PENDING.md
                  └─ MemoryOptimizer 合并 MEMORY.md</pre>
          </div>
          <div class="detail-block">
            <div class="detail-label">状态</div>
            <div class="muted-text" data-rollup-result>选择左侧候选后可编辑并写入。</div>
          </div>
        </div>
      `;
      container.querySelector("[data-rollup-generate]")?.addEventListener("click", () => void _generateCandidates(container));
      return;
    }

    const candidate = item as unknown as RollupCandidate;
    container.innerHTML = `
      <div class="detail-wrap">
        <div class="detail-toolbar">
          <div>
            <div class="detail-title">${escapeHtml(candidate.title)}</div>
            <div class="detail-subtext">${escapeHtml(candidate.id)} · hotness ${escapeHtml(String(candidate.score))}</div>
          </div>
        </div>
        <div class="detail-grid">
          <div class="detail-row">
            <div class="detail-row-label">tag</div>
            <div class="detail-row-val">
              <select data-rollup-tag>
                <option value="preference">preference</option>
                <option value="identity">identity</option>
              </select>
            </div>
          </div>
          <div class="detail-row">
            <div class="detail-row-label">overlap</div>
            <div class="detail-row-val">${candidate.existing_overlap ? "已部分存在" : "未发现"}</div>
          </div>
          <div class="detail-row">
            <div class="detail-row-label">sources</div>
            <div class="detail-row-val">${escapeHtml(String(candidate.evidence_count))}</div>
          </div>
        </div>
        <div class="detail-block">
          <div class="detail-label">候选内容</div>
          <textarea class="rollup-editor" data-rollup-content></textarea>
        </div>
        <div class="detail-block">
          <div class="detail-label">表内指标</div>
          ${_renderMetrics(candidate)}
        </div>
        <div class="detail-block">
          <div class="detail-label">来源记忆</div>
          ${_renderSourceList(candidate)}
        </div>
        <div class="rollup-actions">
          <button class="primary" type="button" data-rollup-commit>写入 PENDING</button>
          <button class="ghost" type="button" data-rollup-ignore>忽略</button>
          <button class="danger-ghost" type="button" data-rollup-delete>删除来源</button>
          <span class="muted-text" data-rollup-result></span>
        </div>
      </div>
    `;
    const tagSelect = container.querySelector<HTMLSelectElement>("[data-rollup-tag]");
    if (tagSelect) tagSelect.value = candidate.tag;
    const editor = container.querySelector<HTMLTextAreaElement>("[data-rollup-content]");
    if (editor) editor.value = candidate.content;
    container.querySelector("[data-rollup-commit]")?.addEventListener("click", () => void _commitCandidate(candidate, container));
    container.querySelector("[data-rollup-ignore]")?.addEventListener("click", () => void _ignoreCandidate(candidate, container));
    container.querySelector("[data-rollup-delete]")?.addEventListener("click", () => void _deleteCandidateSources(candidate, container));
  },

  formatters: {
    "rollup-content": (value: unknown) => _shortContent(value),
  },
});
