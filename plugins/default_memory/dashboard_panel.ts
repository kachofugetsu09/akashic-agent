/// <reference path="../../types/akashic-dashboard.d.ts" />

// Local types for memory data shapes
interface MemoryRow {
  id: string;
  memory_type: string;
  summary: string;
  source_ref: string;
  happened_at: string;
  status: string;
  created_at: string;
  updated_at: string;
  reinforcement: number;
  emotional_weight: number;
  has_embedding: boolean;
  scope_channel: string;
  scope_chat_id: string;
}

interface MemoryDetail extends MemoryRow {
  content_hash?: string;
  extra_json: Record<string, unknown>;
  embedding_dim: number;
  embedding?: number[] | null;
}

interface MemoryOptimizerStatus {
  enabled: boolean;
  running: boolean;
  last_status: string;
  last_error: string | null;
}

// Local helpers (cannot import from format.ts)
// Use prefix dmem_ to avoid global name collisions with other plugins.
function dmemShortTs(value: unknown): string {
  if (!value) return "-";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return `${date.getMonth() + 1}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function dmemMetric(value: unknown): string {
  return String(value ?? 0);
}

function dmemTypeClass(t: string): string {
  return `memory-type-${t || "unknown"}`;
}

function dmemTypePill(memoryType: string): string {
  return `<span class="type-pill ${escapeHtml(dmemTypeClass(memoryType))}">${escapeHtml(memoryType)}</span>`;
}

function dmemStatusPill(status: string): string {
  return `<span class="status-pill memory-status-${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

// Memory type counts fetched once, cached per filter context
let dmemCachedCountsKey = "__initial__";
let dmemCachedCounts: { memory_type: string; total: number }[] = [];

function dmemCountsCacheKey(filters: Readonly<Record<string, string>>): string {
  const q = filters["q"] ?? "";
  const status = filters["status"] ?? "";
  const scope_channel = filters["scope_channel"] ?? "";
  const scope_chat_id = filters["scope_chat_id"] ?? "";
  return JSON.stringify({ q, status, scope_channel, scope_chat_id });
}

async function dmemFetchTypeCounts(filters: Readonly<Record<string, string>>): Promise<{ memory_type: string; total: number }[]> {
  const cacheKey = dmemCountsCacheKey(filters);
  const q = filters["q"] ?? "";
  const status = filters["status"] ?? "";
  const scope_channel = filters["scope_channel"] ?? "";
  const scope_chat_id = filters["scope_chat_id"] ?? "";
  if (cacheKey === dmemCachedCountsKey) return dmemCachedCounts;
  const memoryTypes = ["procedure", "preference", "event", "profile"];
  const result: { memory_type: string; total: number }[] = [];
  for (const t of memoryTypes) {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    params.set("memory_type", t);
    if (status) params.set("status", status);
    if (scope_channel) params.set("scope_channel", scope_channel);
    if (scope_chat_id) params.set("scope_chat_id", scope_chat_id);
    params.set("page", "1");
    params.set("page_size", "1");
    params.set("sort_by", "updated_at");
    params.set("sort_order", "desc");
    const payload = await api<{ items: unknown[]; total: number }>(`/api/dashboard/memories?${params.toString()}`);
    if ((payload.total || 0) > 0) result.push({ memory_type: t, total: payload.total });
  }
  dmemCachedCountsKey = cacheKey;
  dmemCachedCounts = result;
  return result;
}

function dmemRenderTypeList(counts: { memory_type: string; total: number }[], activeType: string): string {
  return counts.map((item) => `
    <button class="memory-quick-item ${activeType === item.memory_type ? "active" : ""}" type="button" data-mem-type="${escapeHtml(item.memory_type)}">
      <div class="nav-item-row">
        <span class="nav-type-dot ${escapeHtml(dmemTypeClass(item.memory_type))}"></span>
        <span class="nav-item-name">${escapeHtml(item.memory_type)}</span>
        <span class="nav-item-count">${item.total}</span>
      </div>
    </button>
  `).join("");
}

function dmemUpdateTotal(container: HTMLElement, counts: { memory_type: string; total: number }[]): void {
  const totalEl = container.querySelector("[data-mem-total]");
  if (totalEl) totalEl.textContent = String(counts.reduce((sum, item) => sum + item.total, 0));
}

function dmemBindTypeListClicks(container: HTMLElement, dispatch: PluginDispatch): void {
  const allBtn = container.querySelector<HTMLButtonElement>("[data-mem-all]");
  if (allBtn) {
    allBtn.onclick = () => {
      dispatch.activate();
      dispatch.clearFilter("memory_type");
    };
  }
  container.querySelectorAll<HTMLButtonElement>("[data-mem-type]").forEach((btn) => {
    btn.onclick = () => {
      const t = btn.getAttribute("data-mem-type") ?? "";
      dispatch.activate();
      dispatch.setFilter("memory_type", t);
    };
  });
}

// renderNavBody: "全部记忆" + type list
// First call: creates DOM, fetches counts async. Subsequent calls: delta-update only.
function dmemRenderNavBody(container: HTMLElement, dispatch: PluginDispatch): void {
  const filters = dispatch.filters;
  const activeType = filters["memory_type"] ?? "";

  const currentCountsKey = container.getAttribute("data-mem-counts-key") ?? "";
  const nextCountsKey = dmemCountsCacheKey(filters);

  // Already mounted: update active states (refetch counts if filter context changed)
  const existingAll = container.querySelector("[data-mem-all]");
  if (existingAll) {
    const allBtn = container.querySelector<HTMLButtonElement>("[data-mem-all]");
    if (allBtn) allBtn.className = `all-messages-row ${activeType === "" ? "active" : ""}`;
    container.querySelectorAll<HTMLButtonElement>("[data-mem-type]").forEach((btn) => {
      const t = btn.getAttribute("data-mem-type") ?? "";
      btn.className = `memory-quick-item ${activeType === t ? "active" : ""}`;
    });
    if (currentCountsKey !== nextCountsKey) {
      container.setAttribute("data-mem-counts-key", nextCountsKey);
      dmemFetchTypeCounts(filters).then((counts) => {
        dmemUpdateTotal(container, counts);
        const list = container.querySelector("[data-mem-count-list]");
        if (!list) return;
        list.innerHTML = dmemRenderTypeList(counts, activeType);
        dmemBindTypeListClicks(container, dispatch);
      }).catch(() => { /* ignore */ });
    } else {
      dmemUpdateTotal(container, dmemCachedCounts);
    }
    return;
  }

  // First call: build skeleton immediately, fetch counts async
  container.setAttribute("data-mem-counts-key", nextCountsKey);
  container.innerHTML = `
    <button class="all-messages-row ${activeType === "" ? "active" : ""}" type="button" data-mem-all>
      <span>全部记忆</span><strong data-mem-total>...</strong>
    </button>
    <div class="memory-quick-list" data-mem-count-list></div>
  `;
  dmemBindTypeListClicks(container, dispatch);

  dmemFetchTypeCounts(filters).then((counts) => {
    dmemUpdateTotal(container, counts);
    const list = container.querySelector("[data-mem-count-list]");
    if (list) {
      list.innerHTML = dmemRenderTypeList(counts, activeType);
      dmemBindTypeListClicks(container, dispatch);
    }
  }).catch(() => { /* ignore */ });
}

// renderFilters: search + type select + status select + scope chip
// First call: creates full DOM. Subsequent calls: delta update preserving focus.
function dmemRenderFilters(container: HTMLElement, dispatch: PluginDispatch): void {
  const filters = dispatch.filters;
  const q = filters["q"] ?? "";
  const memType = filters["memory_type"] ?? "";
  const status = filters["status"] ?? "";
  const scopeChannel = filters["scope_channel"] ?? "";
  const scopeChatId = filters["scope_chat_id"] ?? "";

  const existingSearch = container.querySelector<HTMLInputElement>("[data-mem-search]");
  if (existingSearch) {
    // Delta update: preserve focus by only updating select values and scope chip
    const typeSelect = container.querySelector<HTMLSelectElement>("[data-mem-type-select]");
    const statusSelect = container.querySelector<HTMLSelectElement>("[data-mem-status-select]");
    if (typeSelect && typeSelect.value !== memType) typeSelect.value = memType;
    if (statusSelect && statusSelect.value !== status) statusSelect.value = status;
    const scopeArea = container.querySelector<HTMLElement>("[data-mem-scope-area]");
    if (scopeArea) {
      if (scopeChannel || scopeChatId) {
        scopeArea.innerHTML = `<div class="active-session-chip"><span>scope</span><code>${escapeHtml(scopeChannel)}:${escapeHtml(scopeChatId)}</code><button type="button" data-mem-scope-clear>×</button></div>`;
        scopeArea.querySelector("[data-mem-scope-clear]")?.addEventListener("click", () => {
          dispatch.clearFilters(["scope_channel", "scope_chat_id"]);
        });
      } else {
        scopeArea.innerHTML = "";
      }
    }
    return;
  }

  // First call: create full DOM
  container.innerHTML = `
    <div class="filter-row">
      <label class="search"><span>⌕</span><input type="text" placeholder="搜索 memory / source_ref" value="${escapeHtml(q)}" data-mem-search /></label>
      <select data-mem-type-select>
        <option value="">全部 type</option>
        <option value="procedure">procedure</option>
        <option value="preference">preference</option>
        <option value="event">event</option>
        <option value="profile">profile</option>
      </select>
      <select data-mem-status-select>
        <option value="">全部 status</option>
        <option value="active">active</option>
        <option value="superseded">superseded</option>
      </select>
      <span data-mem-scope-area></span>
    </div>
  `;

  const searchInput = container.querySelector<HTMLInputElement>("[data-mem-search]")!;
  const typeSelect = container.querySelector<HTMLSelectElement>("[data-mem-type-select]")!;
  const statusSelect = container.querySelector<HTMLSelectElement>("[data-mem-status-select]")!;
  typeSelect.value = memType;
  statusSelect.value = status;

  let debounceTimer = 0;
  searchInput.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      const val = searchInput.value.trim();
      if (val) dispatch.setFilter("q", val);
      else dispatch.clearFilter("q");
    }, 300);
  });

  typeSelect.addEventListener("change", () => {
    if (typeSelect.value) dispatch.setFilter("memory_type", typeSelect.value);
    else dispatch.clearFilter("memory_type");
  });

  statusSelect.addEventListener("change", () => {
    if (statusSelect.value) dispatch.setFilter("status", statusSelect.value);
    else dispatch.clearFilter("status");
  });

  const scopeArea = container.querySelector<HTMLElement>("[data-mem-scope-area]")!;
  if (scopeChannel || scopeChatId) {
    scopeArea.innerHTML = `<div class="active-session-chip"><span>scope</span><code>${escapeHtml(scopeChannel)}:${escapeHtml(scopeChatId)}</code><button type="button" data-mem-scope-clear>×</button></div>`;
    scopeArea.querySelector("[data-mem-scope-clear]")?.addEventListener("click", () => {
      dispatch.clearFilters(["scope_channel", "scope_chat_id"]);
    });
  }
}

// renderTopbarAction: memory optimizer button
function dmemRenderTopbarAction(container: HTMLElement, dispatch: PluginDispatch): void {
  const currentToken = Number(container.getAttribute("data-mem-render-token") ?? "0") + 1;
  container.setAttribute("data-mem-render-token", String(currentToken));

  api<MemoryOptimizerStatus>("/api/dashboard/memory/optimizer").then((status) => {
    if (container.getAttribute("data-mem-render-token") !== String(currentToken)) return;
    if (!status.enabled) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = `<button class="ghost" type="button" data-mem-optimizer>记忆优化</button>`;
    const btn = container.querySelector<HTMLButtonElement>("[data-mem-optimizer]")!;
    let pollTimer = 0;

    const poll = (): void => {
      window.clearInterval(pollTimer);
      pollTimer = window.setInterval(() => {
        api<MemoryOptimizerStatus>("/api/dashboard/memory/optimizer").then((s) => {
          if (container.getAttribute("data-mem-render-token") !== String(currentToken)) {
            window.clearInterval(pollTimer);
            return;
          }
          if (!s.running) {
            window.clearInterval(pollTimer);
            btn.disabled = false;
            btn.textContent = s.last_status === "succeeded"
              ? "优化已完成"
              : s.last_status === "failed"
                ? "优化失败"
                : s.last_status === "skipped"
                  ? "已跳过"
                  : "记忆优化";
            dispatch.refresh();
          }
        }).catch(() => { window.clearInterval(pollTimer); });
      }, 2000);
    };

    btn.addEventListener("click", () => {
      btn.disabled = true;
      btn.textContent = "正在启动优化";
      api("/api/dashboard/memory/optimize", { method: "POST" }).then(() => {
        btn.textContent = "记忆优化中";
        poll();
      }).catch(() => {
        btn.disabled = false;
        btn.textContent = "记忆优化";
      });
    });

    if (status.running) {
      btn.disabled = true;
      btn.textContent = "记忆优化中";
      poll();
    }
  }).catch(() => { /* ignore, optimizer not available */ });
}

// renderDetail: full memory detail view
function dmemRenderDetail(
  item: Record<string, unknown> | null,
  container: HTMLElement,
  dispatch?: PluginDispatch,
): void {
  if (!item) {
    container.innerHTML = `
      <div class="detail-empty">
        <div class="detail-empty-title">详情</div>
        <div class="detail-empty-text">点开 memory 后，这里会显示完整字段、JSON 和相似记忆。</div>
      </div>
    `;
    return;
  }

  const mem = item as unknown as MemoryDetail;
  const similar = (item["_similar"] as MemoryRow[] | undefined) ?? [];
  const extraJson = mem.extra_json ?? {};
  const scopeChannel = String(extraJson["scope_channel"] ?? "");
  const scopeChatId = String(extraJson["scope_chat_id"] ?? "");
  const hasScopeBtn = Boolean(scopeChannel || scopeChatId);

  const typePillHtml = dmemTypePill(mem.memory_type);
  const statusPillHtml = dmemStatusPill(mem.status);
  const similarHtml = similar.length
    ? similar.map((s) => `<div class="detail-callout"><code>${escapeHtml(s.id)}</code><div>${escapeHtml(s.summary)}</div></div>`).join("")
    : `<div class="muted-text">没有相似记忆。</div>`;
  const scopeBtnHtml = hasScopeBtn ? `<button class="ghost" type="button" data-mem-scope-btn>查看同 scope 记忆</button>` : "";

  container.innerHTML = `
    <div class="detail-wrap">
      <div class="detail-toolbar">
        <div>
          <div class="detail-title">记忆详情</div>
          <div class="detail-subtext">${escapeHtml(mem.id)}</div>
        </div>
      </div>
      <div class="detail-block">
        <div class="detail-label">Summary</div>
        <div class="detail-content">${escapeHtml(mem.summary)}</div>
      </div>
      <div class="detail-grid">
        <div class="detail-row"><div class="detail-row-label">type</div><div class="detail-row-val">${typePillHtml}</div></div>
        <div class="detail-row"><div class="detail-row-label">status</div><div class="detail-row-val">${statusPillHtml}</div></div>
        <div class="detail-row"><div class="detail-row-label">source_ref</div><div class="detail-row-val"><code>${escapeHtml(mem.source_ref || "-")}</code></div></div>
        <div class="detail-row"><div class="detail-row-label">embedding</div><div class="detail-row-val"><code>${mem.has_embedding ? `${mem.embedding_dim} dims` : "none"}</code></div></div>
      </div>
      ${scopeBtnHtml}
      <div class="detail-block">
        <div class="detail-label">Extra JSON</div>
        ${jvPlaceholder(extraJson)}
      </div>
      <div class="detail-block">
        <div class="detail-label">Similar</div>
        <div class="detail-similar-list">${similarHtml}</div>
      </div>
    </div>
  `;
  attachJsonViewers(container);

  if (hasScopeBtn && dispatch) {
    container.querySelector("[data-mem-scope-btn]")?.addEventListener("click", () => {
      dispatch.activate();
      dispatch.setFilters({
        scope_channel: scopeChannel,
        scope_chat_id: scopeChatId,
      });
    });
  }
}

// fetchDetail: fetch detail + similar, merge into one record
async function dmemFetchDetail(item: Record<string, unknown>): Promise<Record<string, unknown>> {
  const id = String(item["id"] ?? "");
  const [detail, similar] = await Promise.all([
    api<MemoryDetail>(`/api/dashboard/memories/${encodePath(id)}`),
    api<{ items: MemoryRow[]; total: number }>(`/api/dashboard/memories/${encodePath(id)}/similar?top_k=6`).catch(() => ({ items: [], total: 0 })),
  ]);
  return { ...detail as unknown as Record<string, unknown>, _similar: similar.items ?? [] };
}

// fetchPage: translate filters/sortBy/sortOrder into API params
async function dmemFetchPage(opts: FetchPageOpts): Promise<FetchPageResult> {
  const filters = opts.filters ?? {};
  const params = new URLSearchParams();
  if (filters["q"]) params.set("q", filters["q"]);
  if (filters["memory_type"]) params.set("memory_type", filters["memory_type"]);
  if (filters["status"]) params.set("status", filters["status"]);
  if (filters["scope_channel"]) params.set("scope_channel", filters["scope_channel"]);
  if (filters["scope_chat_id"]) params.set("scope_chat_id", filters["scope_chat_id"]);
  params.set("page", String(opts.page));
  params.set("page_size", String(opts.pageSize));
  params.set("sort_by", opts.sortBy || "created_at");
  params.set("sort_order", opts.sortOrder || "desc");
  const payload = await api<{ items: Record<string, unknown>[]; total: number }>(`/api/dashboard/memories?${params.toString()}`);
  return { items: payload.items || [], total: payload.total || 0 };
}

async function dmemGetCount(): Promise<number | null> {
  try {
    const payload = await api<{ items: unknown[]; total: number }>("/api/dashboard/memories?page=1&page_size=1&sort_by=created_at&sort_order=desc");
    return payload.total || 0;
  } catch {
    return null;
  }
}

window.AkashicDashboard.registerPlugin({
  id: "default_memory",
  label: "Memory",
  viewLabel: "memory",
  pageSize: 25,
  rowKey: "id",
  defaultSortBy: "created_at",
  defaultSortOrder: "desc",

  countTitle(n: number): string {
    return `${n} 条记忆`;
  },

  columns: [
    { key: "memory_type", label: "Type", width: 96, cellClass: "cell-type", renderCell: (v) => dmemTypePill(String(v ?? "")) },
    { key: "summary", label: "Summary", flex: true, cellClass: "content-preview" },
    { key: "reinforcement", label: "Uses", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right", sortable: true },
    { key: "emotional_weight", label: "Weight", width: 72, fmt: "metric", cellClass: "mono cell-metric", align: "right", sortable: true },
    { key: "source_ref", label: "Source", width: 120, cellClass: "cell-source" },
    { key: "created_at", label: "Created", width: 96, fmt: "mono-time", cellClass: "mono cell-time", sortable: true },
    { key: "updated_at", label: "Updated", width: 96, fmt: "mono-time", cellClass: "mono cell-time", sortable: true },
    { key: "status", label: "Status", width: 88, cellClass: "cell-status", renderCell: (v) => dmemStatusPill(String(v ?? "")) },
  ],

  batchActions: [
    {
      label: "批量删除",
      className: "danger-ghost",
      async run(ids: string[]): Promise<void> {
        await api("/api/dashboard/memories/batch-delete", {
          method: "POST",
          body: JSON.stringify({ ids }),
        });
      },
    },
  ],

  getCount: dmemGetCount,
  fetchPage: dmemFetchPage,
  fetchDetail: dmemFetchDetail,

  renderNavBody: dmemRenderNavBody,
  renderFilters: dmemRenderFilters,
  renderTopbarAction: dmemRenderTopbarAction,
  renderDetail: dmemRenderDetail,

  formatters: {
    "mono-time": (v) => dmemShortTs(v),
    "metric": (v) => dmemMetric(v),
  },
});
