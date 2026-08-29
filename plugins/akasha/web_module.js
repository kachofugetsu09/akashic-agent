export function activate(ctx) {
  return ctx.ui.inject("workbench.panels.v1", (mount) => mount.register({
    id: "akasha-inspector",
    label: "Akasha 检索",
    order: 10,
    render(host) {
      const panel = document.createElement("section");
      panel.className = "akasha-workbench-panel";
      panel.innerHTML = `<header><div><h1>Akasha 检索</h1><p>查看每次回答使用了哪些记忆。</p></div><label>搜索<input type="search" placeholder="问题、回复或会话"></label></header><p data-status role="status" aria-live="polite"></p><div class="akasha-panel-grid"><div><div data-list></div><footer><button type="button" data-previous>上一页</button><span data-page></span><button type="button" data-next>下一页</button></footer></div><article data-detail><p>选择一轮检索查看详情。</p></article></div>`;
      host.replaceChildren(panel);
      const search = panel.querySelector("input");
      const status = panel.querySelector("[data-status]");
      const list = panel.querySelector("[data-list]");
      const detail = panel.querySelector("[data-detail]");
      const pageText = panel.querySelector("[data-page]");
      const previous = panel.querySelector("[data-previous]");
      const next = panel.querySelector("[data-next]");
      let page = 1;
      let total = 0;
      let timer = 0;
      let listRequest = new AbortController();
      let detailRequest = new AbortController();
      let disposed = false;

      const load = async () => {
        listRequest.abort();
        listRequest = new AbortController();
        const activeRequest = listRequest;
        status.textContent = "正在读取…";
        const params = new URLSearchParams({page: String(page), page_size: "25"});
        if (search.value.trim()) params.set("q", search.value.trim());
        try {
          const result = await json(ctx, `/api/dashboard/akasha-inspector/turns?${params}`, activeRequest.signal);
          if (disposed || activeRequest !== listRequest || activeRequest.signal.aborted) return;
          total = Number(result.total);
          renderRows(result.items);
          const pages = Math.max(1, Math.ceil(total / 25));
          pageText.textContent = `${page} / ${pages}`;
          previous.disabled = page <= 1;
          next.disabled = page >= pages;
          status.textContent = total ? `共 ${total} 轮检索` : "没有符合条件的检索记录。";
        } catch (reason) {
          if (!activeRequest.signal.aborted) showError(status, reason);
        }
      };

      const renderRows = (items) => {
        list.replaceChildren();
        for (const item of items) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "akasha-panel-row";
          button.innerHTML = `<strong>${escapeHtml(item.query_text || "（空问题）")}</strong><span>${escapeHtml(shortTime(item.ts))} · ${Number(item.seed_count || 0)} 条线索 · ${Number(item.completion_count || 0)} 条召回</span>`;
          button.addEventListener("click", () => openDetail(item.query_id));
          list.append(button);
        }
      };

      const openDetail = async (queryId) => {
        detailRequest.abort();
        detailRequest = new AbortController();
        const activeRequest = detailRequest;
        detail.innerHTML = "<p>正在读取详情…</p>";
        try {
          const item = await json(ctx, `/api/dashboard/akasha-inspector/turns/${encodeURIComponent(queryId)}`, activeRequest.signal);
          if (disposed || activeRequest !== detailRequest || activeRequest.signal.aborted) return;
          detail.innerHTML = renderDetail(item);
          const lanes = [...detail.querySelectorAll("details[data-lane]")];
          for (const lane of lanes) lane.addEventListener("toggle", () => {
            if (lane.open) for (const sibling of lanes) if (sibling !== lane) sibling.open = false;
          });
        } catch (reason) {
          if (!activeRequest.signal.aborted) showError(detail, reason);
        }
      };

      search.addEventListener("input", () => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => { page = 1; void load(); }, 200);
      });
      previous.addEventListener("click", () => { if (page > 1) { page -= 1; void load(); } });
      next.addEventListener("click", () => { if (page * 25 < total) { page += 1; void load(); } });

      const overviewRequest = listRequest;
      void json(ctx, "/api/dashboard/akasha-inspector/overview", overviewRequest.signal)
        .then((overview) => {
          if (disposed) return;
          if (!overview.available) {
            status.textContent = "Akasha Inspector 当前不可用。";
            list.replaceChildren();
            return;
          }
          void load();
        })
        .catch((reason) => { if (!overviewRequest.signal.aborted) showError(status, reason); });

      return () => {
        disposed = true;
        window.clearTimeout(timer);
        listRequest.abort();
        detailRequest.abort();
        host.replaceChildren();
      };
    },
  }));
}

function renderDetail(item) {
  const lanes = [
    ["直接线索", item.seeds],
    ["图扩散候选", item.activation_items],
    ["精确回忆", item.left],
    ["模式联想", item.right],
    ["工具精确回忆", item.tool_left],
    ["工具模式联想", item.tool_right],
  ].filter(([, values]) => Array.isArray(values) && values.length);
  return `<header><h2>${escapeHtml(item.query_text || "（空问题）")}</h2><p>${escapeHtml(shortTime(item.ts))} · seq ${Number(item.seq || 0)} · ${escapeHtml(item.session_key || "未知会话")}</p></header><dl class="akasha-panel-metrics"><div><dt>直接线索</dt><dd>${Number(item.seed_count || 0)}</dd></div><div><dt>精确回忆</dt><dd>${Number(item.left_count || 0)}</dd></div><div><dt>模式联想</dt><dd>${item.recall_capture_available ? Number(item.right_count || 0) : "未记录"}</dd></div></dl><details><summary>助手回复</summary><p>${escapeHtml(item.assistant_text || "（没有文本回复）")}</p></details>${lanes.map(([title, values]) => `<details data-lane><summary>${title}<span>${values.length}</span></summary><ol>${values.map((value) => `<li><p>${escapeHtml(value.user_text || "（空消息）")}</p>${value.assistant_preview ? `<small>${escapeHtml(value.assistant_preview)}</small>` : ""}</li>`).join("")}</ol></details>`).join("")}<details><summary>学习变化与技术指标</summary><dl><div><dt>扩散候选 / 次数</dt><dd>${Number(item.activation_count || 0)} / ${Number(item.pushes || 0)}</dd></div><div><dt>惊喜度</dt><dd>${numberText(item.surprise)}</dd></div><div><dt>增强 / 抑制</dt><dd>${numberText(item.potentiated_mass)} / ${numberText(item.inhibited_mass)}</dd></div></dl></details><details><summary>写入 Prompt 的记忆</summary><pre>${escapeHtml(item.text_block_preview || "这一轮没有注入记忆。")}</pre></details>`;
}

function numberText(value) {
  const number = Number(value);
  return value == null || !Number.isFinite(number) ? "—" : number.toFixed(3);
}

async function json(ctx, path, signal) {
  const response = await ctx.http.request(path, {signal});
  const body = await response.json();
  if (!response.ok) throw new Error(body?.detail || body?.message || `HTTP ${response.status}`);
  return body;
}

function showError(target, reason) {
  target.setAttribute("role", "alert");
  target.textContent = reason instanceof Error ? reason.message : String(reason);
}

function shortTime(value) {
  const date = new Date(String(value || ""));
  return Number.isNaN(date.getTime()) ? "—" : new Intl.DateTimeFormat("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false}).format(date);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character]);
}
