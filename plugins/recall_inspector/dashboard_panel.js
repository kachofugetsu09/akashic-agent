(function () {
  const _state = {
    items: [],
    activeId: null,
    activeDetail: null,
  };

  window.AkashicDashboard.registerPlugin({
    id: "recall_inspector",
    label: "Recall Inspector",
    viewLabel: "recall inspector",
    pageSize: 25,

    countTitle(total) {
      return `${total} 轮召回`;
    },

    async getCount() {
      try {
        const r = await api("/api/dashboard/recall-inspector/overview");
        return r.available ? (r.total || 0) : null;
      } catch (_e) {
        return null;
      }
    },

    async loadPanel({ page, pageSize }) {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("page_size", String(pageSize));
      const data = await api(`/api/dashboard/recall-inspector/turns?${params.toString()}`);
      _state.items = data.items || [];
      const total = data.total || 0;
      if (_state.activeId && !_state.items.find((i) => i.turn_id === _state.activeId)) {
        _state.activeId = null;
        _state.activeDetail = null;
      }
      return { total };
    },

    renderTableHead(container) {
      container.className = "table-head mode-recall";
      container.innerHTML = `
        <div>Session</div>
        <div>Time</div>
        <div>User</div>
        <div>Prepare</div>
        <div>Recall</div>
      `;
    },

    renderRows({ bodyEl, batchBarEl, metaEl, pageTextEl, prevBtn, nextBtn, page, total, pageSize, pageCount }) {
      batchBarEl.classList.add("hidden");
      bodyEl.innerHTML = "";
      if (!_state.items.length) {
        bodyEl.innerHTML = '<div class="empty-state">还没有 recall_inspector 记录。</div>';
      }
      _state.items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "table-row mode-recall";
        if (_state.activeId === item.turn_id) {
          row.classList.add("active");
        }
        row.innerHTML = `
          <div class="mono cell-session" title="${escapeHtml(item.session_key || "")}">${escapeHtml(formatSessionKeyForTable(item.session_key || ""))}</div>
          <div class="mono cell-time" title="${escapeHtml(item.timestamp || "")}">${escapeHtml(shortTs(item.timestamp || ""))}</div>
          <div class="content-preview">${escapeHtml(stripMarkdown(item.user_text || ""))}</div>
          <div class="mono cell-metric">${escapeHtml(String(item.context_prepare_count || 0))}</div>
          <div class="mono cell-metric">${escapeHtml(String(item.recall_memory_count || 0))}</div>
        `;
        row.addEventListener("click", async () => {
          _state.activeId = item.turn_id;
          _state.activeDetail = await api(
            `/api/dashboard/recall-inspector/turns/${encodePath(item.turn_id)}`
          );
          render();
        });
        bodyEl.appendChild(row);
      });
      metaEl.textContent = `共 ${total} 轮召回记录`;
      pageTextEl.textContent = `${page} / ${pageCount}`;
      prevBtn.disabled = page <= 1;
      nextBtn.disabled = page >= pageCount;
    },

    renderDetail(container) {
      if (!_state.activeDetail) {
        container.innerHTML = `
          <div class="detail-empty">
            <div class="detail-empty-title">Recall Inspector</div>
            <div class="detail-empty-text">点开一轮记录后，这里会显示 context prepare 和 recall_memory 召回的记忆。</div>
          </div>
        `;
        return;
      }
      const item = _state.activeDetail;
      const contextPrepare = item.context_prepare || {};
      const contextItems = contextPrepare.items || [];
      const injectedItems = contextPrepare.injected_items || [];
      const recallCalls = Array.isArray(item.recall_memory_calls) ? item.recall_memory_calls : [];
      container.innerHTML = `
        <div class="detail-wrap">
          <div class="detail-toolbar">
            <div>
              <div class="detail-title">召回记录</div>
              <div class="detail-subtext">${escapeHtml(item.session_key || "")} · ${escapeHtml(item.turn_id || "")}</div>
            </div>
          </div>
          <div class="detail-block">
            <div class="detail-label">User Message</div>
            <div class="detail-content">${renderMarkdown(item.user_text || "")}</div>
          </div>
          <div class="detail-block">
            <div class="detail-label">预检索总召回</div>
            ${_renderRecallItems(contextItems, "context")}
          </div>
          <div class="detail-block">
            <div class="detail-label">最终注入</div>
            ${_renderRecallItems(injectedItems, "inject")}
          </div>
          <div class="detail-block">
            <div class="detail-label">Recall 返回</div>
            ${
              recallCalls.length
                ? recallCalls.map((call) => _renderRecallItems(call.items || [], "recall")).join("")
                : '<div class="muted-text">本轮没有显式调用 recall_memory。</div>'
            }
          </div>
        </div>
      `;
    },
  });

  function _renderRecallItems(items, source) {
    if (!Array.isArray(items) || !items.length) {
      return '<div class="muted-text">没有召回条目。</div>';
    }
    return `
      <div class="recall-item-list">
        ${items
          .map(
            (item) => `
          <div class="recall-item recall-item-${escapeHtml(source || "unknown")}">
            <div class="recall-item-head">
              <code>${escapeHtml(item.id || "-")}</code>
              ${item.injected === true ? '<span class="recall-tag recall-tag-injected">已注入</span>' : ""}
              ${_renderRecallTags(item.tags || [])}
            </div>
            <div class="recall-summary">${escapeHtml(item.summary || "")}</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  }

  function _renderRecallTags(tags) {
    if (!Array.isArray(tags) || !tags.length) {
      return "";
    }
    return tags.map((tag) => `<span class="recall-tag">${escapeHtml(tag)}</span>`).join("");
  }
})();
