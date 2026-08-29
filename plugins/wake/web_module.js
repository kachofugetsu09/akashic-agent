export function activate(ctx) {
  return ctx.ui.inject("workbench.panels.v1", (mount) => mount.register({
    id: "wake-attempts",
    label: "Wake 检查",
    order: 20,
    render(host) {
      const panel = document.createElement("section");
      panel.className = "wake-workbench-panel";
      panel.innerHTML = `<header><h1>Wake 检查</h1><p>查看 Timer 触发后怎样处理 EventMail。</p></header><p data-status role="status" aria-live="polite"></p><div class="wake-panel-grid"><div><div data-list></div><footer><button type="button" data-previous>上一页</button><span data-page></span><button type="button" data-next>下一页</button></footer></div><article data-detail><p>选择一次检查查看详情。</p></article></div>`;
      host.replaceChildren(panel);
      const status = panel.querySelector("[data-status]");
      const list = panel.querySelector("[data-list]");
      const detail = panel.querySelector("[data-detail]");
      const pageText = panel.querySelector("[data-page]");
      const previous = panel.querySelector("[data-previous]");
      const next = panel.querySelector("[data-next]");
      let page = 1;
      let total = 0;
      let listRequest = new AbortController();
      let detailRequest = new AbortController();
      let disposed = false;

      const load = async () => {
        listRequest.abort();
        listRequest = new AbortController();
        const activeRequest = listRequest;
        const requestedPage = page;
        status.textContent = "正在读取…";
        try {
          const result = await json(ctx, `/api/dashboard/wake/attempts?page=${requestedPage}&page_size=25`, activeRequest.signal);
          if (disposed || activeRequest !== listRequest || activeRequest.signal.aborted) return;
          total = Number(result.total);
          list.replaceChildren();
          for (const item of result.items) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "wake-panel-row";
            button.innerHTML = `<strong>${escapeHtml(outcomeText(item.outcome))}</strong><span>${escapeHtml(timeText(item.fired_at))} · ${escapeHtml(ownerText(item.owner))}</span><small>${escapeHtml(item.detail || "没有补充说明")}</small>`;
            button.addEventListener("click", () => void openDetail(item.attempt_id));
            list.append(button);
          }
          const pages = Math.max(1, Math.ceil(total / 25));
          pageText.textContent = `${requestedPage} / ${pages}`;
          previous.disabled = requestedPage <= 1;
          next.disabled = requestedPage >= pages;
          status.textContent = total ? `共 ${total} 次定时检查` : "还没有 Wake 检查记录。";
        } catch (reason) {
          if (!activeRequest.signal.aborted) showError(status, reason);
        }
      };

      const openDetail = async (attemptId) => {
        detailRequest.abort();
        detailRequest = new AbortController();
        const activeRequest = detailRequest;
        detail.innerHTML = "<p>正在读取详情…</p>";
        try {
          const item = await json(ctx, `/api/dashboard/wake/attempts/${encodeURIComponent(attemptId)}`, activeRequest.signal);
          if (disposed || activeRequest !== detailRequest || activeRequest.signal.aborted) return;
          detail.innerHTML = `<header><p>${escapeHtml(ownerText(item.owner))} · ${escapeHtml(timeText(item.fired_at))}</p><h2>${escapeHtml(outcomeText(item.outcome))}</h2></header><dl><div><dt>计划时间</dt><dd>${escapeHtml(timeText(item.scheduled_for))}</dd></div><div><dt>信箱水位</dt><dd>${item.mail_watermark == null ? "未读取" : Number(item.mail_watermark)}</dd></div><div><dt>检查完成</dt><dd>${escapeHtml(timeText(item.completed_at))}</dd></div></dl><section><h3>这次检查</h3><p>${escapeHtml(item.detail || "Timer 已触发，正在检查 EventMail。")}</p><code>${escapeHtml(item.timer_id)}</code></section>`;
        } catch (reason) {
          if (!activeRequest.signal.aborted) showError(detail, reason);
        }
      };

      previous.addEventListener("click", () => { if (page > 1) { page -= 1; void load(); } });
      next.addEventListener("click", () => { if (page * 25 < total) { page += 1; void load(); } });
      void load();
      return () => {
        disposed = true;
        listRequest.abort();
        detailRequest.abort();
        host.replaceChildren();
      };
    },
  }));
}

async function json(ctx, path, signal) {
  const response = await ctx.http.request(path, {signal});
  const body = await response.json();
  if (!response.ok) throw new Error(body?.detail || body?.message || `HTTP ${response.status}`);
  return body;
}

function outcomeText(value) {
  return ({checking: "检查中", no_due: "没有到期信件", content_insufficient: "Content 不足", admission_rejected: "Admission 未通过", shared: "已发送", model_skip: "模型跳过", deferred: "已延期", cancelled_after_fire: "触发后关闭", delivery_unknown: "送达未知", failed: "检查失败"})[value] || String(value || "未知结果");
}

function ownerText(value) {
  return ({alert: "Alert", content: "Content", drift: "Drift"})[value] || "无待办";
}

function timeText(value) {
  if (!value) return "进行中";
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false}).format(date);
}

function showError(target, reason) {
  target.setAttribute("role", "alert");
  target.textContent = reason instanceof Error ? reason.message : String(reason);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character]);
}
