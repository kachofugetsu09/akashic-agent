let d = null;
async function l(e, t) {
  if (!d) throw new Error("Wake 工作台面板未激活");
  const r = await d(e, t), a = await r.json();
  if (!r.ok) throw new Error(String(a.detail ?? a.message ?? `HTTP ${r.status}`));
  return a;
}
function n(e) {
  return String(e).replace(/[&<>"']/g, (t) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[t] ?? t);
}
function i(e) {
  if (!e) return "进行中";
  const t = new Date(String(e));
  return Number.isNaN(t.getTime()) ? String(e) : new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: !1
  }).format(t);
}
function s(e) {
  return e ? { alert: "Alert", content: "Content", drift: "Drift" }[e] : "无待办";
}
function u(e) {
  return e == null ? "未读取" : String(e);
}
function c(e) {
  return {
    checking: "检查中",
    no_due: "没有到期信件",
    content_insufficient: "Content 不足",
    admission_rejected: "Admission 未通过",
    shared: "已发送",
    model_skip: "模型跳过",
    deferred: "已延期",
    cancelled_after_fire: "触发后关闭",
    delivery_unknown: "送达未知",
    failed: "检查失败"
  }[e];
}
function w(e, t) {
  return `
    <article class="wake-run">
      <header class="wake-run-header">
        <div>
          <p>${n(s(e.owner))} · ${n(i(e.fired_at))}</p>
          <h2>${n(c(e.outcome))}</h2>
        </div>
        ${t ? '<md-icon-button data-wake-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}
      </header>

      <dl class="wake-summary">
        <div><dt>计划时间</dt><dd>${n(i(e.scheduled_for))}</dd></div>
        <div><dt>信箱水位</dt><dd>${n(u(e.mail_watermark))}</dd></div>
        <div><dt>检查完成</dt><dd>${n(i(e.completed_at))}</dd></div>
      </dl>

      <section class="wake-section">
        <h3>这次检查</h3>
        <p>${n(e.detail || "Timer 已触发，正在检查 EventMail。")}</p>
        <p><code>${n(e.timer_id)}</code></p>
      </section>
    </article>
  `;
}
const f = {
  id: "wake-attempts",
  label: "Wake 检查",
  viewLabel: "Wake 检查",
  pageSize: 25,
  rowKey: "attempt_id",
  countTitle(e) {
    return `${e} 次定时检查`;
  },
  columns: [
    { key: "fired_at", label: "触发时间", width: 130, renderCell: i },
    { key: "owner", label: "输入", width: 90, renderCell: s },
    { key: "mail_watermark", label: "信箱水位", width: 90, renderCell: u },
    { key: "outcome", label: "结果", width: 120, renderCell: c },
    { key: "detail", label: "说明", flex: !0, fmt: "text-preview" }
  ],
  async getCount({ signal: e }) {
    return (await l("/api/dashboard/wake/attempts?page=1&page_size=1", { signal: e })).total;
  },
  async fetchPage({ page: e, pageSize: t, signal: r }) {
    return await l(
      `/api/dashboard/wake/attempts?page=${e}&page_size=${t}`,
      { signal: r }
    );
  },
  async fetchDetail(e, { signal: t }) {
    return l(`/api/dashboard/wake/attempts/${encodeURIComponent(String(e.attempt_id ?? ""))}`, { signal: t });
  },
  renderDetail(e, t, r) {
    var a;
    if (!e) {
      t.innerHTML = '<p class="wake-empty">选择一次定时检查，查看它当时看到的 EventMail 水位和结果。</p>';
      return;
    }
    t.innerHTML = w(
      e,
      r == null ? void 0 : r.closePane
    ), (a = t.querySelector("[data-wake-close]")) == null || a.addEventListener("click", () => {
      var o;
      return (o = r == null ? void 0 : r.closePane) == null ? void 0 : o.call(r);
    });
  }
};
function m(e) {
  d = e.http.request;
  const t = e.ui.inject("workbench.panels.v2", (r) => r.register(f));
  return () => {
    t(), d = null;
  };
}
export {
  m as activate
};
